#!/usr/bin/env python
"""Compare full-sequence VGGT-Omega poses against 4+1 sliding-window poses."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


DEFAULT_RGB_DIR = "/home/ubuntu/vggt-omega/public_data/tum/rgbd_dataset_freiburg2_pioneer_360/rgb"
DEFAULT_CHECKPOINT = "/home/ubuntu/vggt-omega/checkpoints/VGGT-Omega-1B-512/model.pt"
DEFAULT_OUTPUT_DIR = "/home/ubuntu/vggt-omega/outputs/tum_pioneer_360_full_vs_sliding"


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(Path(args.rgb_dir))
    start_index = args.start_index - 1 if args.one_based else args.start_index
    end_index = args.end_index if args.one_based else args.end_index
    start_index = max(0, start_index)
    end_index = min(len(image_paths), end_index)
    if end_index <= start_index:
        raise ValueError(f"Empty frame range after clamping: start={start_index}, end={end_index}, total={len(image_paths)}")

    selected_paths = image_paths[start_index:end_index]
    print(f"Total images: {len(image_paths)}")
    print(f"Selected zero-based half-open range: [{start_index}, {end_index}) -> {len(selected_paths)} frames")
    print(f"First selected: {selected_paths[0]}")
    print(f"Last selected:  {selected_paths[-1]}")

    model = load_camera_model(args.checkpoint, args.device)
    images = load_and_preprocess_images(selected_paths, image_resolution=args.image_resolution).to(args.device)
    print(f"Preprocessed tensor: {tuple(images.shape)}")

    with torch.inference_mode():
        print("Running full-sequence VGGT-Omega camera inference...")
        full_extrinsics = run_camera_window(model, images)

        print(
            f"Running sliding-window inference with window_size={args.window_size}, "
            f"alignment={args.window_alignment}..."
        )
        sliding_extrinsics, window_scales, sliding_diagnostics = run_sliding_camera(
            model,
            images,
            args.window_size,
            args.window_alignment,
        )

    full_centers = camera_centers(full_extrinsics)
    sliding_centers_raw = camera_centers(sliding_extrinsics)
    full_from_sliding = estimate_sim3_umeyama(sliding_centers_raw, full_centers)
    sliding_centers_aligned = transform_points_sim3(full_from_sliding, sliding_centers_raw)

    save_outputs(
        output_dir=output_dir,
        selected_paths=selected_paths,
        full_extrinsics=full_extrinsics,
        sliding_extrinsics=sliding_extrinsics,
        full_centers=full_centers,
        sliding_centers_raw=sliding_centers_raw,
        sliding_centers_aligned=sliding_centers_aligned,
        full_from_sliding=full_from_sliding,
        window_scales=window_scales,
        sliding_diagnostics=sliding_diagnostics,
        args=args,
        start_index=start_index,
        end_index=end_index,
        total_images=len(image_paths),
    )
    print(f"Saved comparison to: {output_dir}")
    print(f"Plot: {output_dir / 'trajectory_comparison.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", default=DEFAULT_RGB_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-index", type=int, default=500)
    parser.add_argument("--end-index", type=int, default=1500)
    index_group = parser.add_mutually_exclusive_group()
    index_group.add_argument(
        "--one-based",
        dest="one_based",
        action="store_true",
        help="Interpret start/end as human 1-based inclusive indices.",
    )
    index_group.add_argument(
        "--zero-based",
        dest="one_based",
        action="store_false",
        help="Interpret start/end as Python-style zero-based [start, end).",
    )
    parser.set_defaults(one_based=True)
    parser.add_argument("--window-size", type=int, default=5, help="Use 4+1 when this is 5.")
    parser.add_argument("--window-alignment", default="centers", choices=["centers", "extrinsics"])
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def list_images(rgb_dir: Path) -> list[str]:
    suffixes = {".png", ".jpg", ".jpeg"}
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    paths = [str(path) for path in sorted(rgb_dir.iterdir()) if path.suffix.lower() in suffixes]
    if not paths:
        raise ValueError(f"No RGB images found in {rgb_dir}")
    return paths


def load_camera_model(checkpoint: str, device: str) -> VGGTOmega:
    model = VGGTOmega(enable_depth=False).eval()
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    checkpoint_state = torch.load(checkpoint, map_location="cpu")
    model_state = model.state_dict()
    filtered_state = {key: value for key, value in checkpoint_state.items() if key in model_state}
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    missing_non_depth = [key for key in missing if not key.startswith("dense_head.")]
    if missing_non_depth:
        raise RuntimeError(f"Missing non-depth checkpoint keys: {missing_non_depth[:8]}")
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys after filtering: {unexpected[:8]}")
    print(f"Loaded camera-only model from {checkpoint}; skipped {len(checkpoint_state) - len(filtered_state)} depth/alignment keys.")
    return model.to(device)


def run_camera_window(model: VGGTOmega, images: torch.Tensor) -> np.ndarray:
    predictions = model(images)
    extrinsic, _ = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
    extrinsic = extrinsic.detach().float().cpu().numpy()
    if extrinsic.shape[0] == 1:
        extrinsic = extrinsic[0]
    return extrinsic.astype(np.float32)


def run_sliding_camera(
    model: VGGTOmega,
    images: torch.Tensor,
    window_size: int,
    window_alignment: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if window_size <= 1:
        raise ValueError("window_size must be greater than 1.")
    num_frames = int(images.shape[0])
    if window_size > num_frames:
        window_size = num_frames

    all_extrinsics: list[np.ndarray | None] = [None] * num_frames
    window_scales: list[float] = [1.0]
    fallback_count = 0

    first_local = run_camera_window(model, images[:window_size])
    for idx in range(window_size):
        all_extrinsics[idx] = first_local[idx]

    for candidate_idx in range(window_size, num_frames):
        overlap_indices = list(range(candidate_idx - (window_size - 1), candidate_idx))
        window_indices = overlap_indices + [candidate_idx]
        local = run_camera_window(model, images[window_indices])

        local_overlap = local[: window_size - 1]
        global_overlap = np.stack([all_extrinsics[idx] for idx in overlap_indices], axis=0)
        if window_alignment == "extrinsics":
            try:
                global_from_window = estimate_sim3_from_extrinsics(local_overlap, global_overlap)
            except ValueError as exc:
                fallback_count += 1
                print(f"  extrinsics alignment fallback at frame {candidate_idx}: {exc}")
                global_from_window = estimate_sim3_umeyama(camera_centers(local_overlap), camera_centers(global_overlap))
        elif window_alignment == "centers":
            global_from_window = estimate_sim3_umeyama(camera_centers(local_overlap), camera_centers(global_overlap))
        else:
            raise ValueError(f"Unknown window_alignment: {window_alignment}")
        rebased = transform_extrinsics_sim3(local, global_from_window)
        all_extrinsics[candidate_idx] = rebased[-1]
        window_scales.append(float(np.cbrt(max(np.linalg.det(global_from_window[:3, :3]), 1e-12))))

        if candidate_idx % 50 == 0 or candidate_idx == num_frames - 1:
            print(f"  sliding {candidate_idx + 1}/{num_frames}")

    diagnostics = {
        "alignment_fallback_count": fallback_count,
    }
    return np.stack(all_extrinsics, axis=0).astype(np.float32), np.asarray(window_scales, dtype=np.float32), diagnostics


def camera_center(extrinsic: np.ndarray | None) -> np.ndarray:
    if extrinsic is None:
        raise ValueError("Missing extrinsic.")
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]
    return (-rotation.T @ translation).astype(np.float64)


def camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    return np.stack([camera_center(extrinsic) for extrinsic in extrinsics], axis=0)


def transform_extrinsics_sim3(extrinsics: np.ndarray, transform: np.ndarray) -> np.ndarray:
    scale, rotation, translation = split_sim3(transform)
    out = []
    for extrinsic in extrinsics:
        local_rotation = extrinsic[:3, :3].astype(np.float64)
        local_translation = extrinsic[:3, 3].astype(np.float64)
        global_rotation = local_rotation @ rotation.T
        global_translation = scale * local_translation - global_rotation @ translation
        out.append(np.concatenate([global_rotation, global_translation[:, None]], axis=1))
    return np.stack(out, axis=0).astype(np.float32)


def split_sim3(transform: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    scaled_rotation = transform[:3, :3].astype(np.float64)
    scale = float(np.cbrt(max(np.linalg.det(scaled_rotation), 1e-12)))
    rotation = scaled_rotation / scale
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = transform[:3, 3].astype(np.float64)
    return scale, rotation, translation


def estimate_sim3_umeyama(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source.astype(np.float64)
    target = target.astype(np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"source and target must both have shape [N, 3], got {source.shape} and {target.shape}")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_var = np.mean(np.sum(source_centered * source_centered, axis=1))
    if source_var <= 1e-12:
        raise ValueError("Cannot estimate Sim(3) scale from nearly identical source points.")

    covariance = (target_centered.T @ source_centered) / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    scale = float(np.sum(singular_values * sign) / source_var)
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError(f"Estimated invalid Sim(3) scale: {scale}")
    translation = target_mean - scale * rotation @ source_mean

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform.astype(np.float32)


def estimate_sim3_from_extrinsics(local_extrinsics: np.ndarray, target_extrinsics: np.ndarray) -> np.ndarray:
    local_extrinsics = np.asarray(local_extrinsics, dtype=np.float64)
    target_extrinsics = np.asarray(target_extrinsics, dtype=np.float64)
    if (
        local_extrinsics.shape != target_extrinsics.shape
        or local_extrinsics.ndim != 3
        or local_extrinsics.shape[1:] != (3, 4)
    ):
        raise ValueError(
            "local_extrinsics and target_extrinsics must both have shape [N, 3, 4], got "
            f"{local_extrinsics.shape} and {target_extrinsics.shape}"
        )
    if local_extrinsics.shape[0] < 2:
        raise ValueError(f"At least two extrinsics are required to estimate Sim(3); got {local_extrinsics.shape[0]}.")

    local_rotations = local_extrinsics[:, :3, :3]
    target_rotations = target_extrinsics[:, :3, :3]
    world_rotations = np.einsum("nij,njk->nik", np.swapaxes(target_rotations, 1, 2), local_rotations)
    rotation = project_rotation_matrix(world_rotations.mean(axis=0))

    local_centers = camera_centers(local_extrinsics)
    target_centers = camera_centers(target_extrinsics)
    rotated_local_centers = local_centers @ rotation.T

    source_mean = rotated_local_centers.mean(axis=0)
    target_mean = target_centers.mean(axis=0)
    source_centered = rotated_local_centers - source_mean
    target_centered = target_centers - target_mean
    source_energy = float(np.sum(source_centered * source_centered))
    if source_energy <= 1e-12:
        raise ValueError("Cannot estimate Sim(3) scale from nearly identical overlap camera centers.")
    scale = float(np.sum(source_centered * target_centered) / source_energy)
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError(f"Estimated invalid Sim(3) scale: {scale}")
    translation = target_mean - scale * source_mean

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform.astype(np.float32)


def project_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix.astype(np.float64))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def transform_points_sim3(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def save_outputs(
    *,
    output_dir: Path,
    selected_paths: list[str],
    full_extrinsics: np.ndarray,
    sliding_extrinsics: np.ndarray,
    full_centers: np.ndarray,
    sliding_centers_raw: np.ndarray,
    sliding_centers_aligned: np.ndarray,
    full_from_sliding: np.ndarray,
    window_scales: np.ndarray,
    sliding_diagnostics: dict[str, int],
    args: argparse.Namespace,
    start_index: int,
    end_index: int,
    total_images: int,
) -> None:
    np.savez_compressed(
        output_dir / "trajectory_comparison.npz",
        full_extrinsics=full_extrinsics,
        sliding_extrinsics=sliding_extrinsics,
        full_centers=full_centers,
        sliding_centers_raw=sliding_centers_raw,
        sliding_centers_aligned=sliding_centers_aligned,
        full_from_sliding=full_from_sliding,
        window_scales=window_scales,
        image_paths=np.asarray(selected_paths),
    )
    metadata = {
        "rgb_dir": args.rgb_dir,
        "checkpoint": args.checkpoint,
        "image_resolution": args.image_resolution,
        "window_size": args.window_size,
        "window_alignment": args.window_alignment,
        "requested_start_index": args.start_index,
        "requested_end_index": args.end_index,
        "one_based": args.one_based,
        "actual_zero_based_range": [start_index, end_index],
        "selected_frames": len(selected_paths),
        "total_images": total_images,
        "first_selected": selected_paths[0],
        "last_selected": selected_paths[-1],
        "full_from_sliding": full_from_sliding.tolist(),
        "window_scale_min": float(np.min(window_scales)),
        "window_scale_max": float(np.max(window_scales)),
        "window_scale_mean": float(np.mean(window_scales)),
        "alignment_fallback_count": int(sliding_diagnostics["alignment_fallback_count"]),
    }
    center_errors = np.linalg.norm(full_centers - sliding_centers_aligned, axis=1)
    metadata.update(
        {
            "aligned_center_error_rmse": float(np.sqrt(np.mean(center_errors**2))),
            "aligned_center_error_mean": float(np.mean(center_errors)),
            "aligned_center_error_median": float(np.median(center_errors)),
            "aligned_center_error_p95": float(np.percentile(center_errors, 95)),
            "aligned_center_error_max": float(np.max(center_errors)),
        }
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_trajectories(output_dir / "trajectory_comparison.png", full_centers, sliding_centers_raw, sliding_centers_aligned)


def plot_trajectories(
    output_path: Path,
    full_centers: np.ndarray,
    sliding_centers_raw: np.ndarray,
    sliding_centers_aligned: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    ax_xy = fig.add_subplot(1, 2, 1)
    ax_xz = fig.add_subplot(1, 2, 2)

    plot_2d(ax_xy, full_centers[:, 0], full_centers[:, 1], "Full VGGT-Omega", "tab:blue")
    plot_2d(ax_xy, sliding_centers_aligned[:, 0], sliding_centers_aligned[:, 1], "Sliding 4+1 aligned", "tab:orange")
    ax_xy.scatter(full_centers[0, 0], full_centers[0, 1], c="tab:green", s=35, label="start")
    ax_xy.scatter(full_centers[-1, 0], full_centers[-1, 1], c="tab:red", s=35, label="end")
    ax_xy.set_title("Top view: x/y")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=0.25)
    ax_xy.legend()

    plot_2d(ax_xz, full_centers[:, 0], full_centers[:, 2], "Full VGGT-Omega", "tab:blue")
    plot_2d(ax_xz, sliding_centers_aligned[:, 0], sliding_centers_aligned[:, 2], "Sliding 4+1 aligned", "tab:orange")
    ax_xz.plot(sliding_centers_raw[:, 0], sliding_centers_raw[:, 2], color="0.6", linewidth=1.0, alpha=0.45, label="Sliding raw")
    ax_xz.set_title("Side view: x/z")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xz.axis("equal")
    ax_xz.grid(True, alpha=0.25)
    ax_xz.legend()

    fig.suptitle("VGGT-Omega Full Sequence vs Sliding Window Camera Trajectory")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_2d(ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str, color: str) -> None:
    ax.plot(x, y, color=color, linewidth=1.6, label=label)
    step = max(1, len(x) // 20)
    ax.scatter(x[::step], y[::step], color=color, s=8, alpha=0.75)


if __name__ == "__main__":
    main()

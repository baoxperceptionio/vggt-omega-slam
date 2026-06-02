#!/usr/bin/env python3

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_fixed_pose_kv_slam import (
    _apply_global_transform,
    _camera_center,
    _estimate_ground_normalization,
    _estimate_sim3_umeyama,
    _load_or_initialize_model,
    _store_pose_only,
    _transform_points_sim3,
    run_fixed_pose_kv_slam,
    run_fixed_pose_window_slam,
)
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.utils.slam import unproject_depth_map_to_point_map_torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full, old sliding-window, and fixed-pose KV SLAM.")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--checkpoint", default="checkpoints/VGGT-Omega-1B-512/model.pt")
    parser.add_argument(
        "--fixed-checkpoint",
        help="Optional CausalVGGTOmega/fixed-pose checkpoint for the fixed-pose KV runner. Defaults to --checkpoint.",
    )
    parser.add_argument("--output-dir", default="outputs/slam_mode_comparison")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--allow-random-weights", action="store_true")
    args = parser.parse_args()

    image_paths = _expand_images(args.images)
    if not image_paths:
        raise ValueError("No input images were provided.")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full = run_full_vggt(
        image_paths,
        checkpoint=args.checkpoint,
        image_resolution=args.image_resolution,
        device=args.device,
        allow_random_weights=args.allow_random_weights,
    )
    np.savez_compressed(out_dir / "full_vggt.npz", **full, image_paths=np.array(image_paths))

    sliding = run_old_sliding_window(
        image_paths,
        checkpoint=args.checkpoint,
        image_resolution=args.image_resolution,
        window_size=args.window_size,
        device=args.device,
        allow_random_weights=args.allow_random_weights,
    )
    np.savez_compressed(out_dir / "old_sliding_window.npz", **sliding, image_paths=np.array(image_paths))

    _, fixed_state, fixed_ply = run_fixed_pose_window_slam(
        image_paths,
        checkpoint=args.fixed_checkpoint or args.checkpoint,
        output_dir=str(out_dir / "fixed_pose_kv"),
        window_size=args.window_size,
        image_resolution=args.image_resolution,
        displacement_threshold=0.1,
        max_points=300000,
        conf_percentile=20.0,
        device=args.device,
        allow_random_weights=args.allow_random_weights,
    )
    fixed_npz = np.load(out_dir / "fixed_pose_kv" / "fixed_pose_kv_slam_predictions.npz", allow_pickle=True)
    fixed = {key: fixed_npz[key] for key in fixed_npz.files}

    metrics = compare_trajectories(full, sliding, fixed)
    np.savez_compressed(out_dir / "comparison_metrics.npz", **metrics)
    plot_comparison(out_dir, full, sliding, fixed, metrics)

    print(f"Saved comparison directory: {out_dir}")
    print(f"Fixed-pose PLY: {fixed_ply}")
    print(f"Fixed-pose state: {fixed_state}")
    for name in ("old_sliding", "fixed_pose_kv"):
        err = metrics[f"{name}_error"]
        print(
            f"{name}: mean={float(err.mean()):.6f}, median={float(np.median(err)):.6f}, "
            f"rmse={float(np.sqrt(np.mean(err**2))):.6f}, max={float(err.max()):.6f}"
        )


def run_full_vggt(
    image_paths: list[str],
    *,
    checkpoint: str,
    image_resolution: int,
    device: str,
    allow_random_weights: bool,
) -> dict[str, np.ndarray]:
    model = VGGTOmega().eval()
    _load_or_initialize_model(model, checkpoint, allow_random_weights)
    model = model.to(device)
    images = load_and_preprocess_images(image_paths, image_resolution=image_resolution).to(device)
    with torch.inference_mode():
        pred = model(images)
        extrinsic, intrinsic = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
        world_points = unproject_depth_map_to_point_map_torch(pred["depth"], extrinsic, intrinsic)
    out = _tensor_dict_to_numpy(
        {
            "pose_enc": pred["pose_enc"],
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "depth": pred["depth"],
            "depth_conf": pred["depth_conf"],
            "images": pred["images"],
            "world_points_from_depth": world_points,
        }
    )
    del model, images, pred
    torch.cuda.empty_cache()
    return out


def run_old_sliding_window(
    image_paths: list[str],
    *,
    checkpoint: str,
    image_resolution: int,
    window_size: int,
    device: str,
    allow_random_weights: bool,
) -> dict[str, np.ndarray]:
    if window_size <= 1:
        raise ValueError("window_size must be greater than 1 for old sliding-window mode.")
    model = VGGTOmega().eval()
    _load_or_initialize_model(model, checkpoint, allow_random_weights)
    model = model.to(device)
    images = load_and_preprocess_images(image_paths, image_resolution=image_resolution).to(device)
    num_frames = int(images.shape[0])
    window_size = min(window_size, num_frames)

    all_pose_enc: list[np.ndarray | None] = [None] * num_frames
    all_extrinsic: list[np.ndarray | None] = [None] * num_frames
    all_intrinsic: list[np.ndarray | None] = [None] * num_frames
    global_extrinsics: list[np.ndarray | None] = [None] * num_frames
    accepted_indices: list[int] = []
    alignments: list[np.ndarray] = []

    with torch.inference_mode():
        initial_local = _run_full_window(model, images[:window_size])
        ground_transform, _ = _estimate_ground_normalization(initial_local)
        initial_rebased = _apply_global_transform(initial_local, ground_transform)
        alignments.append(ground_transform)
        for local_idx in range(window_size):
            global_idx = local_idx
            _store_pose_only(all_pose_enc, all_extrinsic, all_intrinsic, initial_rebased, initial_local, local_idx, global_idx)
            global_extrinsics[global_idx] = initial_rebased["extrinsic"][local_idx]
            accepted_indices.append(global_idx)

        for candidate_idx in range(window_size, num_frames):
            overlap_indices = accepted_indices[-(window_size - 1) :]
            window_indices = overlap_indices + [candidate_idx]
            local = _run_full_window(model, images[window_indices])
            global_from_window = _estimate_global_from_window_transform_for_indices(
                local["extrinsic"],
                global_extrinsics,
                window_indices,
                overlap_count=len(overlap_indices),
            )
            rebased = _apply_global_transform(local, global_from_window)
            alignments.append(global_from_window)
            candidate_local_idx = len(window_indices) - 1
            _store_pose_only(all_pose_enc, all_extrinsic, all_intrinsic, rebased, local, candidate_local_idx, candidate_idx)
            global_extrinsics[candidate_idx] = rebased["extrinsic"][candidate_local_idx]
            accepted_indices.append(candidate_idx)

    out = {
        "pose_enc": np.stack(all_pose_enc, axis=0),
        "extrinsic": np.stack(all_extrinsic, axis=0),
        "intrinsic": np.stack(all_intrinsic, axis=0),
        "global_from_window": np.stack(alignments, axis=0),
    }
    del model, images
    torch.cuda.empty_cache()
    return out


def compare_trajectories(full: dict, sliding: dict, fixed: dict) -> dict[str, np.ndarray]:
    full_centers = _camera_centers(full["extrinsic"])
    sliding_centers = _camera_centers(sliding["extrinsic"])
    fixed_centers = _camera_centers(fixed["extrinsic"])
    sliding_aligned, sliding_transform = _align_centers_to_full(sliding_centers, full_centers)
    fixed_aligned, fixed_transform = _align_centers_to_full(fixed_centers, full_centers)
    return {
        "full_centers": full_centers,
        "old_sliding_centers": sliding_centers,
        "fixed_pose_kv_centers": fixed_centers,
        "old_sliding_aligned": sliding_aligned,
        "fixed_pose_kv_aligned": fixed_aligned,
        "old_sliding_error": np.linalg.norm(sliding_aligned - full_centers, axis=1),
        "fixed_pose_kv_error": np.linalg.norm(fixed_aligned - full_centers, axis=1),
        "old_sliding_to_full_sim3": sliding_transform,
        "fixed_pose_kv_to_full_sim3": fixed_transform,
    }


def plot_comparison(out_dir: Path, full: dict, sliding: dict, fixed: dict, metrics: dict[str, np.ndarray]) -> None:
    full_centers = metrics["full_centers"]
    sliding_aligned = metrics["old_sliding_aligned"]
    fixed_aligned = metrics["fixed_pose_kv_aligned"]
    sliding_error = metrics["old_sliding_error"]
    fixed_error = metrics["fixed_pose_kv_error"]
    frames = np.arange(len(full_centers))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(full_centers[:, 0], full_centers[:, 2], label="full VGGT-Omega", linewidth=2)
    axes[0].plot(sliding_aligned[:, 0], sliding_aligned[:, 2], label="old sliding window", linewidth=1.5)
    axes[0].plot(fixed_aligned[:, 0], fixed_aligned[:, 2], label="fixed-pose KV SLAM", linewidth=1.5)
    axes[0].scatter(full_centers[0, 0], full_centers[0, 2], marker="o", color="black", s=32, label="start")
    axes[0].set_title("Trajectory aligned to full VGGT-Omega (X/Z)")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Z")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(frames, sliding_error, label="old sliding window")
    axes[1].plot(frames, fixed_error, label="fixed-pose KV SLAM")
    axes[1].set_title("ATE vs full VGGT-Omega after Sim(3) alignment")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("position error")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_and_error.png", dpi=180)
    plt.close(fig)

    scales = np.linalg.norm(sliding["global_from_window"][:, 0, :3], axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(scales)), scales, marker="o")
    ax.set_title("Old sliding-window alignment scale")
    ax.set_xlabel("window")
    ax.set_ylabel("Sim(3) scale")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "old_sliding_window_scale.png", dpi=180)
    plt.close(fig)


def _run_full_window(model: VGGTOmega, images: torch.Tensor) -> dict[str, np.ndarray]:
    pred = model(images)
    extrinsic, intrinsic = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
    world_points = unproject_depth_map_to_point_map_torch(pred["depth"], extrinsic, intrinsic)
    return _tensor_dict_to_numpy(
        {
            "pose_enc": pred["pose_enc"],
            "depth": pred["depth"],
            "depth_conf": pred["depth_conf"],
            "images": pred["images"],
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "world_points_from_depth": world_points,
        }
    )


def _estimate_global_from_window_transform_for_indices(
    local_extrinsics: np.ndarray,
    global_extrinsics: list[np.ndarray | None],
    global_indices: list[int],
    *,
    overlap_count: int,
) -> np.ndarray:
    local_centers = []
    global_centers = []
    for local_idx in range(overlap_count):
        global_idx = global_indices[local_idx]
        global_extrinsic = global_extrinsics[global_idx]
        if global_extrinsic is None:
            continue
        local_centers.append(_camera_center(local_extrinsics[local_idx]))
        global_centers.append(_camera_center(global_extrinsic))
    if len(local_centers) < 2:
        raise ValueError("At least two overlap frames are required to estimate Sim(3).")
    return _estimate_sim3_umeyama(np.stack(local_centers, axis=0), np.stack(global_centers, axis=0)).astype(np.float32)


def _align_centers_to_full(centers: np.ndarray, full_centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = _estimate_sim3_umeyama(centers, full_centers).astype(np.float32)
    return _transform_points_sim3(transform, centers).astype(np.float32), transform


def _camera_centers(extrinsic: np.ndarray) -> np.ndarray:
    return np.stack([_camera_center(ext) for ext in extrinsic], axis=0).astype(np.float32)


def _tensor_dict_to_numpy(tensors: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    out = {}
    for key, value in tensors.items():
        array = value.detach().float().cpu().numpy()
        if array.shape[0] == 1:
            array = array[0]
        out[key] = array
    return out


def _expand_images(inputs: list[str]) -> list[str]:
    image_paths = []
    for item in inputs:
        matches = sorted(glob.glob(item))
        if matches:
            image_paths.extend(matches)
        elif os.path.isfile(item):
            image_paths.append(item)
    return sorted(dict.fromkeys(image_paths))


if __name__ == "__main__":
    main()

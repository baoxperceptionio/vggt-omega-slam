#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggt_omega.models import CausalVGGTOmega, VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera, extri_intri_to_pose_encoding
from vggt_omega.utils.slam import (
    predictions_to_point_cloud,
    save_point_cloud_ply,
    unproject_depth_map_to_point_map_torch,
)


def run_offline_incremental(
    image_paths: list[str],
    *,
    checkpoint: str | None,
    output_dir: str,
    chunk_size: int = 2,
    keyframe_stride: int = 8,
    image_resolution: int = 512,
    conf_percentile: float = 20.0,
    max_points: int = 300000,
    device: str = "cuda",
    allow_random_weights: bool = False,
    embed_dim: int = 1024,
    depth: int = 24,
    num_heads: int = 16,
) -> tuple[dict[str, np.ndarray], dict, str]:
    if len(image_paths) == 0:
        raise ValueError("No input images were provided.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    model = CausalVGGTOmega(embed_dim=embed_dim, depth=depth, num_heads=num_heads).eval()
    _load_or_initialize_model(model, checkpoint, allow_random_weights)
    model = model.to(device)
    images = load_and_preprocess_images(image_paths, image_resolution=image_resolution).to(device)

    state = model.init_slam_state()
    chunks: list[dict[str, torch.Tensor]] = []
    with torch.inference_mode():
        for start in range(0, images.shape[0], chunk_size):
            end = min(start + chunk_size, images.shape[0])
            predictions, state = model.step(
                images[start:end],
                state,
                keyframe_stride=keyframe_stride,
            )
            chunks.append(_detach_prediction_chunk(predictions))
            print(f"Processed frames {start}:{end}; state has {state['num_frames_seen']} frames.")

    merged = _merge_prediction_chunks(chunks)
    ply_path = _save_slam_outputs(
        merged,
        image_paths,
        output_dir=output_dir,
        ply_name="incremental_slam_points.ply",
        npz_name="incremental_slam_predictions.npz",
        conf_percentile=conf_percentile,
        max_points=max_points,
    )
    return merged, state, str(ply_path)


def run_sliding_window_slam(
    image_paths: list[str],
    *,
    checkpoint: str | None,
    output_dir: str,
    window_size: int = 5,
    image_resolution: int = 512,
    conf_percentile: float = 20.0,
    max_points: int = 300000,
    device: str = "cuda",
    allow_random_weights: bool = False,
) -> tuple[dict[str, np.ndarray], dict, str]:
    if len(image_paths) == 0:
        raise ValueError("No input images were provided.")
    if window_size <= 1:
        raise ValueError("window_size must be greater than 1 for sliding-window SLAM.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    model = VGGTOmega().eval()
    _load_or_initialize_model(model, checkpoint, allow_random_weights)
    model = model.to(device)
    images = load_and_preprocess_images(image_paths, image_resolution=image_resolution).to(device)
    num_frames = int(images.shape[0])
    if window_size > num_frames:
        window_size = num_frames

    global_predictions: dict[str, list[np.ndarray | None]] = {
        "pose_enc": [None] * num_frames,
        "depth": [None] * num_frames,
        "depth_conf": [None] * num_frames,
        "world_points_from_depth": [None] * num_frames,
        "extrinsic": [None] * num_frames,
        "intrinsic": [None] * num_frames,
        "images": [None] * num_frames,
    }
    global_extrinsics: list[np.ndarray | None] = [None] * num_frames
    alignments: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, num_frames - window_size + 1):
            end = start + window_size
            window_images = images[start:end]
            local = _run_full_window(model, window_images)

            if start == 0:
                global_from_window = np.eye(4, dtype=np.float32)
                frame_indices_to_store = range(0, window_size)
            else:
                global_from_window = _estimate_global_from_window_transform(
                    local["extrinsic"],
                    global_extrinsics,
                    start,
                    overlap_count=window_size - 1,
                )
                frame_indices_to_store = [window_size - 1]

            alignments.append(global_from_window)
            rebased = _rebase_window_predictions(local, global_from_window)
            for local_idx in frame_indices_to_store:
                global_idx = start + local_idx
                _store_frame_prediction(global_predictions, rebased, local, local_idx, global_idx)
                global_extrinsics[global_idx] = rebased["extrinsic"][local_idx]

            print(
                f"Processed window {start}:{end}; "
                f"stored frame(s) {[start + i for i in frame_indices_to_store]}."
            )

    merged = {key: np.stack(values, axis=0) for key, values in global_predictions.items() if values and values[0] is not None}
    ply_path = _save_slam_outputs(
        merged,
        image_paths,
        output_dir=output_dir,
        ply_name="sliding_window_slam_points.ply",
        npz_name="sliding_window_slam_predictions.npz",
        conf_percentile=conf_percentile,
        max_points=max_points,
        extra_payload={"window_size": np.array(window_size), "global_from_window": np.stack(alignments, axis=0)},
    )
    state = {"num_frames_seen": num_frames, "window_size": window_size, "num_windows": len(alignments)}
    return merged, state, str(ply_path)


def _run_full_window(model: VGGTOmega, images: torch.Tensor) -> dict[str, np.ndarray]:
    predictions = model(images)
    extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], predictions["images"].shape[-2:])
    world_points = unproject_depth_map_to_point_map_torch(predictions["depth"], extrinsic, intrinsic)

    tensors = {
        "pose_enc": predictions["pose_enc"],
        "depth": predictions["depth"],
        "depth_conf": predictions["depth_conf"],
        "images": predictions["images"],
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "world_points_from_depth": world_points,
    }
    out = {}
    for key, value in tensors.items():
        array = value.detach().float().cpu().numpy()
        if array.shape[0] == 1:
            array = array[0]
        out[key] = array
    return out


def _estimate_global_from_window_transform(
    local_extrinsics: np.ndarray,
    global_extrinsics: list[np.ndarray | None],
    window_start: int,
    *,
    overlap_count: int,
) -> np.ndarray:
    local_centers = []
    global_centers = []
    for local_idx in range(overlap_count):
        global_idx = window_start + local_idx
        global_extrinsic = global_extrinsics[global_idx]
        if global_extrinsic is None:
            continue
        local_centers.append(_camera_center(local_extrinsics[local_idx]))
        global_centers.append(_camera_center(global_extrinsic))

    if len(local_centers) < 2:
        raise ValueError(
            f"At least two overlap frames are required to estimate Sim(3) scale for "
            f"sliding window starting at frame {window_start}; got {len(local_centers)}."
        )
    return _estimate_sim3_umeyama(
        np.stack(local_centers, axis=0),
        np.stack(global_centers, axis=0),
    ).astype(np.float32)


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
        raise ValueError(f"At least two overlap frames are required to estimate Sim(3); got {len(local_centers)}.")
    return _estimate_sim3_umeyama(
        np.stack(local_centers, axis=0),
        np.stack(global_centers, axis=0),
    ).astype(np.float32)


def _rebase_window_predictions(local: dict[str, np.ndarray], global_from_window: np.ndarray) -> dict[str, np.ndarray]:
    scale, rotation, translation = _split_sim3(global_from_window)
    extrinsics = []
    for local_extrinsic in local["extrinsic"]:
        local_rotation = local_extrinsic[:3, :3]
        local_translation = local_extrinsic[:3, 3]
        global_rotation = local_rotation @ rotation.T
        global_translation = scale * local_translation - global_rotation @ translation
        extrinsics.append(np.concatenate([global_rotation, global_translation[:, None]], axis=1))
    extrinsics = np.stack(extrinsics, axis=0).astype(np.float32)

    points = local["world_points_from_depth"]
    points_global = _transform_points_sim3(global_from_window, points.reshape(-1, 3)).reshape(points.shape)
    pose_enc = _pose_encoding_from_camera(
        extrinsics,
        local["intrinsic"].astype(np.float32),
        image_size_hw=local["images"].shape[-2:],
    )
    return {
        "pose_enc": pose_enc,
        "extrinsic": extrinsics,
        "world_points_from_depth": points_global.astype(np.float32),
    }



def _pose_encoding_from_camera(
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    *,
    image_size_hw: tuple[int, int],
) -> np.ndarray:
    pose = extri_intri_to_pose_encoding(
        torch.from_numpy(extrinsics)[None],
        torch.from_numpy(intrinsics)[None],
        image_size_hw,
    )
    return pose[0].numpy().astype(np.float32)

def _store_frame_prediction(
    global_predictions: dict[str, list[np.ndarray | None]],
    rebased: dict[str, np.ndarray],
    local: dict[str, np.ndarray],
    local_idx: int,
    global_idx: int,
) -> None:
    for key in ("depth", "depth_conf", "intrinsic", "images"):
        global_predictions[key][global_idx] = local[key][local_idx]
    global_predictions["pose_enc"][global_idx] = rebased["pose_enc"][local_idx]
    global_predictions["extrinsic"][global_idx] = rebased["extrinsic"][local_idx]
    global_predictions["world_points_from_depth"][global_idx] = rebased["world_points_from_depth"][local_idx]


def _camera_center(extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]
    return (-rotation.T @ translation).astype(np.float64)


def _split_sim3(transform: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
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


def _transform_points_sim3(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _estimate_sim3_umeyama(source: np.ndarray, target: np.ndarray) -> np.ndarray:
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
        raise ValueError("Cannot estimate Sim(3) scale from nearly identical overlap camera centers.")

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
    return transform


def _load_or_initialize_model(model: torch.nn.Module, checkpoint: str | None, allow_random_weights: bool) -> None:
    loaded_checkpoint = False
    if checkpoint:
        if not os.path.isfile(checkpoint):
            if not allow_random_weights:
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        else:
            state_dict = torch.load(checkpoint, map_location="cpu")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:8]}")
            if missing:
                print(f"Warning: missing {len(missing)} checkpoint keys; first keys: {missing[:8]}")
            loaded_checkpoint = True
    elif not allow_random_weights:
        raise ValueError("A checkpoint is required unless --allow-random-weights is set.")

    if allow_random_weights and not loaded_checkpoint:
        print("Warning: running with stabilized random weights; output geometry is for smoke testing only.")
        _stabilize_random_weights(model)


def _detach_prediction_chunk(predictions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keep = [
        "pose_enc",
        "depth",
        "depth_conf",
        "world_points_from_depth",
        "extrinsic",
        "intrinsic",
        "images",
    ]
    return {
        key: value.detach().float().cpu()
        for key, value in predictions.items()
        if key in keep and isinstance(value, torch.Tensor)
    }


def _stabilize_random_weights(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1e-3)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            out_features = module.bias_mask.numel()
            module.bias_mask.fill_(1)
            module.bias_mask[out_features // 3 : 2 * out_features // 3].fill_(0)


def _merge_prediction_chunks(chunks: list[dict[str, torch.Tensor]]) -> dict[str, np.ndarray]:
    merged = {}
    for key in chunks[0]:
        values = [chunk[key] for chunk in chunks if key in chunk]
        if not values:
            continue
        tensor = torch.cat(values, dim=1)
        array = tensor.numpy()
        if array.shape[0] == 1:
            array = array[0]
        merged[key] = array
    return merged


def _save_slam_outputs(
    merged: dict[str, np.ndarray],
    image_paths: list[str],
    *,
    output_dir: str,
    ply_name: str,
    npz_name: str,
    conf_percentile: float,
    max_points: int,
    extra_payload: dict[str, np.ndarray] | None = None,
) -> Path:
    vertices, colors = predictions_to_point_cloud(
        merged,
        conf_percentile=conf_percentile,
        max_points=max_points,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ply_path = output_path / ply_name
    save_point_cloud_ply(ply_path, vertices, colors)

    payload = {
        "pose_enc": merged.get("pose_enc"),
        "extrinsic": merged.get("extrinsic"),
        "intrinsic": merged.get("intrinsic"),
        "image_paths": np.array(image_paths),
    }
    if extra_payload is not None:
        payload.update(extra_payload)
    np.savez_compressed(output_path / npz_name, **payload)
    return ply_path


def _expand_images(inputs: list[str]) -> list[str]:
    image_paths = []
    for item in inputs:
        matches = sorted(glob.glob(item))
        if matches:
            image_paths.extend(matches)
        elif os.path.isfile(item):
            image_paths.append(item)
    return sorted(dict.fromkeys(image_paths))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VGGT-Omega in offline SLAM mode.")
    parser.add_argument("images", nargs="+", help="Input image files or glob patterns.")
    parser.add_argument("--checkpoint", default="checkpoints/VGGT-Omega-1B-512/model.pt")
    parser.add_argument("--output-dir", default="outputs/incremental_slam")
    parser.add_argument("--mode", choices=["incremental", "sliding-window"], default="incremental")
    parser.add_argument("--chunk-size", type=int, default=2, help="Chunk size for incremental mode.")
    parser.add_argument("--window-size", type=int, default=5, help="Sliding window size for sliding-window mode.")
    parser.add_argument("--keyframe-stride", type=int, default=8)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--conf-percentile", type=float, default=20.0)
    parser.add_argument("--max-points", type=int, default=300000)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument(
        "--allow-random-weights",
        action="store_true",
        help="Run without a checkpoint for API smoke tests only; output geometry is not meaningful.",
    )
    args = parser.parse_args()

    image_paths = _expand_images(args.images)
    if args.mode == "sliding-window":
        _, state, ply_path = run_sliding_window_slam(
            image_paths,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            window_size=args.window_size,
            image_resolution=args.image_resolution,
            conf_percentile=args.conf_percentile,
            max_points=args.max_points,
            device=args.device,
            allow_random_weights=args.allow_random_weights,
        )
    else:
        _, state, ply_path = run_offline_incremental(
            image_paths,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            keyframe_stride=args.keyframe_stride,
            image_resolution=args.image_resolution,
            conf_percentile=args.conf_percentile,
            max_points=args.max_points,
            device=args.device,
            allow_random_weights=args.allow_random_weights,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
        )
    print(f"Saved PLY: {ply_path}")
    print(f"Final state frames: {state['num_frames_seen']}")


if __name__ == "__main__":
    main()

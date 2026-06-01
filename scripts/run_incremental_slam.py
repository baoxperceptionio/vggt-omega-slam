#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Ground-normalized sliding-window SLAM for VGGT-Omega.

This is the single supported SLAM pipeline in this repo. It runs overlapping
full VGGT-Omega windows, aligns every new window into a ground-normalized global
frame with Sim(3), and only accepts new frames into the map/future window when
translation exceeds a configurable threshold.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera, extri_intri_to_pose_encoding
from vggt_omega.utils.slam import (
    predictions_to_point_cloud,
    save_point_cloud_ply,
    unproject_depth_map_to_point_map_torch,
)


def run_ground_tracking_slam(
    image_paths: list[str],
    *,
    checkpoint: str | None,
    output_dir: str,
    window_size: int = 5,
    image_resolution: int = 512,
    displacement_threshold: float = 0.1,
    conf_percentile: float = 20.0,
    max_points: int = 300000,
    device: str = "cuda",
    allow_random_weights: bool = False,
) -> tuple[dict[str, np.ndarray], dict, str]:
    if len(image_paths) == 0:
        raise ValueError("No input images were provided.")
    if window_size <= 1:
        raise ValueError("window_size must be greater than 1.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    model = VGGTOmega().eval()
    _load_or_initialize_model(model, checkpoint, allow_random_weights)
    model = model.to(device)

    images = load_and_preprocess_images(image_paths, image_resolution=image_resolution).to(device)
    num_frames = int(images.shape[0])
    if window_size > num_frames:
        window_size = num_frames

    all_pose_enc: list[np.ndarray | None] = [None] * num_frames
    all_extrinsic: list[np.ndarray | None] = [None] * num_frames
    all_intrinsic: list[np.ndarray | None] = [None] * num_frames
    accepted_mask = np.zeros(num_frames, dtype=bool)
    displacements = np.full(num_frames, np.nan, dtype=np.float32)
    global_extrinsics: list[np.ndarray | None] = [None] * num_frames
    accepted_indices: list[int] = []
    map_predictions: dict[str, list[np.ndarray]] = {
        "depth": [],
        "depth_conf": [],
        "images": [],
        "world_points_from_depth": [],
    }
    alignments: list[np.ndarray] = []

    with torch.inference_mode():
        initial_local = _run_full_window(model, images[:window_size])
        ground_transform, ground_info = _estimate_ground_normalization(initial_local)
        initial_rebased = _rebase_window_predictions(initial_local, ground_transform)
        alignments.append(ground_transform)

        for local_idx in range(window_size):
            global_idx = local_idx
            _store_pose_only(all_pose_enc, all_extrinsic, all_intrinsic, initial_rebased, initial_local, local_idx, global_idx)
            _append_map_frame(map_predictions, initial_rebased, initial_local, local_idx)
            global_extrinsics[global_idx] = initial_rebased["extrinsic"][local_idx]
            accepted_indices.append(global_idx)
            accepted_mask[global_idx] = True
            displacements[global_idx] = 0.0 if global_idx == 0 else float(
                np.linalg.norm(_camera_center(global_extrinsics[global_idx]) - _camera_center(global_extrinsics[global_idx - 1]))
            )

        print(
            f"Initialized frames 0:{window_size}; accepted all initial frames; "
            f"ground scale={ground_info['scale']:.6f}; ground inliers={ground_info['inliers']}."
        )

        for candidate_idx in range(window_size, num_frames):
            overlap_indices = accepted_indices[-(window_size - 1):]
            if len(overlap_indices) < 2:
                raise ValueError("Need at least two accepted overlap frames for Sim(3) tracking.")

            window_indices = overlap_indices + [candidate_idx]
            local = _run_full_window(model, images[window_indices])
            global_from_window = _estimate_global_from_window_transform_for_indices(
                local["extrinsic"],
                global_extrinsics,
                window_indices,
                overlap_count=len(overlap_indices),
            )
            rebased = _rebase_window_predictions(local, global_from_window)
            alignments.append(global_from_window)

            candidate_local_idx = len(window_indices) - 1
            _store_pose_only(
                all_pose_enc,
                all_extrinsic,
                all_intrinsic,
                rebased,
                local,
                candidate_local_idx,
                candidate_idx,
            )

            last_center = _camera_center(global_extrinsics[accepted_indices[-1]])
            candidate_center = _camera_center(rebased["extrinsic"][candidate_local_idx])
            displacement = float(np.linalg.norm(candidate_center - last_center))
            displacements[candidate_idx] = displacement

            if displacement > displacement_threshold:
                _append_map_frame(map_predictions, rebased, local, candidate_local_idx)
                global_extrinsics[candidate_idx] = rebased["extrinsic"][candidate_local_idx]
                accepted_indices.append(candidate_idx)
                accepted_mask[candidate_idx] = True
                decision = "accepted"
            else:
                decision = "pose-only"

            scale = float(np.cbrt(np.maximum(np.linalg.det(global_from_window[:3, :3]), 1e-12)))
            print(
                f"Tracked frame {candidate_idx}; displacement={displacement:.4f}; "
                f"scale={scale:.4f}; {decision}."
            )

    all_pose = np.stack(all_pose_enc, axis=0)
    all_ext = np.stack(all_extrinsic, axis=0)
    all_int = np.stack(all_intrinsic, axis=0)
    map_merged = {key: np.stack(value, axis=0) for key, value in map_predictions.items()}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vertices, colors = predictions_to_point_cloud(map_merged, conf_percentile=conf_percentile, max_points=max_points)
    ply_path = output_path / "ground_tracking_slam_points.ply"
    save_point_cloud_ply(ply_path, vertices, colors)

    np.savez_compressed(
        output_path / "ground_tracking_slam_predictions.npz",
        pose_enc=all_pose,
        extrinsic=all_ext,
        intrinsic=all_int,
        image_paths=np.array(image_paths),
        accepted_mask=accepted_mask,
        accepted_indices=np.array(accepted_indices, dtype=np.int64),
        displacements=displacements,
        ground_transform=ground_transform,
        ground_plane=np.array(ground_info["plane"], dtype=np.float32),
        ground_inliers=np.array(ground_info["inliers"], dtype=np.int64),
        ground_ransac_threshold=np.array(ground_info["ransac_threshold"], dtype=np.float32),
        ground_coarse_distance=np.array(ground_info["coarse_distance"], dtype=np.float32),
        global_from_window=np.stack(alignments, axis=0),
    )

    state = {
        "num_frames_seen": num_frames,
        "window_size": window_size,
        "accepted_frames": len(accepted_indices),
        "displacement_threshold": displacement_threshold,
    }
    return map_merged, state, str(ply_path)


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


def _estimate_ground_normalization(predictions: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    candidates = _select_ground_candidates(predictions["world_points_from_depth"][0], predictions["depth_conf"][0])
    coarse_plane = _fit_plane_svd(candidates)
    coarse_distance = _plane_camera_distance(coarse_plane)
    ransac_threshold = max(coarse_distance / 10.0, 1e-4)
    plane, inlier_count = _ransac_plane(candidates, threshold=ransac_threshold, num_iterations=1000)
    normal = plane[:3].astype(np.float64)
    offset = float(plane[3])

    inlier_mask = np.abs(candidates @ normal + offset) < ransac_threshold
    centroid = candidates[inlier_mask].mean(axis=0)
    if normal @ centroid < 0:
        normal = -normal
        offset = -offset

    target = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    rotation = _rotation_between_vectors(normal, target)
    distance = abs(offset) / max(np.linalg.norm(normal), 1e-12)
    if distance <= 1e-8:
        raise ValueError("Estimated ground plane is too close to first camera center.")
    scale = 1.0 / distance

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    info = {
        "plane": [float(normal[0]), float(normal[1]), float(normal[2]), float(offset)],
        "inliers": int(inlier_count),
        "scale": float(scale),
        "distance": float(distance),
        "coarse_distance": float(coarse_distance),
        "ransac_threshold": float(ransac_threshold),
    }
    return transform.astype(np.float32), info


def _select_ground_candidates(points: np.ndarray, confidence: np.ndarray, max_candidates: int = 50000) -> np.ndarray:
    height, _ = confidence.shape
    yy = np.arange(height)[:, None]
    finite = np.isfinite(points).all(axis=-1) & np.isfinite(confidence)
    if not np.any(finite):
        raise ValueError("No finite points available for ground estimation.")

    conf_mask = confidence >= np.percentile(confidence[finite], 50.0)
    lower_image = yy >= int(height * 0.55)
    below_camera = points[..., 1] >= np.percentile(points[..., 1][finite], 55.0)
    mask = finite & conf_mask & lower_image & below_camera
    candidates = points[mask].reshape(-1, 3).astype(np.float64)
    if len(candidates) < 100:
        candidates = points[finite].reshape(-1, 3).astype(np.float64)
    if len(candidates) < 3:
        raise ValueError("Not enough valid points to estimate ground plane.")
    if len(candidates) > max_candidates:
        indices = np.linspace(0, len(candidates) - 1, max_candidates).astype(np.int64)
        candidates = candidates[indices]
    return candidates


def _plane_camera_distance(plane: np.ndarray) -> float:
    return abs(float(plane[3])) / max(float(np.linalg.norm(plane[:3])), 1e-12)


def _ransac_plane(points: np.ndarray, *, threshold: float, num_iterations: int = 1000) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(0)
    best_plane = None
    best_inliers = -1
    for _ in range(num_iterations):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm <= 1e-12:
            continue
        normal = normal / norm
        offset = -float(normal @ sample[0])
        distances = np.abs(points @ normal + offset)
        inliers = int(np.count_nonzero(distances < threshold))
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = np.array([normal[0], normal[1], normal[2], offset], dtype=np.float64)
    if best_plane is None:
        raise ValueError("Ground RANSAC failed to find a plane.")

    inlier_mask = np.abs(points @ best_plane[:3] + best_plane[3]) < threshold
    if np.count_nonzero(inlier_mask) >= 3:
        best_plane = _fit_plane_svd(points[inlier_mask])
        best_inliers = int(np.count_nonzero(np.abs(points @ best_plane[:3] + best_plane[3]) < threshold))
    return best_plane, best_inliers


def _fit_plane_svd(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    normal = vt[-1]
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    offset = -float(normal @ center)
    return np.array([normal[0], normal[1], normal[2], offset], dtype=np.float64)


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
    return _estimate_sim3_umeyama(np.stack(local_centers, axis=0), np.stack(global_centers, axis=0)).astype(np.float32)


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
    pose_enc = _pose_encoding_from_camera(extrinsics, local["intrinsic"].astype(np.float32), image_size_hw=local["images"].shape[-2:])
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
    pose = extri_intri_to_pose_encoding(torch.from_numpy(extrinsics)[None], torch.from_numpy(intrinsics)[None], image_size_hw)
    return pose[0].numpy().astype(np.float32)


def _store_pose_only(
    all_pose_enc: list[np.ndarray | None],
    all_extrinsic: list[np.ndarray | None],
    all_intrinsic: list[np.ndarray | None],
    rebased: dict[str, np.ndarray],
    local: dict[str, np.ndarray],
    local_idx: int,
    global_idx: int,
) -> None:
    all_pose_enc[global_idx] = rebased["pose_enc"][local_idx]
    all_extrinsic[global_idx] = rebased["extrinsic"][local_idx]
    all_intrinsic[global_idx] = local["intrinsic"][local_idx]


def _append_map_frame(
    map_predictions: dict[str, list[np.ndarray]],
    rebased: dict[str, np.ndarray],
    local: dict[str, np.ndarray],
    local_idx: int,
) -> None:
    map_predictions["depth"].append(local["depth"][local_idx])
    map_predictions["depth_conf"].append(local["depth_conf"][local_idx])
    map_predictions["images"].append(local["images"][local_idx])
    map_predictions["world_points_from_depth"].append(rebased["world_points_from_depth"][local_idx])


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


def _rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / max(np.linalg.norm(source), 1e-12)
    target = target / max(np.linalg.norm(target), 1e-12)
    cross = np.cross(source, target)
    dot = float(np.clip(source @ target, -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-10:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(source[0]) > 0.9:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis = axis - source * (axis @ source)
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        return _axis_angle_to_matrix(axis, np.pi)
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - dot) / max(np.linalg.norm(cross) ** 2, 1e-12))


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


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
    parser = argparse.ArgumentParser(description="Run the ground-normalized VGGT-Omega SLAM tracker.")
    parser.add_argument("images", nargs="+", help="Input image files or glob patterns.")
    parser.add_argument("--checkpoint", default="checkpoints/VGGT-Omega-1B-512/model.pt")
    parser.add_argument("--output-dir", default="outputs/ground_tracking_slam")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--displacement-threshold", type=float, default=0.1)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--conf-percentile", type=float, default=20.0)
    parser.add_argument("--max-points", type=int, default=300000)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--allow-random-weights",
        action="store_true",
        help="Run without a checkpoint for API smoke tests only; output geometry is not meaningful.",
    )
    args = parser.parse_args()

    image_paths = _expand_images(args.images)
    _, state, ply_path = run_ground_tracking_slam(
        image_paths,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        window_size=args.window_size,
        image_resolution=args.image_resolution,
        displacement_threshold=args.displacement_threshold,
        conf_percentile=args.conf_percentile,
        max_points=args.max_points,
        device=args.device,
        allow_random_weights=args.allow_random_weights,
    )
    print(f"Saved PLY: {ply_path}")
    print(f"Final state frames: {state['num_frames_seen']}")
    print(f"Accepted frames: {state['accepted_frames']}")


if __name__ == "__main__":
    main()

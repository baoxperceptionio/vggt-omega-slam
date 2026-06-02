#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Fixed-pose KV-cache SLAM runner for VGGT-Omega.

The supported pipeline is append-only: each new frame is predicted against a
cache of pose-conditioned history tokens, then committed back into the cache as
a fixed SLAM anchor. It does not run overlapping full windows and does not
estimate per-window Sim(3) alignment scales.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggt_omega.models import CausalVGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera, extri_intri_to_pose_encoding
from vggt_omega.utils.slam import (
    predictions_to_point_cloud,
    save_point_cloud_ply,
    unproject_depth_map_to_point_map_torch,
)


def run_fixed_pose_kv_slam(
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
    """Run append-only fixed-pose incremental SLAM.

    ``window_size`` is accepted for CLI compatibility but is no longer used: the
    runner never reprocesses overlapping windows or estimates per-window Sim(3)
    scales. Every frame is committed into the causal KV cache as a locked pose
    anchor after its pose has been predicted once.
    """

    if len(image_paths) == 0:
        raise ValueError("No input images were provided.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    model = CausalVGGTOmega().eval()
    _load_or_initialize_model(model, checkpoint, allow_random_weights)
    model = model.to(device)

    images = load_and_preprocess_images(image_paths, image_resolution=image_resolution).to(device)
    num_frames = int(images.shape[0])

    all_pose_enc: list[np.ndarray | None] = [None] * num_frames
    all_extrinsic: list[np.ndarray | None] = [None] * num_frames
    all_intrinsic: list[np.ndarray | None] = [None] * num_frames
    accepted_mask = np.zeros(num_frames, dtype=bool)
    displacements = np.full(num_frames, np.nan, dtype=np.float32)
    locked_pose_mask = np.zeros(num_frames, dtype=bool)
    map_predictions: dict[str, list[np.ndarray]] = {
        "depth": [],
        "depth_conf": [],
        "images": [],
        "world_points_from_depth": [],
    }
    accepted_indices: list[int] = []

    history_state = model.init_slam_state()
    ground_transform: np.ndarray | None = None
    ground_info: dict | None = None
    last_map_center: np.ndarray | None = None

    with torch.inference_mode():
        for frame_idx in range(num_frames):
            image_chunk = images[frame_idx : frame_idx + 1]

            # Probe with fixed history, but discard the returned state so the
            # unconditioned current frame is not written into memory.
            predictions, _ = model.forward_incremental(image_chunk, _copy_slam_state(history_state))
            local = _causal_predictions_to_numpy(predictions)

            if ground_transform is None:
                ground_transform, ground_info = _estimate_ground_normalization(local)
                print(
                    f"Initialized fixed-pose KV SLAM with frame 0; "
                    f"ground scale={ground_info['scale']:.6f}; ground inliers={ground_info['inliers']}."
                )

            rebased = _apply_global_transform(local, ground_transform)
            _store_pose_only(all_pose_enc, all_extrinsic, all_intrinsic, rebased, local, 0, frame_idx)

            current_center = _camera_center(rebased["extrinsic"][0])
            if last_map_center is None:
                displacement = 0.0
                accept_for_map = True
            else:
                displacement = float(np.linalg.norm(current_center - last_map_center))
                accept_for_map = displacement > displacement_threshold
            displacements[frame_idx] = displacement

            if accept_for_map:
                _append_map_frame(map_predictions, rebased, local, 0)
                accepted_indices.append(frame_idx)
                accepted_mask[frame_idx] = True
                last_map_center = current_center

            # Commit in the model's native gauge. The exported trajectory is
            # ground-normalized separately; mixing that world gauge into the KV
            # cache makes fixed-pose conditioning inconsistent with training.
            fixed_pose = torch.from_numpy(local["pose_enc"][None]).to(
                device=predictions["pose_enc"].device,
                dtype=predictions["pose_enc"].dtype,
            )
            fixed_mask = torch.ones(fixed_pose.shape[:2], dtype=torch.bool, device=fixed_pose.device)
            _, history_state = model.forward_incremental(
                image_chunk,
                history_state,
                fixed_pose_enc=fixed_pose,
                fixed_pose_mask=fixed_mask,
            )
            locked_pose_mask[frame_idx] = True

            cache_tokens = _count_cached_tokens(history_state)
            decision = "accepted" if accept_for_map else "pose-only"
            print(
                f"Tracked frame {frame_idx}; displacement={displacement:.4f}; "
                f"fixed_pose=locked; cache_tokens={cache_tokens}; {decision}."
            )

    if ground_transform is None or ground_info is None:
        raise RuntimeError("No frames were processed.")

    all_pose = np.stack(all_pose_enc, axis=0)
    all_ext = np.stack(all_extrinsic, axis=0)
    all_int = np.stack(all_intrinsic, axis=0)
    map_merged = {key: np.stack(value, axis=0) for key, value in map_predictions.items()}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vertices, colors = predictions_to_point_cloud(map_merged, conf_percentile=conf_percentile, max_points=max_points)
    ply_path = output_path / "fixed_pose_kv_slam_points.ply"
    save_point_cloud_ply(ply_path, vertices, colors)

    np.savez_compressed(
        output_path / "fixed_pose_kv_slam_predictions.npz",
        pose_enc=all_pose,
        extrinsic=all_ext,
        intrinsic=all_int,
        image_paths=np.array(image_paths),
        accepted_mask=accepted_mask,
        accepted_indices=np.array(accepted_indices, dtype=np.int64),
        displacements=displacements,
        locked_pose_mask=locked_pose_mask,
        ground_transform=ground_transform,
        ground_plane=np.array(ground_info["plane"], dtype=np.float32),
        ground_inliers=np.array(ground_info["inliers"], dtype=np.int64),
        ground_ransac_threshold=np.array(ground_info["ransac_threshold"], dtype=np.float32),
        ground_coarse_distance=np.array(ground_info["coarse_distance"], dtype=np.float32),
        kv_cache_tokens=np.array(_count_cached_tokens(history_state), dtype=np.int64),
    )

    state = {
        "num_frames_seen": num_frames,
        "accepted_frames": len(accepted_indices),
        "locked_pose_frames": int(locked_pose_mask.sum()),
        "displacement_threshold": displacement_threshold,
        "kv_cache_tokens": _count_cached_tokens(history_state),
        "pipeline": "fixed_pose_kv",
    }
    return map_merged, state, str(ply_path)


def run_fixed_pose_window_slam(
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
    """Run fixed-pose inference with the same 4-fixed + 1-target setup as FT."""

    if len(image_paths) == 0:
        raise ValueError("No input images were provided.")
    if window_size != 5:
        raise ValueError("Fixed-pose window SLAM is trained for window_size=5.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    model = CausalVGGTOmega().eval()
    _load_or_initialize_model(model, checkpoint, allow_random_weights)
    model = model.to(device)

    images = load_and_preprocess_images(image_paths, image_resolution=image_resolution).to(device)
    num_frames = int(images.shape[0])
    if num_frames < window_size:
        raise ValueError(f"Need at least {window_size} frames for fixed-pose window inference.")

    all_pose_enc: list[np.ndarray | None] = [None] * num_frames
    all_pose_enc_model: list[np.ndarray | None] = [None] * num_frames
    all_extrinsic: list[np.ndarray | None] = [None] * num_frames
    all_intrinsic: list[np.ndarray | None] = [None] * num_frames
    accepted_mask = np.zeros(num_frames, dtype=bool)
    displacements = np.full(num_frames, np.nan, dtype=np.float32)
    locked_pose_mask = np.zeros(num_frames, dtype=bool)
    map_predictions: dict[str, list[np.ndarray]] = {
        "depth": [],
        "depth_conf": [],
        "images": [],
        "world_points_from_depth": [],
    }
    accepted_indices: list[int] = []
    last_map_center: np.ndarray | None = None

    def store_frame(
        frame_idx: int,
        rebased_np: dict[str, np.ndarray],
        model_np: dict[str, np.ndarray],
        local_idx: int,
    ) -> None:
        nonlocal last_map_center
        all_pose_enc[frame_idx] = rebased_np["pose_enc"][local_idx]
        all_pose_enc_model[frame_idx] = model_np["pose_enc"][local_idx]
        all_extrinsic[frame_idx] = rebased_np["extrinsic"][local_idx]
        all_intrinsic[frame_idx] = model_np["intrinsic"][local_idx]
        locked_pose_mask[frame_idx] = True

        current_center = _camera_center(rebased_np["extrinsic"][local_idx])
        if last_map_center is None:
            displacement = 0.0
            accept_for_map = True
        else:
            displacement = float(np.linalg.norm(current_center - last_map_center))
            accept_for_map = displacement > displacement_threshold
        displacements[frame_idx] = displacement

        if accept_for_map:
            _append_map_frame(map_predictions, rebased_np, model_np, local_idx)
            accepted_indices.append(frame_idx)
            accepted_mask[frame_idx] = True
            last_map_center = current_center

        decision = "accepted" if accept_for_map else "pose-only"
        print(
            f"Tracked frame {frame_idx}; displacement={displacement:.4f}; "
            f"fixed_window=4+1; {decision}."
        )

    with torch.inference_mode():
        initial_predictions, _ = model.forward_incremental(images[:window_size], None)
        initial_local = _causal_predictions_to_numpy(initial_predictions)
        ground_transform, ground_info = _estimate_ground_normalization(initial_local)
        print(
            f"Initialized fixed-pose window SLAM with frames 0-{window_size - 1}; "
            f"ground scale={ground_info['scale']:.6f}; ground inliers={ground_info['inliers']}."
        )
        initial_rebased = _apply_global_transform(initial_local, ground_transform)

        # Bootstrap the four fixed history frames. Frame 4 and later are then
        # predicted with exactly the FT setup: 4 locked poses + 1 unlocked target.
        for frame_idx in range(window_size - 1):
            store_frame(frame_idx, initial_rebased, initial_local, frame_idx)

        for frame_idx in range(window_size - 1, num_frames):
            window_indices = list(range(frame_idx - (window_size - 1), frame_idx + 1))
            fixed_pose = np.zeros((1, window_size, 9), dtype=np.float32)
            fixed_mask_np = np.zeros((1, window_size), dtype=bool)
            for local_idx, history_idx in enumerate(window_indices[:-1]):
                if all_pose_enc_model[history_idx] is None:
                    raise RuntimeError(f"Missing fixed pose for history frame {history_idx}.")
                fixed_pose[0, local_idx] = all_pose_enc_model[history_idx]
                fixed_mask_np[0, local_idx] = True

            fixed_pose_tensor = torch.from_numpy(fixed_pose).to(device=device, dtype=images.dtype)
            fixed_mask = torch.from_numpy(fixed_mask_np).to(device=device)
            predictions, _ = model.forward_incremental(
                images[window_indices],
                None,
                fixed_pose_enc=fixed_pose_tensor,
                fixed_pose_mask=fixed_mask,
            )
            pred_np = _causal_predictions_to_numpy(predictions)
            rebased_np = _apply_global_transform(pred_np, ground_transform)
            store_frame(frame_idx, rebased_np, pred_np, window_size - 1)

    all_pose = np.stack(all_pose_enc, axis=0)
    all_ext = np.stack(all_extrinsic, axis=0)
    all_int = np.stack(all_intrinsic, axis=0)
    map_merged = {key: np.stack(value, axis=0) for key, value in map_predictions.items()}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vertices, colors = predictions_to_point_cloud(map_merged, conf_percentile=conf_percentile, max_points=max_points)
    ply_path = output_path / "fixed_pose_kv_slam_points.ply"
    save_point_cloud_ply(ply_path, vertices, colors)

    np.savez_compressed(
        output_path / "fixed_pose_kv_slam_predictions.npz",
        pose_enc=all_pose,
        pose_enc_model=np.stack(all_pose_enc_model, axis=0),
        extrinsic=all_ext,
        intrinsic=all_int,
        image_paths=np.array(image_paths),
        accepted_mask=accepted_mask,
        accepted_indices=np.array(accepted_indices, dtype=np.int64),
        displacements=displacements,
        locked_pose_mask=locked_pose_mask,
        ground_transform=ground_transform,
        ground_plane=np.array(ground_info["plane"], dtype=np.float32),
        ground_inliers=np.array(ground_info["inliers"], dtype=np.int64),
        ground_ransac_threshold=np.array(ground_info["ransac_threshold"], dtype=np.float32),
        ground_coarse_distance=np.array(ground_info["coarse_distance"], dtype=np.float32),
        kv_cache_tokens=np.array(0, dtype=np.int64),
    )

    state = {
        "num_frames_seen": num_frames,
        "accepted_frames": len(accepted_indices),
        "locked_pose_frames": int(locked_pose_mask.sum()),
        "displacement_threshold": displacement_threshold,
        "kv_cache_tokens": 0,
        "pipeline": "fixed_pose_window_4plus1",
    }
    return map_merged, state, str(ply_path)


def _copy_slam_state(state: dict) -> dict:
    copied = dict(state)
    copied["layer_kv_cache"] = {layer_idx: dict(cache) for layer_idx, cache in state.get("layer_kv_cache", {}).items()}
    copied["keyframe_poses"] = list(state.get("keyframe_poses", []))
    copied["keyframe_points"] = list(state.get("keyframe_points", []))
    return copied


def _count_cached_tokens(state: dict) -> int:
    cache = state.get("layer_kv_cache", {})
    if not cache:
        return 0
    first_cache = next(iter(cache.values()))
    k = first_cache.get("k")
    return 0 if k is None else int(k.shape[2])


def _causal_predictions_to_numpy(predictions: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    tensors = {
        "pose_enc": predictions["pose_enc"],
        "depth": predictions["depth"],
        "depth_conf": predictions["depth_conf"],
        "images": predictions["images"],
        "extrinsic": predictions["extrinsic"],
        "intrinsic": predictions["intrinsic"],
        "world_points_from_depth": predictions["world_points_from_depth"],
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


def _apply_global_transform(local: dict[str, np.ndarray], global_from_model: np.ndarray) -> dict[str, np.ndarray]:
    scale, rotation, translation = _split_sim3(global_from_model)
    extrinsics = []
    for local_extrinsic in local["extrinsic"]:
        local_rotation = local_extrinsic[:3, :3]
        local_translation = local_extrinsic[:3, 3]
        global_rotation = local_rotation @ rotation.T
        global_translation = scale * local_translation - global_rotation @ translation
        extrinsics.append(np.concatenate([global_rotation, global_translation[:, None]], axis=1))
    extrinsics = np.stack(extrinsics, axis=0).astype(np.float32)

    points = local["world_points_from_depth"]
    points_global = _transform_points_sim3(global_from_model, points.reshape(-1, 3)).reshape(points.shape)
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
    parser = argparse.ArgumentParser(description="Run the fixed-pose KV-cache VGGT-Omega SLAM tracker.")
    parser.add_argument("images", nargs="+", help="Input image files or glob patterns.")
    parser.add_argument("--checkpoint", default="checkpoints/VGGT-Omega-1B-512/model.pt")
    parser.add_argument("--output-dir", default="outputs/fixed_pose_kv_slam")
    parser.add_argument("--window-size", type=int, default=5, help="Deprecated compatibility option; fixed-pose KV SLAM does not use sliding windows.")
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
    _, state, ply_path = run_fixed_pose_kv_slam(
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
    print(f"Accepted map frames: {state['accepted_frames']}")
    print(f"Locked pose frames: {state['locked_pose_frames']}")
    print(f"KV cache tokens: {state['kv_cache_tokens']}")


if __name__ == "__main__":
    main()


# Backward-compatible alias for programmatic callers.
run_ground_tracking_slam = run_fixed_pose_kv_slam

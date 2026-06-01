# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
import torch


def unproject_depth_map_to_point_map_torch(
    depth_map: torch.Tensor,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
) -> torch.Tensor:
    """Unproject VGGT-Omega depth into world points, matching the demo convention."""

    depth = depth_map[..., 0]
    batch_size, num_frames, height, width = depth.shape

    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    x = x.view(1, 1, height, width).expand(batch_size, num_frames, height, width)
    y = y.view(1, 1, height, width).expand(batch_size, num_frames, height, width)

    fx = intrinsic[..., 0, 0][..., None, None]
    fy = intrinsic[..., 1, 1][..., None, None]
    cx = intrinsic[..., 0, 2][..., None, None]
    cy = intrinsic[..., 1, 2][..., None, None]

    camera_points = torch.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        dim=-1,
    )

    rotation = extrinsic[..., :3, :3]
    translation = extrinsic[..., :3, 3]
    return torch.einsum(
        "bsij,bshwj->bshwi",
        rotation.transpose(-1, -2),
        camera_points - translation[..., None, None, :],
    )


def depth_edge_mask(depth: np.ndarray, rtol: float = 0.03, kernel_size: int = 3) -> np.ndarray:
    depth = np.asarray(depth)
    original_shape = depth.shape
    depth = depth.reshape(-1, *original_shape[-2:])

    pad = kernel_size // 2
    padded = np.pad(depth, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(depth, -np.inf)
    depth_min = np.full_like(depth, np.inf)

    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[:, y : y + depth.shape[-2], x : x + depth.shape[-1]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)

    relative_jump = (depth_max - depth_min) / np.maximum(np.abs(depth), 1e-6)
    return (relative_jump > rtol).reshape(original_shape)


def predictions_to_point_cloud(
    predictions: dict[str, np.ndarray],
    *,
    conf_percentile: float = 20.0,
    max_points: int = 300000,
    filter_depth_edges: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    points = predictions["world_points_from_depth"]
    conf = predictions["depth_conf"].copy()
    images = predictions["images"]

    if filter_depth_edges and "depth" in predictions:
        conf[depth_edge_mask(predictions["depth"][..., 0])] = 0.0

    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))

    vertices = points.reshape(-1, 3)
    colors = (images.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)
    conf = conf.reshape(-1)

    mask = np.isfinite(vertices).all(axis=1) & np.isfinite(conf)
    if conf_percentile > 0 and np.any(mask):
        threshold = np.percentile(conf[mask], max(0.0, min(100.0, conf_percentile)))
        mask &= conf >= threshold
    mask &= conf > 1e-5

    vertices = vertices[mask].astype(np.float32)
    colors = colors[mask]
    if max_points > 0 and len(vertices) > max_points:
        indices = np.linspace(0, len(vertices) - 1, max_points).astype(np.int64)
        vertices = vertices[indices]
        colors = colors[indices]
    return vertices, colors


def save_point_cloud_ply(path: str | Path, vertices: np.ndarray, colors: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(vertices, colors):
            handle.write(
                f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )

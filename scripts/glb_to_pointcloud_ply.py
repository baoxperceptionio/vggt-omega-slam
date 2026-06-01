#!/usr/bin/env python3
"""Extract point-cloud geometry from a GLB scene and write a simple PLY file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GLB point clouds to an easy-to-open PLY point cloud.")
    parser.add_argument("input_glb", type=Path, help="Input .glb file.")
    parser.add_argument("output_ply", type=Path, help="Output .ply point cloud file.")
    parser.add_argument(
        "--include-mesh-vertices",
        action="store_true",
        help="Also include mesh vertices, such as camera frustums. Default only exports PointCloud geometry.",
    )
    parser.add_argument("--max-points", type=int, default=0, help="Optional deterministic point limit. 0 keeps all.")
    parser.add_argument("--ascii", action="store_true", help="Write ASCII PLY instead of binary little-endian PLY.")
    return parser.parse_args()


def transformed_vertices(scene: trimesh.Scene, geometry_name: str, vertices: np.ndarray) -> np.ndarray:
    nodes = scene.graph.geometry_nodes.get(geometry_name) or [geometry_name]
    transformed = []
    vertices_h = np.column_stack([vertices, np.ones(len(vertices), dtype=vertices.dtype)])
    for node in nodes:
        transform, _ = scene.graph.get(node)
        transformed.append((vertices_h @ transform.T)[:, :3])
    return np.concatenate(transformed, axis=0)


def geometry_colors(geometry, count: int) -> np.ndarray:
    colors = getattr(getattr(geometry, "visual", None), "vertex_colors", None)
    if colors is None or len(colors) == 0:
        return np.full((count, 3), 255, dtype=np.uint8)
    colors = np.asarray(colors)
    if colors.shape[1] >= 3:
        colors = colors[:, :3]
    if len(colors) != count:
        colors = np.resize(colors, (count, colors.shape[1]))
    return colors.astype(np.uint8, copy=False)


def collect_points(scene: trimesh.Scene, include_mesh_vertices: bool) -> tuple[np.ndarray, np.ndarray]:
    point_chunks = []
    color_chunks = []

    for name, geometry in scene.geometry.items():
        is_pointcloud = isinstance(geometry, trimesh.points.PointCloud)
        if not is_pointcloud and not include_mesh_vertices:
            continue
        if not hasattr(geometry, "vertices") or len(geometry.vertices) == 0:
            continue

        vertices = np.asarray(geometry.vertices, dtype=np.float32)
        vertices = transformed_vertices(scene, name, vertices)
        colors = geometry_colors(geometry, len(geometry.vertices))

        nodes = scene.graph.geometry_nodes.get(name) or [name]
        if len(nodes) > 1:
            colors = np.tile(colors, (len(nodes), 1))

        point_chunks.append(vertices)
        color_chunks.append(colors)

    if not point_chunks:
        raise ValueError("No point-cloud geometry found in the GLB.")

    points = np.concatenate(point_chunks, axis=0)
    colors = np.concatenate(color_chunks, axis=0)
    mask = np.isfinite(points).all(axis=1)
    return points[mask], colors[mask]


def limit_points(points: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    indices = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
    return points[indices], colors[indices]


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray, ascii_ply: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if ascii_ply:
        with path.open("w", encoding="ascii") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for point, color in zip(points, colors, strict=True):
                f.write(
                    f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )
        return

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    data = np.empty(len(points), dtype=dtype)
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        data.tofile(f)


def main() -> None:
    args = parse_args()
    scene = trimesh.load(args.input_glb, force="scene")
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(scene)

    points, colors = collect_points(scene, args.include_mesh_vertices)
    points, colors = limit_points(points, colors, args.max_points)
    write_ply(args.output_ply, points.astype(np.float32, copy=False), colors, args.ascii)
    print(f"Wrote {len(points):,} points to {args.output_ply}")


if __name__ == "__main__":
    main()

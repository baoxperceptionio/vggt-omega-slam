#!/usr/bin/env python3

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_fixed_pose_kv_slam import _camera_center, _estimate_sim3_umeyama, _transform_points_sim3
from vggt_omega.models import CausalVGGTOmega, VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher-forced 4+1 fixed-window evaluation.")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--checkpoint", default="checkpoints/VGGT-Omega-1B-512/model.pt")
    parser.add_argument("--fixed-checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/teacher_forced_fixed_window")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    image_paths = _expand_images(args.images)[: args.max_frames]
    if len(image_paths) < 5:
        raise ValueError("Need at least 5 images.")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    images = load_and_preprocess_images(image_paths, image_resolution=args.image_resolution).to(args.device)

    teacher = VGGTOmega().eval().to(args.device)
    teacher.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    student = CausalVGGTOmega().eval().to(args.device)
    missing, unexpected = student.load_state_dict(torch.load(args.fixed_checkpoint, map_location="cpu"), strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected fixed-checkpoint keys: {unexpected[:8]}")
    if missing:
        print(f"Warning: fixed-checkpoint missing {len(missing)} keys; first keys: {missing[:8]}")

    with torch.inference_mode():
        teacher_pred = teacher(images)
        teacher_pose = teacher_pred["pose_enc"].detach()
        teacher_ext, _ = encoding_to_camera(teacher_pose, teacher_pred["images"].shape[-2:])

        pred_pose = torch.empty_like(teacher_pose)
        pred_raw_pose = torch.empty_like(teacher_pose)
        pred_pose[:, :4] = teacher_pose[:, :4]
        pred_raw_pose[:, :4] = teacher_pose[:, :4]

        per_frame: list[dict] = []
        for frame_idx in range(4, len(image_paths)):
            window_indices = list(range(frame_idx - 4, frame_idx + 1))
            fixed_pose = torch.zeros((1, 5, teacher_pose.shape[-1]), device=args.device, dtype=teacher_pose.dtype)
            fixed_pose[:, :4] = teacher_pose[:, frame_idx - 4 : frame_idx]
            fixed_mask = torch.zeros((1, 5), device=args.device, dtype=torch.bool)
            fixed_mask[:, :4] = True

            student_pred, _ = student.forward_incremental(
                images[window_indices],
                None,
                fixed_pose_enc=fixed_pose,
                fixed_pose_mask=fixed_mask,
            )
            window_pose = student_pred["pose_enc"].detach()
            window_raw_pose = student_pred.get("pose_enc_model", window_pose).detach()
            pred_pose[:, frame_idx] = window_pose[:, 4]
            pred_raw_pose[:, frame_idx] = window_raw_pose[:, 4]

            pose_abs = (pred_pose[:, frame_idx] - teacher_pose[:, frame_idx]).abs()
            per_frame.append(
                {
                    "frame": frame_idx,
                    "start": frame_idx - 4,
                    "pose_l1_mean": float(pose_abs.mean().cpu()),
                    "pose_linf": float(pose_abs.max().cpu()),
                    "translation_l2_enc": float(
                        torch.linalg.norm(pred_pose[:, frame_idx, :3] - teacher_pose[:, frame_idx, :3]).cpu()
                    ),
                    "quat_angle_deg_enc": _quat_angle_deg(pred_pose[:, frame_idx, 3:7], teacher_pose[:, frame_idx, 3:7]),
                    "fov_l1_mean": float((pred_pose[:, frame_idx, 7:9] - teacher_pose[:, frame_idx, 7:9]).abs().mean().cpu()),
                }
            )

        pred_ext, _ = encoding_to_camera(pred_pose, teacher_pred["images"].shape[-2:])

    teacher_centers = _camera_centers(teacher_ext[0].detach().cpu().numpy())
    pred_centers = _camera_centers(pred_ext[0].detach().cpu().numpy())
    aligned_centers, sim3 = _align_centers(pred_centers, teacher_centers)
    ate = np.linalg.norm(aligned_centers - teacher_centers, axis=1)

    # For the first 4 bootstrap frames we copied teacher pose, so report target
    # window frames separately as the meaningful teacher-forced score.
    target_ate = ate[4:]
    summary = {
        "num_frames": len(image_paths),
        "num_windows": len(per_frame),
        "pose_l1_mean": float(np.mean([item["pose_l1_mean"] for item in per_frame])),
        "pose_l1_median": float(np.median([item["pose_l1_mean"] for item in per_frame])),
        "translation_l2_enc_mean": float(np.mean([item["translation_l2_enc"] for item in per_frame])),
        "quat_angle_deg_mean": float(np.mean([item["quat_angle_deg_enc"] for item in per_frame])),
        "quat_angle_deg_median": float(np.median([item["quat_angle_deg_enc"] for item in per_frame])),
        "ate_mean": float(target_ate.mean()),
        "ate_median": float(np.median(target_ate)),
        "ate_rmse": float(np.sqrt(np.mean(target_ate**2))),
        "ate_max": float(target_ate.max()),
        "worst_frames": [
            {"frame": int(idx), "ate": float(ate[idx])}
            for idx in np.argsort(ate[4:])[-10:][::-1] + 4
        ],
    }

    np.savez_compressed(
        out_dir / "teacher_forced_fixed_window.npz",
        image_paths=np.array(image_paths),
        teacher_pose=teacher_pose.detach().cpu().numpy(),
        pred_pose=pred_pose.detach().cpu().numpy(),
        pred_raw_pose=pred_raw_pose.detach().cpu().numpy(),
        teacher_centers=teacher_centers,
        pred_centers=pred_centers,
        pred_aligned_centers=aligned_centers,
        ate=ate,
        sim3_to_teacher=sim3,
        per_frame=np.array(per_frame, dtype=object),
    )
    (out_dir / "teacher_forced_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(out_dir, teacher_centers, aligned_centers, ate)

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_dir}")


def _expand_images(inputs: list[str]) -> list[str]:
    image_paths = []
    for item in inputs:
        matches = sorted(glob.glob(item))
        if matches:
            image_paths.extend(matches)
        elif os.path.isfile(item):
            image_paths.append(item)
    return sorted(dict.fromkeys(image_paths))


def _quat_angle_deg(predicted: torch.Tensor, target: torch.Tensor) -> float:
    predicted = F.normalize(predicted.float(), dim=-1, eps=1e-8)
    target = F.normalize(target.float(), dim=-1, eps=1e-8)
    dot = torch.sum(predicted * target, dim=-1).abs().clamp(0.0, 1.0)
    return float((2.0 * torch.acos(dot)).mean().detach().cpu() * 180.0 / math.pi)


def _camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    return np.stack([_camera_center(ext) for ext in extrinsics], axis=0).astype(np.float32)


def _align_centers(centers: np.ndarray, teacher_centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = _estimate_sim3_umeyama(centers, teacher_centers).astype(np.float32)
    return _transform_points_sim3(transform, centers).astype(np.float32), transform


def _plot(out_dir: Path, teacher_centers: np.ndarray, aligned_centers: np.ndarray, ate: np.ndarray) -> None:
    frames = np.arange(len(ate))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(teacher_centers[:, 0], teacher_centers[:, 2], label="full VGGT teacher", linewidth=2.2)
    axes[0].plot(aligned_centers[:, 0], aligned_centers[:, 2], label="teacher-forced fixed 4+1", linewidth=1.8)
    axes[0].scatter(teacher_centers[0, 0], teacher_centers[0, 2], c="black", s=35, label="start")
    axes[0].set_title("Teacher-forced fixed-window trajectory")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Z")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(frames[4:], ate[4:], color="tab:red")
    axes[1].set_title("ATE for predicted 5th frames")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("position error")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "teacher_forced_fixed_window.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

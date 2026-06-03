#!/usr/bin/env python3
"""Profile full-sequence VGGT-Omega inference against 4+1 window inference."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images


DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/VGGT-Omega-1B-512/model.pt"
DEFAULT_IMAGE_DIR = REPO_ROOT / "outputs/dji_0005_full_2fps/frames"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/profile_full_vs_4plus1"
DEFAULT_COUNTS = (10, 50, 100, 200, 300, 400, 500)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    image_paths = list_images(args.image_dir)
    if len(image_paths) < max(args.counts):
        raise ValueError(f"Need at least {max(args.counts)} images, found {len(image_paths)} in {args.image_dir}.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = VGGTOmega(enable_depth=not args.camera_only).eval()
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if args.camera_only:
        model_state = model.state_dict()
        checkpoint = {key: value for key, value in checkpoint.items() if key in model_state}
        model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model = model.to(args.device)

    # Load the largest requested clip once. Timings below cover model forward
    # only; preprocessing and host-to-device transfer are intentionally excluded.
    images = load_and_preprocess_images(
        image_paths[: max(args.counts)],
        image_resolution=args.image_resolution,
    ).to(args.device)

    warmup_count = min(args.window_size, images.shape[0])
    with torch.inference_mode():
        _ = model(images[:warmup_count])
        if args.device == "cuda":
            torch.cuda.synchronize()

    rows = []
    for count in args.counts:
        clip = images[:count]
        for mode in ("full", "4+1"):
            result = profile_mode(model, clip, mode=mode, window_size=args.window_size, device=args.device)
            result.update(
                {
                    "frames": int(count),
                    "mode": mode,
                    "seconds_per_frame": result["total_seconds"] / float(count),
                    "image_resolution": args.image_resolution,
                    "checkpoint": str(args.checkpoint),
                    "image_dir": str(args.image_dir),
                    "camera_only": bool(args.camera_only),
                    "window_size": int(args.window_size),
                }
            )
            rows.append(result)
            print(
                f"{mode:>4} n={count:>3}: "
                f"{result['total_seconds']:.3f}s total, "
                f"{result['seconds_per_frame']:.4f}s/frame, "
                f"{result['peak_allocated_gib']:.2f} GiB allocated",
                flush=True,
            )

    csv_path = output_dir / "full_vs_4plus1_profile.csv"
    json_path = output_dir / "full_vs_4plus1_profile.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved CSV: {csv_path}", flush=True)
    print(f"Saved JSON: {json_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--counts", type=int, nargs="+", default=list(DEFAULT_COUNTS))
    parser.add_argument("--window-size", type=int, default=5, help="Use 5 for 4+1 inference.")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--camera-only",
        action="store_true",
        help="Disable the dense depth head. By default, profile the full VGGT-Omega forward pass.",
    )
    return parser.parse_args()


def list_images(image_dir: Path) -> list[str]:
    suffixes = {".jpg", ".jpeg", ".png"}
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    paths = [str(path) for path in sorted(image_dir.iterdir()) if path.suffix.lower() in suffixes]
    if not paths:
        raise ValueError(f"No images found in {image_dir}")
    return paths


def profile_mode(
    model: VGGTOmega,
    images: torch.Tensor,
    *,
    mode: str,
    window_size: int,
    device: str,
) -> dict[str, float]:
    if window_size <= 1:
        raise ValueError("window_size must be greater than 1.")
    if mode not in {"full", "4+1"}:
        raise ValueError(f"Unknown profiling mode: {mode}")

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    started = time.perf_counter()
    with torch.inference_mode():
        if mode == "full":
            _ = model(images)
        else:
            clipped_window = min(window_size, int(images.shape[0]))
            _ = model(images[:clipped_window])
            for candidate_idx in range(clipped_window, int(images.shape[0])):
                window = images[candidate_idx - (window_size - 1) : candidate_idx + 1]
                _ = model(window)
        if device == "cuda":
            torch.cuda.synchronize()
    total_seconds = time.perf_counter() - started

    if device == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    else:
        peak_allocated = 0.0
        peak_reserved = 0.0

    return {
        "total_seconds": float(total_seconds),
        "peak_allocated_gib": float(peak_allocated),
        "peak_reserved_gib": float(peak_reserved),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "frames",
        "mode",
        "total_seconds",
        "seconds_per_frame",
        "peak_allocated_gib",
        "peak_reserved_gib",
        "image_resolution",
        "window_size",
        "camera_only",
        "checkpoint",
        "image_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


if __name__ == "__main__":
    main()

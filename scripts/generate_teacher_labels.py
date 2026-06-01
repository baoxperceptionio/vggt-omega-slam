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

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images


def generate_teacher_labels(
    image_paths: list[str],
    *,
    checkpoint: str,
    output: str,
    image_resolution: int = 512,
    device: str = "cuda",
) -> None:
    if len(image_paths) == 0:
        raise ValueError("No input images were provided.")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model = model.to(device)

    images = load_and_preprocess_images(image_paths, image_resolution=image_resolution).to(device)
    with torch.inference_mode():
        predictions = model(images)

    aggregator = model.aggregator
    with torch.inference_mode():
        aggregated_tokens_list, patch_token_start = aggregator(images.unsqueeze(0))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_paths": np.array(image_paths),
        "patch_token_start": np.array(patch_token_start),
    }
    for key in ("pose_enc", "depth", "depth_conf", "camera_and_register_tokens"):
        if key in predictions:
            payload[key] = predictions[key].detach().float().cpu().numpy()
    for layer_idx in (4, 11, 17, 23):
        tokens = aggregated_tokens_list[layer_idx]
        if tokens is not None:
            payload[f"tokens_layer_{layer_idx}"] = tokens.detach().float().cpu().numpy()

    np.savez_compressed(output_path, **payload)
    print(f"Saved teacher labels: {output_path}")


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
    parser = argparse.ArgumentParser(description="Generate full-sequence VGGT-Omega teacher labels.")
    parser.add_argument("images", nargs="+", help="Input image files or glob patterns.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    generate_teacher_labels(
        _expand_images(args.images),
        checkpoint=args.checkpoint,
        output=args.output,
        image_resolution=args.image_resolution,
        device=args.device,
    )


if __name__ == "__main__":
    main()

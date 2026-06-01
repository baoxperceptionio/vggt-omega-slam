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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggt_omega.models import CausalVGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images


def train_causal_student(
    image_paths: list[str],
    *,
    labels: str,
    output: str,
    init_checkpoint: str | None,
    image_resolution: int = 512,
    chunk_size: int = 2,
    epochs: int = 1,
    lr: float = 1e-5,
    device: str = "cuda",
) -> None:
    if len(image_paths) == 0:
        raise ValueError("No input images were provided.")
    if not os.path.isfile(labels):
        raise FileNotFoundError(f"Teacher label file not found: {labels}")

    model = CausalVGGTOmega().to(device).train()
    if init_checkpoint:
        state_dict = torch.load(init_checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    teacher = np.load(labels)
    teacher_tensors = {
        key: torch.from_numpy(teacher[key]).float().to(device)
        for key in teacher.files
        if key.startswith("tokens_layer_") or key in {"pose_enc", "depth", "depth_conf"}
    }
    images = load_and_preprocess_images(image_paths, image_resolution=image_resolution).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        state = model.init_slam_state()
        total_loss = torch.zeros((), device=device)

        for start in range(0, images.shape[0], chunk_size):
            end = min(start + chunk_size, images.shape[0])
            predictions, state = model.forward_incremental(images[start:end], state)
            total_loss = total_loss + _distillation_loss(
                predictions,
                teacher_tensors,
                start=start,
                end=end,
            )

        total_loss.backward()
        optimizer.step()
        print(f"epoch={epoch} loss={float(total_loss.detach().cpu()):.6f}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved causal student checkpoint: {output_path}")


def _distillation_loss(
    predictions: dict[str, torch.Tensor],
    teacher: dict[str, torch.Tensor],
    *,
    start: int,
    end: int,
) -> torch.Tensor:
    loss = torch.zeros((), device=next(iter(predictions.values())).device)

    if "pose_enc" in predictions and "pose_enc" in teacher:
        loss = loss + F.smooth_l1_loss(predictions["pose_enc"], teacher["pose_enc"][:, start:end])

    if "depth" in predictions and "depth" in teacher:
        target_depth = teacher["depth"][:, start:end]
        weight = teacher.get("depth_conf")
        if weight is not None:
            weight = weight[:, start:end][..., None].clamp_min(1.0)
            weight = weight / weight.detach().mean().clamp_min(1.0)
        else:
            weight = 1.0
        loss = loss + (weight * (predictions["depth"] - target_depth).abs()).mean()
        loss = loss + F.l1_loss(
            torch.log(predictions["depth"].clamp_min(1e-6)),
            torch.log(target_depth.clamp_min(1e-6)),
        )

    for layer_idx in (4, 11, 17, 23):
        key = f"tokens_layer_{layer_idx}"
        if key in teacher and "camera_and_register_tokens" in predictions and layer_idx == 23:
            pred_tokens = predictions["camera_and_register_tokens"]
            target_tokens = teacher[key][:, start:end, : pred_tokens.shape[2]]
            loss = loss + 0.1 * F.mse_loss(pred_tokens.float(), target_tokens.float())

    return loss


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
    parser = argparse.ArgumentParser(description="Train CausalVGGTOmega from teacher pseudo-labels.")
    parser.add_argument("images", nargs="+", help="Input image files or glob patterns.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    train_causal_student(
        _expand_images(args.images),
        labels=args.labels,
        output=args.output,
        init_checkpoint=args.init_checkpoint,
        image_resolution=args.image_resolution,
        chunk_size=args.chunk_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()

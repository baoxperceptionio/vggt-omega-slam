# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn

from vggt_omega.models.aggregator import Aggregator, CausalAggregator, init_slam_state
from vggt_omega.models.heads import CameraHead, DenseHead, TextAlignmentHead
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.utils.slam import unproject_depth_map_to_point_map_torch


class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega inference model for camera and depth prediction."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
    ) -> None:
        super().__init__()

        self.aggregator = Aggregator(patch_size=patch_size, embed_dim=embed_dim)
        _warn_if_rope_not_max(self.aggregator)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_depth else None
        self.text_alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            aggregated_tokens_list, patch_token_start = self.aggregator(images)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        predictions = {
            "camera_and_register_tokens": final_tokens[:, :, :patch_token_start].contiguous(),
        }
        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                predictions["pose_enc"] = self.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=patch_token_start,
                )

            if self.dense_head is not None:
                depth, depth_conf = self.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.text_alignment_head is not None:
                predictions.update(
                    self.text_alignment_head(
                        aggregated_tokens_list,
                        patch_token_start=patch_token_start,
                    )
                )

        if not self.training:
            predictions["images"] = images
        return predictions


def _warn_if_rope_not_max(aggregator: nn.Module) -> None:
    for name, module in (("aggregator.patch_embed", aggregator.patch_embed), ("aggregator", aggregator)):
        rope_embed = getattr(module, "rope_embed", None)
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released VGGT-Omega checkpoint was trained with 'max'.",
                stacklevel=2,
            )


class CausalVGGTOmega(nn.Module):
    """Cache-backed VGGT-Omega student for offline incremental SLAM inference."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        cached_layer_indices: tuple[int, ...] | None = None,
        register_attention_block_indices: list[int] | None = None,
        enable_camera: bool = True,
        enable_depth: bool = True,
    ) -> None:
        super().__init__()

        self.aggregator = CausalAggregator(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            cached_layer_indices=(4, 11, 17, 23) if cached_layer_indices is None else cached_layer_indices,
            register_attention_block_indices=(
                [2, 6, 9, 14, 20]
                if register_attention_block_indices is None
                else register_attention_block_indices
            ),
        )
        _warn_if_rope_not_max(self.aggregator)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_depth else None

    def init_slam_state(self) -> dict:
        return init_slam_state()

    def forward_incremental(self, images: torch.Tensor, state: dict | None = None) -> tuple[dict[str, torch.Tensor], dict]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        use_cuda_amp = images.is_cuda
        amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda_amp):
            aggregated_tokens_list, patch_token_start, state = self.aggregator.forward_incremental(images, state)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which CausalVGGTOmega needs.")

        predictions = {
            "camera_and_register_tokens": final_tokens[:, :, :patch_token_start].contiguous(),
        }
        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                predictions["pose_enc"] = self.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=patch_token_start,
                )

            if self.dense_head is not None:
                depth, depth_conf = self.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if "pose_enc" in predictions and "depth" in predictions:
                extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], images.shape[-2:])
                predictions["extrinsic"] = extrinsic
                predictions["intrinsic"] = intrinsic
                predictions["world_points_from_depth"] = unproject_depth_map_to_point_map_torch(
                    predictions["depth"],
                    extrinsic,
                    intrinsic,
                )

        if not self.training:
            predictions["images"] = images
        return predictions, state

    def step(
        self,
        images_chunk: torch.Tensor,
        state: dict | None = None,
        *,
        keyframe_stride: int = 8,
    ) -> tuple[dict[str, torch.Tensor], dict]:
        state = self.init_slam_state() if state is None else state
        start_frame = int(state.get("num_frames_seen", 0))
        predictions, state = self.forward_incremental(images_chunk, state)

        if keyframe_stride > 0 and "pose_enc" in predictions and "world_points_from_depth" in predictions:
            num_new = predictions["pose_enc"].shape[1]
            for local_idx in range(num_new):
                global_idx = start_frame + local_idx
                if global_idx % keyframe_stride == 0:
                    state["keyframe_poses"].append(predictions["pose_enc"][:, local_idx].detach().cpu())
                    state["keyframe_points"].append(
                        predictions["world_points_from_depth"][:, local_idx].detach().cpu()
                    )

        return predictions, state

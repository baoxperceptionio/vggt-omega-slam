# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt_omega.models.layers import Mlp, RopePositionEmbedding, SelfAttentionBlock
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """Alternating-attention encoder over video frames."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        register_attention_block_indices: list[int] = [2, 6, 9, 14, 20],
        cached_layer_indices: tuple[int, ...] = (4, 11, 17, 23),
    ) -> None:
        super().__init__()

        self.patch_embed = _build_patch_embed(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100,
            normalize_coords="max",
            dtype=torch.float32,
        )

        self.frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )
        self.inter_frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.patch_size = patch_size
        self.cached_layer_indices = set(cached_layer_indices)
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, num_register_tokens, embed_dim))
        self.patch_token_start = 1 + num_register_tokens

        self.inter_frame_attention_types = ["global"] * depth
        for idx in register_attention_block_indices:
            if idx < 0 or idx >= depth:
                raise ValueError(f"register_attention_block_indices contains invalid block index {idx}")
            self.inter_frame_attention_types[idx] = "register"

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.camera_token, std=1e-3)
        nn.init.normal_(self.register_token, std=1e-3)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[list[torch.Tensor | None], int]:
        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * num_frames, num_channels, height, width)

        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)

        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape

        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        outputs = []
        for block_idx in range(self.depth):
            tokens, frame_tokens = self._run_frame_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                frame_rope,
            )
            tokens = self._run_inter_frame_attention_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.inter_frame_attention_types[block_idx],
            )
            if block_idx in self.cached_layer_indices:
                outputs.append(torch.cat([frame_tokens, tokens], dim=-1))
            else:
                outputs.append(None)

        return outputs, self.patch_token_start

    def _run_frame_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
        tokens = self.frame_blocks[block_idx](tokens, rope_sincos)
        return tokens, tokens.view(batch_size, num_frames, num_tokens, embed_dim)

    def _run_inter_frame_attention_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        attention_type: str,
    ) -> torch.Tensor:
        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type == "global":
            tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
            tokens = self.inter_frame_blocks[block_idx](tokens, None)
            return tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")

        patch_token_start = self.patch_token_start
        camera_and_register_tokens = tokens[:, :, :patch_token_start].reshape(
            batch_size,
            num_frames * patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, :, patch_token_start:].reshape(
            batch_size,
            num_frames * (num_tokens - patch_token_start),
            embed_dim,
        )

        camera_and_register_tokens = self.inter_frame_blocks[block_idx](camera_and_register_tokens, None)
        tokens = torch.cat([camera_and_register_tokens, patch_tokens], dim=1)

        camera_and_register_tokens = tokens[:, : num_frames * patch_token_start].view(
            batch_size,
            num_frames,
            patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, num_frames * patch_token_start :].view(
            batch_size,
            num_frames,
            num_tokens - patch_token_start,
            embed_dim,
        )
        return torch.cat([camera_and_register_tokens, patch_tokens], dim=2)


class CausalInterFrameSelfAttention(nn.Module):
    """Self-attention with append-only KV cache and frame-causal masking."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.source = source

    @property
    def num_heads(self) -> int:
        return self.source.num_heads

    def forward_incremental(
        self,
        x: torch.Tensor,
        *,
        past_k: torch.Tensor | None,
        past_v: torch.Tensor | None,
        num_current_frames: int,
        tokens_per_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, dim = x.shape
        if num_tokens != num_current_frames * tokens_per_frame:
            raise ValueError(
                "Incremental attention expected a frame-major sequence with "
                f"{num_current_frames}*{tokens_per_frame} tokens, got {num_tokens}."
            )

        qkv = self.source.qkv(x)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.source.num_heads, dim // self.source.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        if self.source.use_qk_norm:
            q = self.source.q_norm(q)
            k = self.source.k_norm(k)

        if past_k is not None:
            if past_v is None:
                raise ValueError("past_v is required when past_k is provided")
            k_all = torch.cat([past_k, k], dim=2)
            v_all = torch.cat([past_v, v], dim=2)
        else:
            k_all = k
            v_all = v

        attn_mask = _build_frame_causal_mask(
            num_current_frames=num_current_frames,
            tokens_per_frame=tokens_per_frame,
            past_tokens=0 if past_k is None else past_k.shape[2],
            device=x.device,
        )
        x = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(batch_size, num_tokens, dim)
        x = self.source.proj(x)
        x = self.source.proj_drop(x)
        return x, k.detach(), v.detach()


class CausalSelfAttentionBlockAdapter(nn.Module):
    """Incremental counterpart of SelfAttentionBlock that reuses its weights."""

    def __init__(self, source: SelfAttentionBlock) -> None:
        super().__init__()
        self.source = source
        self.attn = CausalInterFrameSelfAttention(source.attn)

    def forward_incremental(
        self,
        x: torch.Tensor,
        *,
        past_k: torch.Tensor | None,
        past_v: torch.Tensor | None,
        num_current_frames: int,
        tokens_per_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.source.training and self.source.sample_drop_ratio > 0.0:
            raise RuntimeError("Causal incremental attention does not support stochastic depth training mode.")

        residual, current_k, current_v = self.attn.forward_incremental(
            self.source.norm1(x),
            past_k=past_k,
            past_v=past_v,
            num_current_frames=num_current_frames,
            tokens_per_frame=tokens_per_frame,
        )
        x_attn = x + self.source.ls1(residual)
        x_ffn = x_attn + self.source.ls2(self.source.mlp(self.source.norm2(x_attn)))
        return x_ffn, current_k, current_v


class CausalAggregator(Aggregator):
    """Frame-causal, cache-backed aggregator for offline incremental SLAM."""

    def __init__(self, *args, pose_dim: int = 9, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fixed_pose_conditioner = FixedPoseConditioner(pose_dim=pose_dim, embed_dim=self.camera_token.shape[-1])

    def forward_incremental(
        self,
        images: torch.Tensor,
        state: dict | None = None,
        *,
        fixed_pose_enc: torch.Tensor | None = None,
        fixed_pose_mask: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor | None], int, dict]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        state = _normalize_causal_state(state, device=images.device)
        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")
        if batch_size != 1:
            raise ValueError("The first CausalVGGTOmega SLAM state supports batch_size=1.")

        image_size_hw = (int(height), int(width))
        if state["image_size_hw"] is None:
            state["image_size_hw"] = image_size_hw
        elif tuple(state["image_size_hw"]) != image_size_hw:
            raise ValueError(
                f"All incremental chunks must have the same image size. "
                f"State has {state['image_size_hw']}, got {image_size_hw}."
            )

        images = (images - self._resnet_mean) / self._resnet_std
        flat_images = images.view(batch_size * num_frames, num_channels, height, width)

        first_global_frame_idx = int(state["num_frames_seen"])
        camera_token = slice_expand_and_flatten_from_offset(
            self.camera_token,
            batch_size,
            num_frames,
            first_global_frame_idx=first_global_frame_idx,
        )
        register_token = slice_expand_and_flatten_from_offset(
            self.register_token,
            batch_size,
            num_frames,
            first_global_frame_idx=first_global_frame_idx,
        )

        patch_tokens = self.patch_embed(flat_images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape
        tokens = self.fixed_pose_conditioner(
            tokens,
            fixed_pose_enc=fixed_pose_enc,
            fixed_pose_mask=fixed_pose_mask,
            batch_size=batch_size,
            num_frames=num_frames,
            num_tokens=num_tokens,
            patch_token_start=self.patch_token_start,
        )

        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        outputs = []
        layer_kv_cache = state["layer_kv_cache"]
        for block_idx in range(self.depth):
            tokens, frame_tokens = self._run_frame_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                frame_rope,
            )
            tokens = self._run_inter_frame_attention_block_incremental(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.inter_frame_attention_types[block_idx],
                layer_kv_cache,
            )
            tokens = self.fixed_pose_conditioner.condition_tokens(
                tokens,
                fixed_pose_enc=fixed_pose_enc,
                fixed_pose_mask=fixed_pose_mask,
                batch_size=batch_size,
                num_frames=num_frames,
                num_tokens=num_tokens,
                patch_token_start=self.patch_token_start,
                projection="deep",
            )
            if block_idx in self.cached_layer_indices:
                tokens_for_cache = tokens.view(batch_size, num_frames, num_tokens, embed_dim)
                cached_tokens = torch.cat([frame_tokens, tokens_for_cache], dim=-1)
                cached_tokens = self.fixed_pose_conditioner.condition_cached_output(
                    cached_tokens,
                    fixed_pose_enc=fixed_pose_enc,
                    fixed_pose_mask=fixed_pose_mask,
                    patch_token_start=self.patch_token_start,
                )
                outputs.append(cached_tokens)
            else:
                outputs.append(None)

        state["num_frames_seen"] += int(num_frames)
        return outputs, self.patch_token_start, state

    def _run_inter_frame_attention_block_incremental(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        attention_type: str,
        layer_kv_cache: dict,
    ) -> torch.Tensor:
        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type == "global":
            current = tokens.reshape(batch_size, num_frames * num_tokens, embed_dim)
            out = self._run_cached_inter_block(
                current,
                block_idx=block_idx,
                layer_kv_cache=layer_kv_cache,
                num_current_frames=num_frames,
                tokens_per_frame=num_tokens,
            )
            return out.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")

        patch_token_start = self.patch_token_start
        camera_and_register_tokens = tokens[:, :, :patch_token_start].reshape(
            batch_size,
            num_frames * patch_token_start,
            embed_dim,
        )
        camera_and_register_tokens = self._run_cached_inter_block(
            camera_and_register_tokens,
            block_idx=block_idx,
            layer_kv_cache=layer_kv_cache,
            num_current_frames=num_frames,
            tokens_per_frame=patch_token_start,
        ).view(batch_size, num_frames, patch_token_start, embed_dim)
        return torch.cat([camera_and_register_tokens, tokens[:, :, patch_token_start:]], dim=2)

    def _run_cached_inter_block(
        self,
        tokens: torch.Tensor,
        *,
        block_idx: int,
        layer_kv_cache: dict,
        num_current_frames: int,
        tokens_per_frame: int,
    ) -> torch.Tensor:
        cache = layer_kv_cache.get(block_idx, {})
        block = CausalSelfAttentionBlockAdapter(self.inter_frame_blocks[block_idx])
        out, current_k, current_v = block.forward_incremental(
            tokens,
            past_k=cache.get("k"),
            past_v=cache.get("v"),
            num_current_frames=num_current_frames,
            tokens_per_frame=tokens_per_frame,
        )

        if "tokens_per_frame" in cache and cache["tokens_per_frame"] != tokens_per_frame:
            raise ValueError(
                f"Layer {block_idx} cache token width changed from "
                f"{cache['tokens_per_frame']} to {tokens_per_frame}."
            )
        layer_kv_cache[block_idx] = {
            "k": current_k if cache.get("k") is None else torch.cat([cache["k"], current_k], dim=2).detach(),
            "v": current_v if cache.get("v") is None else torch.cat([cache["v"], current_v], dim=2).detach(),
            "tokens_per_frame": tokens_per_frame,
        }
        return out


class FixedPoseConditioner(nn.Module):
    """No-op-initialized adapter that marks known SLAM poses inside history tokens."""

    def __init__(self, pose_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.pose_norm = nn.LayerNorm(pose_dim)
        self.pose_proj = nn.Linear(pose_dim, embed_dim)
        self.deep_pose_proj = nn.Linear(pose_dim, embed_dim)
        self.cached_pose_proj = nn.Linear(pose_dim, 2 * embed_dim)
        nn.init.zeros_(self.pose_proj.weight)
        nn.init.zeros_(self.pose_proj.bias)
        nn.init.zeros_(self.deep_pose_proj.weight)
        nn.init.zeros_(self.deep_pose_proj.bias)
        nn.init.zeros_(self.cached_pose_proj.weight)
        nn.init.zeros_(self.cached_pose_proj.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        fixed_pose_enc: torch.Tensor | None,
        fixed_pose_mask: torch.Tensor | None,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        patch_token_start: int,
    ) -> torch.Tensor:
        if fixed_pose_enc is None:
            if fixed_pose_mask is not None:
                raise ValueError("fixed_pose_mask requires fixed_pose_enc.")
            return tokens

        if fixed_pose_enc.shape[:2] != (batch_size, num_frames):
            raise ValueError(
                "fixed_pose_enc must have shape "
                f"({batch_size}, {num_frames}, pose_dim), got {tuple(fixed_pose_enc.shape)}."
            )
        if fixed_pose_mask is None:
            fixed_pose_mask = torch.ones(
                batch_size,
                num_frames,
                dtype=torch.bool,
                device=fixed_pose_enc.device,
            )
        if fixed_pose_mask.shape != (batch_size, num_frames):
            raise ValueError(
                "fixed_pose_mask must have shape "
                f"({batch_size}, {num_frames}), got {tuple(fixed_pose_mask.shape)}."
            )

        return self.condition_tokens(
            tokens,
            fixed_pose_enc=fixed_pose_enc,
            fixed_pose_mask=fixed_pose_mask,
            batch_size=batch_size,
            num_frames=num_frames,
            num_tokens=num_tokens,
            patch_token_start=patch_token_start,
            projection="input",
        )

    def condition_tokens(
        self,
        tokens: torch.Tensor,
        *,
        fixed_pose_enc: torch.Tensor | None,
        fixed_pose_mask: torch.Tensor | None,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        patch_token_start: int,
        projection: str,
    ) -> torch.Tensor:
        if fixed_pose_enc is None:
            return tokens

        if projection == "input":
            pose_proj = self.pose_proj
        elif projection == "deep":
            pose_proj = self.deep_pose_proj
        else:
            raise ValueError(f"Unknown fixed-pose token projection: {projection}")

        pose_delta = self._project_pose_delta(
            fixed_pose_enc,
            fixed_pose_mask,
            pose_proj=pose_proj,
            tokens=tokens,
            batch_size=batch_size,
            num_frames=num_frames,
        )
        tokens = tokens.view(batch_size, num_frames, num_tokens, -1).clone()
        tokens[:, :, :patch_token_start] = tokens[:, :, :patch_token_start] + pose_delta.unsqueeze(2)
        return tokens.view(batch_size * num_frames, num_tokens, -1)

    def condition_cached_output(
        self,
        cached_tokens: torch.Tensor,
        *,
        fixed_pose_enc: torch.Tensor | None,
        fixed_pose_mask: torch.Tensor | None,
        patch_token_start: int,
    ) -> torch.Tensor:
        if fixed_pose_enc is None:
            return cached_tokens

        batch_size, num_frames, _, _ = cached_tokens.shape
        pose_delta = self._project_pose_delta(
            fixed_pose_enc,
            fixed_pose_mask,
            pose_proj=self.cached_pose_proj,
            tokens=cached_tokens,
            batch_size=batch_size,
            num_frames=num_frames,
        )
        cached_tokens = cached_tokens.clone()
        cached_tokens[:, :, :patch_token_start] = cached_tokens[:, :, :patch_token_start] + pose_delta.unsqueeze(2)
        return cached_tokens

    def _project_pose_delta(
        self,
        fixed_pose_enc: torch.Tensor,
        fixed_pose_mask: torch.Tensor | None,
        *,
        pose_proj: nn.Linear,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
    ) -> torch.Tensor:
        if fixed_pose_enc.shape[:2] != (batch_size, num_frames):
            raise ValueError(
                "fixed_pose_enc must have shape "
                f"({batch_size}, {num_frames}, pose_dim), got {tuple(fixed_pose_enc.shape)}."
            )
        if fixed_pose_mask is None:
            fixed_pose_mask = torch.ones(
                batch_size,
                num_frames,
                dtype=torch.bool,
                device=fixed_pose_enc.device,
            )
        if fixed_pose_mask.shape != (batch_size, num_frames):
            raise ValueError(
                "fixed_pose_mask must have shape "
                f"({batch_size}, {num_frames}), got {tuple(fixed_pose_mask.shape)}."
            )

        fixed_pose_enc = fixed_pose_enc.to(device=tokens.device, dtype=torch.float32)
        fixed_pose_mask = fixed_pose_mask.to(device=tokens.device, dtype=torch.bool)
        pose_delta = pose_proj(self.pose_norm(fixed_pose_enc)).to(dtype=tokens.dtype)
        return pose_delta * fixed_pose_mask.unsqueeze(-1).to(dtype=pose_delta.dtype)


def init_slam_state() -> dict:
    """Create the public mutable state used by CausalVGGTOmega incremental inference."""

    return {
        "num_frames_seen": 0,
        "layer_kv_cache": {},
        "keyframe_poses": [],
        "keyframe_points": [],
        "image_size_hw": None,
    }


def _normalize_causal_state(state: dict | None, device: torch.device) -> dict:
    if state is None:
        return init_slam_state()

    normalized = init_slam_state()
    normalized.update(state)
    normalized["layer_kv_cache"] = dict(normalized.get("layer_kv_cache") or {})
    for layer_idx, cache in list(normalized["layer_kv_cache"].items()):
        if "k" in cache and cache["k"] is not None:
            cache["k"] = cache["k"].to(device=device)
        if "v" in cache and cache["v"] is not None:
            cache["v"] = cache["v"].to(device=device)
    return normalized


def _build_frame_causal_mask(
    *,
    num_current_frames: int,
    tokens_per_frame: int,
    past_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    current_tokens = num_current_frames * tokens_per_frame
    query_frame = torch.arange(current_tokens, device=device) // tokens_per_frame
    key_frame = torch.arange(current_tokens, device=device) // tokens_per_frame
    current_mask = key_frame[None, :] <= query_frame[:, None]
    if past_tokens > 0:
        past_mask = torch.ones(current_tokens, past_tokens, dtype=torch.bool, device=device)
        current_mask = torch.cat([past_mask, current_mask], dim=1)
    return current_mask.unsqueeze(0).unsqueeze(0)


def _build_patch_embed(
    patch_size: int,
    embed_dim: int,
    depth: int = 24,
    num_heads: int = 16,
    mlp_ratio: float = 4.0,
) -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        ffn_ratio=mlp_ratio,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    )
    model.init_weights()
    return model


def slice_expand_and_flatten(token_tensor: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
    first_frame_token = token_tensor[:, 0:1].expand(batch_size, 1, *token_tensor.shape[2:])
    other_frame_tokens = token_tensor[:, 1:].expand(batch_size, num_frames - 1, *token_tensor.shape[2:])
    tokens = torch.cat([first_frame_token, other_frame_tokens], dim=1)
    return tokens.view(batch_size * num_frames, *tokens.shape[2:])


def slice_expand_and_flatten_from_offset(
    token_tensor: torch.Tensor,
    batch_size: int,
    num_frames: int,
    *,
    first_global_frame_idx: int,
) -> torch.Tensor:
    if first_global_frame_idx < 0:
        raise ValueError("first_global_frame_idx must be non-negative")
    if first_global_frame_idx == 0:
        return slice_expand_and_flatten(token_tensor, batch_size, num_frames)

    tokens = token_tensor[:, 1:].expand(batch_size, num_frames, *token_tensor.shape[2:])
    return tokens.reshape(batch_size * num_frames, *tokens.shape[2:])

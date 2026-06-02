#!/usr/bin/env python3

import argparse
import glob
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggt_omega.models import CausalVGGTOmega, VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera, extri_intri_to_pose_encoding


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class FrameSequence:
    name: str
    frames: list[str]


@dataclass(frozen=True)
class ClipSample:
    sequence_index: int
    start: int


@dataclass(frozen=True)
class CachedClip:
    sequence: str
    start: int
    path: str
    clip_length: int
    cache_dir: str
    subclip_start: int = 0
    cache_type: str = "clip"


@dataclass(frozen=True)
class TrainingStage:
    name: str
    clip_length: int
    fixed_frames: int | None
    items: list[Any]
    epochs: int
    max_steps: int | None


@dataclass(frozen=True)
class PoseLoss:
    total: torch.Tensor
    translation: torch.Tensor
    rotation: torch.Tensor
    fov: torch.Tensor


@dataclass(frozen=True)
class TokenLoss:
    total: torch.Tensor
    fixed: torch.Tensor
    target: torch.Tensor


def prepare_teacher_cache(
    frame_sequences: list[FrameSequence],
    *,
    teacher_checkpoint: str,
    cache_dir: str,
    image_resolution: int = 512,
    clip_length: int = 5,
    stride: int = 1,
    overwrite_cache: bool = False,
    cache_teacher_tokens: bool = True,
    device: str = "cuda",
    log_every: int = 10,
) -> list[CachedClip]:
    if not frame_sequences:
        raise ValueError("No frame sequences were provided.")
    if clip_length < 2:
        raise ValueError("clip_length must be at least 2.")
    if stride < 1:
        raise ValueError("stride must be at least 1.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    samples = _build_clip_index(frame_sequences, clip_length, stride=stride)
    if not samples:
        raise ValueError(f"No sequence has at least {clip_length} frames.")

    teacher = VGGTOmega().eval().to(device)
    teacher.load_state_dict(torch.load(teacher_checkpoint, map_location="cpu"))
    for param in teacher.parameters():
        param.requires_grad_(False)

    clips: list[CachedClip] = []
    cache_times: list[dict[str, Any]] = []
    print(
        f"Preparing teacher cache: sequences={len(frame_sequences)} clips={len(samples)} "
        f"clip_length={clip_length} stride={stride} resolution={image_resolution} cache_dir={cache_root}"
    )

    for index, sample in enumerate(samples):
        sequence = frame_sequences[sample.sequence_index]
        output_path = cache_root / sequence.name / f"clip_{sample.start:06d}.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        clips.append(CachedClip(sequence.name, sample.start, str(output_path), clip_length, str(cache_root)))
        if output_path.exists() and not overwrite_cache:
            continue

        total_start = time.perf_counter()
        clip_paths = sequence.frames[sample.start : sample.start + clip_length]

        load_start = time.perf_counter()
        images = load_and_preprocess_images(clip_paths, image_resolution=image_resolution).to(device, non_blocking=True)
        _sync_cuda(device)
        load_s = time.perf_counter() - load_start

        teacher_start = time.perf_counter()
        with torch.no_grad():
            teacher_predictions = teacher(images)
            teacher_pose = teacher_predictions["pose_enc"].detach().clone()
        _sync_cuda(device)
        teacher_s = time.perf_counter() - teacher_start

        save_start = time.perf_counter()
        payload = {
            "sequence": sequence.name,
            "start": sample.start,
            "frame_paths": clip_paths,
            "image_resolution": image_resolution,
            "clip_length": clip_length,
            "images": images.detach().cpu().to(torch.float16),
            "teacher_pose": teacher_pose.detach().cpu().to(torch.float32),
        }
        if cache_teacher_tokens:
            payload["teacher_camera_and_register_tokens"] = (
                teacher_predictions["camera_and_register_tokens"].detach().cpu().to(torch.float16)
            )
        torch.save(payload, output_path)
        save_s = time.perf_counter() - save_start
        total_s = time.perf_counter() - total_start
        cache_times.append(
            {
                "index": index,
                "sequence": sequence.name,
                "start": sample.start,
                "clip_length": clip_length,
                "load_s": load_s,
                "teacher_s": teacher_s,
                "save_s": save_s,
                "total_s": total_s,
            }
        )
        if index == 0 or (index + 1) % max(log_every, 1) == 0 or index + 1 == len(samples):
            print(
                f"cache={index + 1:04d}/{len(samples)} seq={sequence.name} start={sample.start} "
                f"load={load_s:.3f}s teacher={teacher_s:.3f}s save={save_s:.3f}s total={total_s:.3f}s"
            )

    manifest = {
        "version": 1,
        "teacher_checkpoint": teacher_checkpoint,
        "image_resolution": image_resolution,
        "clip_length": clip_length,
        "stride": stride,
        "cache_teacher_tokens": cache_teacher_tokens,
        "sequences": [{"name": seq.name, "num_frames": len(seq.frames)} for seq in frame_sequences],
        "clips": [
            {
                "sequence": clip.sequence,
                "start": clip.start,
                "path": clip.path,
                "clip_length": clip.clip_length,
                "cache_dir": clip.cache_dir,
            }
            for clip in clips
        ],
        "cache_profile": cache_times,
    }
    (cache_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved teacher cache manifest: {cache_root / MANIFEST_NAME}")

    return clips


def prepare_global_window_teacher_cache(
    frame_sequences: list[FrameSequence],
    *,
    teacher_checkpoint: str,
    cache_dir: str,
    image_resolution: int = 512,
    teacher_window_length: int = 100,
    teacher_window_stride: int | None = None,
    subclip_lengths: list[int] | None = None,
    subclip_stride: int = 1,
    canonicalize_subclips: bool = False,
    cache_full_windows: bool = True,
    overwrite_cache: bool = False,
    cache_images: bool = False,
    cache_teacher_tokens: bool = True,
    device: str = "cuda",
    log_every: int = 1,
) -> list[CachedClip]:
    if not frame_sequences:
        raise ValueError("No frame sequences were provided.")
    if teacher_window_length < 2:
        raise ValueError("teacher_window_length must be at least 2.")
    if subclip_stride < 1:
        raise ValueError("subclip_stride must be at least 1.")
    subclip_lengths = subclip_lengths or [5]
    if any(length < 2 or length > teacher_window_length for length in subclip_lengths):
        raise ValueError("Every subclip length must be in [2, teacher_window_length].")
    if cache_full_windows and canonicalize_subclips:
        raise ValueError("Full-window cache stores model-gauge teacher outputs and cannot canonicalize subclips.")
    max_exhaustive_stride = _max_exhaustive_teacher_window_stride(teacher_window_length, subclip_lengths)
    if teacher_window_stride is None:
        teacher_window_stride = max_exhaustive_stride
    if teacher_window_stride < 1:
        raise ValueError("teacher_window_stride must be at least 1.")
    if teacher_window_stride > max_exhaustive_stride:
        raise ValueError(
            "teacher_window_stride would skip cross-window subclips. "
            f"Use <= {max_exhaustive_stride} for exhaustive coverage with subclip_lengths={subclip_lengths}."
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    teacher = VGGTOmega().eval().to(device)
    teacher.load_state_dict(torch.load(teacher_checkpoint, map_location="cpu"))
    for param in teacher.parameters():
        param.requires_grad_(False)

    clips: list[CachedClip] = []
    window_profiles: list[dict[str, Any]] = []
    window_starts_by_sequence = {
        seq.name: _build_teacher_window_starts(len(seq.frames), teacher_window_length, teacher_window_stride)
        for seq in frame_sequences
    }
    window_count = sum(len(starts) for starts in window_starts_by_sequence.values())
    print(
        f"Preparing global-window teacher cache: sequences={len(frame_sequences)} windows={window_count} "
        f"teacher_window_length={teacher_window_length} teacher_window_stride={teacher_window_stride} "
        f"subclip_lengths={subclip_lengths} subclip_stride={subclip_stride} "
        f"canonicalize_subclips={canonicalize_subclips} cache_full_windows={cache_full_windows} "
        f"cache_teacher_tokens={cache_teacher_tokens} cache_dir={cache_root}"
    )

    seen_subclips: set[tuple[str, int, int]] = set()
    window_index = 0
    for sequence in frame_sequences:
        for window_start in window_starts_by_sequence[sequence.name]:
            total_start = time.perf_counter()
            window_frames = sequence.frames[window_start : window_start + teacher_window_length]
            current_window_length = len(window_frames)
            full_window_path = cache_root / sequence.name / f"window_{window_start:06d}.pt"

            virtual_subclips = 0
            for subclip_length in subclip_lengths:
                for subclip_start in range(0, current_window_length - subclip_length + 1, subclip_stride):
                    global_start = window_start + subclip_start
                    subclip_key = (sequence.name, subclip_length, global_start)
                    if subclip_key in seen_subclips:
                        continue
                    seen_subclips.add(subclip_key)
                    if cache_full_windows:
                        output_path = full_window_path
                        cache_type = "global_window_full"
                    else:
                        output_path = (
                            cache_root
                            / sequence.name
                            / f"window_{window_start:06d}"
                            / f"clip_L{subclip_length:03d}_{global_start:06d}.pt"
                        )
                        cache_type = "global_window_subclip"
                    clips.append(
                        CachedClip(
                            sequence.name,
                            global_start,
                            str(output_path),
                            subclip_length,
                            str(cache_root),
                            subclip_start=subclip_start,
                            cache_type=cache_type,
                        )
                    )
                    virtual_subclips += 1

            if cache_full_windows and full_window_path.exists() and not overwrite_cache:
                load_s = 0.0
                teacher_s = 0.0
                save_s = 0.0
                total_s = time.perf_counter() - total_start
                window_profiles.append(
                    {
                        "window_index": window_index,
                        "sequence": sequence.name,
                        "window_start": window_start,
                        "window_length": current_window_length,
                        "num_subclips": virtual_subclips,
                        "load_s": load_s,
                        "teacher_s": teacher_s,
                        "save_s": save_s,
                        "total_s": total_s,
                        "cache_hit": True,
                    }
                )
                window_index += 1
                if window_index == 1 or window_index % max(log_every, 1) == 0 or window_index == window_count:
                    print(
                        f"window={window_index:04d}/{window_count} seq={sequence.name} start={window_start} "
                        f"subclips={virtual_subclips} cache_hit=True total={total_s:.3f}s"
                    )
                continue

            load_start = time.perf_counter()
            window_images = load_and_preprocess_images(window_frames, image_resolution=image_resolution).to(device, non_blocking=True)
            _sync_cuda(device)
            load_s = time.perf_counter() - load_start

            teacher_start = time.perf_counter()
            with torch.no_grad():
                teacher_predictions = teacher(window_images)
                teacher_extrinsic, teacher_intrinsic = encoding_to_camera(
                    teacher_predictions["pose_enc"],
                    teacher_predictions["images"].shape[-2:],
                )
                # Re-encode from the full-window camera solution. Each subclip
                # then inherits this 100-frame teacher coordinate frame instead
                # of getting its own first-frame-origin local VGGT frame.
                teacher_pose_global = extri_intri_to_pose_encoding(
                    teacher_extrinsic,
                    teacher_intrinsic,
                    teacher_predictions["images"].shape[-2:],
                ).detach().clone()
            _sync_cuda(device)
            teacher_s = time.perf_counter() - teacher_start

            save_start = time.perf_counter()
            saved_subclips = 0
            if cache_full_windows:
                full_window_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "cache_type": "global_window_full",
                    "sequence": sequence.name,
                    "start": window_start,
                    "window_start": window_start,
                    "window_length": current_window_length,
                    "frame_paths": window_frames,
                    "image_resolution": image_resolution,
                    "coordinate_frame": "full_vggt_teacher_window",
                    "canonicalized_to_first_frame": False,
                    "teacher_pose": teacher_pose_global.detach().cpu().to(torch.float32),
                }
                if cache_images:
                    payload["images"] = window_images.detach().cpu().to(torch.float16)
                if cache_teacher_tokens:
                    payload["teacher_camera_and_register_tokens"] = (
                        teacher_predictions["camera_and_register_tokens"].detach().cpu().to(torch.float16)
                    )
                torch.save(payload, full_window_path)
                saved_subclips = virtual_subclips
            else:
                for clip in clips[-virtual_subclips:]:
                    output_path = Path(clip.path)
                    if output_path.exists() and not overwrite_cache:
                        saved_subclips += 1
                        continue
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    clip_slice = slice(clip.subclip_start, clip.subclip_start + clip.clip_length)
                    if canonicalize_subclips:
                        subclip_extrinsic = _canonicalize_extrinsics_to_first_camera(
                            teacher_extrinsic[:, clip_slice]
                        )
                        subclip_pose = extri_intri_to_pose_encoding(
                            subclip_extrinsic,
                            teacher_intrinsic[:, clip_slice],
                            teacher_predictions["images"].shape[-2:],
                        )
                        coordinate_frame = "subclip_first_camera_canonical_from_full_vggt_teacher_window"
                    else:
                        subclip_pose = teacher_pose_global[:, clip_slice]
                        coordinate_frame = "full_vggt_teacher_window"
                    payload = {
                        "sequence": sequence.name,
                        "start": clip.start,
                        "window_start": window_start,
                        "window_length": current_window_length,
                        "subclip_start": clip.subclip_start,
                        "frame_paths": window_frames[clip_slice],
                        "image_resolution": image_resolution,
                        "clip_length": clip.clip_length,
                        "coordinate_frame": coordinate_frame,
                        "canonicalized_to_first_frame": canonicalize_subclips,
                        "teacher_pose": subclip_pose.detach().cpu().to(torch.float32),
                    }
                    if cache_images:
                        payload["images"] = window_images[clip_slice].detach().cpu().to(torch.float16)
                    if cache_teacher_tokens:
                        payload["teacher_camera_and_register_tokens"] = (
                            teacher_predictions["camera_and_register_tokens"][:, clip_slice].detach().cpu().to(torch.float16)
                        )
                    torch.save(payload, output_path)
                    saved_subclips += 1
            save_s = time.perf_counter() - save_start
            total_s = time.perf_counter() - total_start
            window_profiles.append(
                {
                    "window_index": window_index,
                    "sequence": sequence.name,
                    "window_start": window_start,
                    "window_length": current_window_length,
                    "num_subclips": saved_subclips,
                    "load_s": load_s,
                    "teacher_s": teacher_s,
                    "save_s": save_s,
                    "total_s": total_s,
                }
            )
            window_index += 1
            if window_index == 1 or window_index % max(log_every, 1) == 0 or window_index == window_count:
                print(
                    f"window={window_index:04d}/{window_count} seq={sequence.name} start={window_start} "
                    f"subclips={saved_subclips} load={load_s:.3f}s teacher={teacher_s:.3f}s "
                    f"save={save_s:.3f}s total={total_s:.3f}s"
                )

    manifest = {
        "version": 3 if cache_full_windows else 2,
        "cache_type": "global_window_full" if cache_full_windows else "global_window_subclips",
        "teacher_checkpoint": teacher_checkpoint,
        "image_resolution": image_resolution,
        "teacher_window_length": teacher_window_length,
        "teacher_window_stride": teacher_window_stride,
        "subclip_lengths": subclip_lengths,
        "subclip_stride": subclip_stride,
        "canonicalize_subclips": canonicalize_subclips,
        "cache_full_windows": cache_full_windows,
        "cache_images": cache_images,
        "cache_teacher_tokens": cache_teacher_tokens,
        "sequences": [{"name": seq.name, "num_frames": len(seq.frames)} for seq in frame_sequences],
        "clips": [
            {
                "sequence": clip.sequence,
                "start": clip.start,
                "path": clip.path,
                "clip_length": clip.clip_length,
                "cache_dir": clip.cache_dir,
                "subclip_start": clip.subclip_start,
                "cache_type": clip.cache_type,
            }
            for clip in clips
        ],
        "window_profile": window_profiles,
    }
    (cache_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved global-window teacher cache manifest: {cache_root / MANIFEST_NAME}")
    return clips


def convert_subclip_cache_to_full_window_cache(
    *,
    source_cache_dir: str,
    output_cache_dir: str,
    subclip_lengths: list[int] | None = None,
    subclip_stride: int = 1,
    overwrite_cache: bool = False,
    log_every: int = 1,
) -> list[CachedClip]:
    subclip_lengths = subclip_lengths or [2, 3, 4, 5]
    if any(length < 2 for length in subclip_lengths):
        raise ValueError("Every virtual subclip length must be at least 2.")
    if subclip_stride < 1:
        raise ValueError("subclip_stride must be at least 1.")

    source_root = Path(source_cache_dir)
    source_manifest_path = source_root / MANIFEST_NAME
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"Source teacher cache manifest not found: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("cache_type") != "global_window_subclips":
        raise ValueError("Source cache must be a global_window_subclips cache.")
    if source_manifest.get("canonicalize_subclips"):
        raise ValueError("Cannot build model-gauge full-window cache from canonicalized subclips.")
    if not source_manifest.get("cache_teacher_tokens"):
        raise ValueError("Source cache must contain teacher camera/register tokens.")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for item in source_manifest.get("clips", []):
        path = Path(item["path"])
        if not path.exists():
            continue
        window_start = int(item.get("window_start", _parse_window_start_from_cache_path(path)))
        grouped.setdefault((str(item["sequence"]), window_start), []).append(item)
    if not grouped:
        raise ValueError(f"No existing source cache clips found in {source_cache_dir}.")

    output_root = Path(output_cache_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    clips: list[CachedClip] = []
    profiles: list[dict[str, Any]] = []
    seen_subclips: set[tuple[str, int, int]] = set()
    window_items = sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))
    print(
        f"Converting subclip cache to full-window cache: source={source_root} output={output_root} "
        f"windows={len(window_items)} virtual_subclip_lengths={subclip_lengths}"
    )

    for window_index, ((sequence_name, window_start), source_items) in enumerate(window_items):
        total_start = time.perf_counter()
        source_items = sorted(source_items, key=lambda item: int(item.get("subclip_start", int(item["start"]) - window_start)))

        sample_payload = torch.load(source_items[0]["path"], map_location="cpu", weights_only=False)
        window_length = int(sample_payload.get("window_length", max(int(item["start"]) - window_start + int(item["clip_length"]) for item in source_items)))
        image_resolution = int(sample_payload.get("image_resolution", source_manifest.get("image_resolution", 512)))
        pose_dim = int(sample_payload["teacher_pose"].shape[-1])
        token_shape = tuple(sample_payload["teacher_camera_and_register_tokens"].shape[2:])
        teacher_pose = torch.empty((1, window_length, pose_dim), dtype=torch.float32)
        teacher_tokens = torch.empty((1, window_length, *token_shape), dtype=torch.float16)
        frame_paths: list[str | None] = [None] * window_length
        filled = torch.zeros(window_length, dtype=torch.bool)

        for item in source_items:
            payload = torch.load(item["path"], map_location="cpu", weights_only=False)
            clip_length = int(payload.get("clip_length", item["clip_length"]))
            subclip_start = int(payload.get("subclip_start", int(item["start"]) - window_start))
            clip_slice = slice(subclip_start, subclip_start + clip_length)
            teacher_pose[:, clip_slice] = payload["teacher_pose"].to(torch.float32)
            teacher_tokens[:, clip_slice] = payload["teacher_camera_and_register_tokens"].to(torch.float16)
            for offset, frame_path in enumerate(payload.get("frame_paths", [])):
                if subclip_start + offset < len(frame_paths):
                    frame_paths[subclip_start + offset] = frame_path
            filled[clip_slice] = True

        covered_ranges: list[tuple[int, int]] = []
        pos = 0
        while pos < window_length:
            while pos < window_length and not bool(filled[pos]):
                pos += 1
            if pos >= window_length:
                break
            range_start = pos
            while pos < window_length and bool(filled[pos]):
                pos += 1
            covered_ranges.append((range_start, pos))

        virtual_subclips = 0
        saved_ranges = 0
        save_s = 0.0
        for range_start, range_end in covered_ranges:
            range_length = range_end - range_start
            if range_length < min(subclip_lengths):
                continue
            range_paths = frame_paths[range_start:range_end]
            missing_paths = [idx for idx, path in enumerate(range_paths, start=range_start) if not isinstance(path, str) or not path]
            if missing_paths:
                raise ValueError(
                    f"Source cache is missing frame paths sequence={sequence_name} "
                    f"window_start={window_start}; missing positions={missing_paths[:10]}"
                )
            if range_start == 0 and range_end == window_length:
                output_path = output_root / sequence_name / f"window_{window_start:06d}.pt"
            else:
                output_path = output_root / sequence_name / f"window_{window_start:06d}_range_{range_start:03d}.pt"

            range_virtual_subclips = 0
            for subclip_length in subclip_lengths:
                for local_start in range(0, range_length - subclip_length + 1, subclip_stride):
                    global_start = window_start + range_start + local_start
                    subclip_key = (sequence_name, subclip_length, global_start)
                    if subclip_key in seen_subclips:
                        continue
                    seen_subclips.add(subclip_key)
                    clips.append(
                        CachedClip(
                            sequence_name,
                            global_start,
                            str(output_path),
                            subclip_length,
                            str(output_root),
                            subclip_start=local_start,
                            cache_type="global_window_full",
                        )
                    )
                    range_virtual_subclips += 1

            if range_virtual_subclips == 0:
                continue
            save_start = time.perf_counter()
            if overwrite_cache or not output_path.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "cache_type": "global_window_full",
                    "sequence": sequence_name,
                    "start": window_start + range_start,
                    "window_start": window_start + range_start,
                    "source_window_start": window_start,
                    "source_range_start": range_start,
                    "window_length": range_length,
                    "frame_paths": [str(path) for path in range_paths],
                    "image_resolution": image_resolution,
                    "coordinate_frame": "full_vggt_teacher_window",
                    "canonicalized_to_first_frame": False,
                    "teacher_pose": teacher_pose[:, range_start:range_end].contiguous(),
                    "teacher_camera_and_register_tokens": teacher_tokens[:, range_start:range_end].contiguous(),
                }
                torch.save(payload, output_path)
            save_s += time.perf_counter() - save_start
            virtual_subclips += range_virtual_subclips
            saved_ranges += 1
        total_s = time.perf_counter() - total_start
        profiles.append(
            {
                "window_index": window_index,
                "sequence": sequence_name,
                "window_start": window_start,
                "window_length": window_length,
                "covered_ranges": [[start, end] for start, end in covered_ranges],
                "saved_ranges": saved_ranges,
                "num_source_subclips": len(source_items),
                "num_virtual_subclips": virtual_subclips,
                "save_s": save_s,
                "total_s": total_s,
            }
        )
        if window_index == 0 or (window_index + 1) % max(log_every, 1) == 0 or window_index + 1 == len(window_items):
            print(
                f"convert_window={window_index + 1:04d}/{len(window_items)} seq={sequence_name} "
                f"start={window_start} source_subclips={len(source_items)} "
                f"ranges={len(covered_ranges)} saved_ranges={saved_ranges} "
                f"virtual_subclips={virtual_subclips} save={save_s:.3f}s total={total_s:.3f}s"
            )

    manifest = {
        "version": 3,
        "cache_type": "global_window_full",
        "source_cache_dir": source_cache_dir,
        "image_resolution": source_manifest.get("image_resolution", 512),
        "teacher_window_length": source_manifest.get("teacher_window_length"),
        "teacher_window_stride": source_manifest.get("teacher_window_stride"),
        "subclip_lengths": subclip_lengths,
        "subclip_stride": subclip_stride,
        "canonicalize_subclips": False,
        "cache_full_windows": True,
        "cache_images": False,
        "cache_teacher_tokens": True,
        "sequences": source_manifest.get("sequences", []),
        "clips": [
            {
                "sequence": clip.sequence,
                "start": clip.start,
                "path": clip.path,
                "clip_length": clip.clip_length,
                "cache_dir": clip.cache_dir,
                "subclip_start": clip.subclip_start,
                "cache_type": clip.cache_type,
            }
            for clip in clips
        ],
        "conversion_profile": profiles,
    }
    (output_root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved full-window teacher cache manifest: {output_root / MANIFEST_NAME}")
    return clips


def _parse_window_start_from_cache_path(path: Path) -> int:
    for part in path.parts:
        if part.startswith("window_"):
            return int(part.removeprefix("window_"))
    raise ValueError(f"Cannot parse window_start from cache path: {path}")


def train_fixed_pose_student(
    frame_sequences: list[FrameSequence],
    *,
    teacher_checkpoint: str,
    output: str,
    student_init_checkpoint: str | None,
    profile_output: str | None,
    teacher_cache_dirs: list[str],
    image_resolution: int = 512,
    steps: int | None = None,
    epochs: int = 1,
    clip_length: int = 5,
    fixed_frames: int | None = None,
    curriculum_clip_lengths: list[int] | None = None,
    curriculum_steps_per_stage: list[int] | None = None,
    cache_stride: int = 10,
    lr: float = 1e-5,
    pose_weight: float = 1.0,
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
    fov_weight: float = 1.0,
    fixed_raw_pose_weight: float = 1.0,
    token_weight: float = 0.1,
    fixed_token_weight: float = 1.0,
    target_token_weight: float = 1.0,
    batch_size: int = 1,
    freeze_backbone: bool = True,
    seed: int = 0,
    device: str = "cuda",
    log_every: int = 10,
    wandb_enabled: bool = False,
    wandb_project: str = "vggt-fixed-pose",
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_tags: list[str] | None = None,
    wandb_mode: str = "online",
    wandb_log_every: int | None = None,
    wandb_log_samples: int = 1,
) -> None:
    curriculum_clip_lengths = curriculum_clip_lengths or []
    curriculum_steps_per_stage = curriculum_steps_per_stage or []
    if fixed_frames is not None and (fixed_frames < 1 or fixed_frames >= clip_length) and not curriculum_clip_lengths:
        raise ValueError("fixed_frames must be in [1, clip_length - 1].")
    if any(length < 2 for length in curriculum_clip_lengths):
        raise ValueError("Every curriculum clip length must be at least 2.")
    if curriculum_steps_per_stage and len(curriculum_steps_per_stage) != len(curriculum_clip_lengths):
        raise ValueError("--curriculum-steps-per-stage must have one value per curriculum clip length.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if steps is not None and steps < 1:
        raise ValueError("steps must be positive when provided.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    rng = random.Random(seed)
    torch.manual_seed(seed)

    cached_clips = _load_cached_clips(teacher_cache_dirs) if teacher_cache_dirs else []
    if teacher_cache_dirs and not cached_clips:
        raise ValueError(
            "Teacher cache directories were provided, but no cached clips could be loaded. "
            "Check that manifest clip paths are valid in the current runtime, e.g. /app paths inside docker compose."
        )
    use_cache = len(cached_clips) > 0
    if not use_cache:
        if not frame_sequences:
            raise ValueError("No frame sequences were provided.")
        online_samples = _build_clip_index(frame_sequences, clip_length, stride=1)
        if not online_samples:
            raise ValueError(f"No sequence has at least {clip_length} frames.")
        teacher = VGGTOmega().eval().to(device)
        teacher.load_state_dict(torch.load(teacher_checkpoint, map_location="cpu"))
        for param in teacher.parameters():
            param.requires_grad_(False)
    else:
        online_samples = []
        teacher = None

    student = CausalVGGTOmega().to(device)
    init_checkpoint = student_init_checkpoint or teacher_checkpoint
    state_dict = torch.load(init_checkpoint, map_location="cpu")
    missing, unexpected = student.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:8]}")
    if missing:
        print(f"Initialized student with {len(missing)} missing keys; first keys: {missing[:8]}")

    if freeze_backbone:
        # Keep the frozen VGGT-Omega backbone in inference mode while training
        # only the no-op fixed-pose adapter. This avoids train/eval-mode drift
        # from frozen dropout-like modules and matches the SLAM rollout path.
        student.eval()
        for name, param in student.named_parameters():
            param.requires_grad = name.startswith("aggregator.fixed_pose_conditioner")
        student.aggregator.fixed_pose_conditioner.train()
        trainable = sum(param.numel() for param in student.parameters() if param.requires_grad)
        print(f"Backbone frozen in eval mode; trainable fixed-pose parameters: {trainable:,}")
    else:
        student.train()

    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=lr, weight_decay=0.01)
    profiles: list[dict[str, Any]] = []
    loss_history: list[float] = []
    training_stages = _build_training_stages(
        cached_clips if use_cache else online_samples,
        use_cache=use_cache,
        frame_sequences=frame_sequences,
        clip_length=clip_length,
        fixed_frames=fixed_frames,
        curriculum_clip_lengths=curriculum_clip_lengths,
        curriculum_steps_per_stage=curriculum_steps_per_stage,
        epochs=epochs,
        batch_size=batch_size,
    )
    source_count = sum(len(stage.items) for stage in training_stages)
    planned_steps = sum(_planned_stage_steps(stage, batch_size) for stage in training_stages)
    max_steps = steps if steps is not None else planned_steps
    wandb_log_every = wandb_log_every if wandb_log_every is not None else log_every
    wandb_module = None
    wandb_run = None

    print(
        f"Training fixed-pose student: source={'cache' if use_cache else 'online'} clips={source_count} "
        f"stages={[(stage.name, stage.clip_length, len(stage.items)) for stage in training_stages]} "
        f"epochs_per_stage={epochs} max_steps={max_steps} clip_length={clip_length if not use_cache else 'mixed'} "
        f"fixed_frames={fixed_frames if fixed_frames is not None else 'auto-last'} "
        f"batch_size={batch_size} resolution={image_resolution}"
    )
    if wandb_enabled:
        wandb_config = {
            "teacher_checkpoint": teacher_checkpoint,
            "student_init_checkpoint": student_init_checkpoint,
            "teacher_cache_dirs": teacher_cache_dirs,
            "source": "cache" if use_cache else "online",
            "source_count": source_count,
            "cache_clip_lengths": _count_by_clip_length(cached_clips),
            "image_resolution": image_resolution,
            "steps": steps,
            "epochs": epochs,
            "clip_length": clip_length,
            "fixed_frames": fixed_frames,
            "curriculum_clip_lengths": curriculum_clip_lengths,
            "curriculum_steps_per_stage": curriculum_steps_per_stage,
            "training_stages": [
                {
                    "name": stage.name,
                    "clip_length": stage.clip_length,
                    "fixed_frames": stage.fixed_frames,
                    "items": len(stage.items),
                    "epochs": stage.epochs,
                    "max_steps": stage.max_steps,
                }
                for stage in training_stages
            ],
            "cache_stride": cache_stride,
            "lr": lr,
            "pose_weight": pose_weight,
            "translation_weight": translation_weight,
            "rotation_weight": rotation_weight,
            "fov_weight": fov_weight,
            "fixed_raw_pose_weight": fixed_raw_pose_weight,
            "token_weight": token_weight,
            "fixed_token_weight": fixed_token_weight,
            "target_token_weight": target_token_weight,
            "batch_size": batch_size,
            "freeze_backbone": freeze_backbone,
            "seed": seed,
            "device": device,
            "log_every": log_every,
            "wandb_log_every": wandb_log_every,
            "wandb_log_samples": wandb_log_samples,
        }
        wandb_module, wandb_run = _init_wandb(
            project=wandb_project,
            entity=wandb_entity,
            run_name=wandb_run_name,
            tags=wandb_tags,
            mode=wandb_mode,
            config=wandb_config,
        )

    global_step = 0
    for stage_index, stage in enumerate(training_stages):
        print(
            f"Starting curriculum stage {stage_index + 1}/{len(training_stages)}: "
            f"name={stage.name} clip_length={stage.clip_length} "
            f"fixed_frames={stage.fixed_frames if stage.fixed_frames is not None else 'auto-last'} "
            f"items={len(stage.items)} max_steps={stage.max_steps if stage.max_steps is not None else 'full'}"
        )
        stage_losses: list[float] = []
        stage_step = 0
        for epoch in range(stage.epochs):
            epoch_items: list[Any] = list(stage.items)
            rng.shuffle(epoch_items)
            epoch_losses: list[float] = []

            for batch_start in range(0, len(epoch_items), batch_size):
                if global_step >= max_steps or (stage.max_steps is not None and stage_step >= stage.max_steps):
                    break

                batch_items = epoch_items[batch_start : batch_start + batch_size]
                total_start = time.perf_counter()
                batch_records: list[dict[str, Any]] = []
                batch_losses: list[float] = []
                batch_target_pose_losses: list[float] = []
                batch_fixed_raw_pose_losses: list[float] = []
                batch_token_losses: list[float] = []
                batch_fixed_token_losses: list[float] = []
                batch_target_token_losses: list[float] = []
                batch_translation_losses: list[float] = []
                batch_rotation_losses: list[float] = []
                batch_fov_losses: list[float] = []
                wandb_sample_images = None
                wandb_sample_raw_pose = None
                wandb_sample_teacher_pose = None
                wandb_sample_fixed_frames = None
                wandb_sample_caption = None
                load_s = 0.0
                teacher_s = 0.0
                student_s = 0.0
                backward_s = 0.0

                optimizer.zero_grad(set_to_none=True)
                for item in batch_items:
                    item_load_start = time.perf_counter()
                    if use_cache:
                        (
                            images_cpu,
                            frame_paths,
                            teacher_pose_cpu,
                            teacher_tokens_cpu,
                            sequence_name,
                            start,
                            current_clip_length,
                        ) = _load_cached_payload(item)
                        if images_cpu is None:
                            if not frame_paths:
                                raise ValueError(f"Cached clip has no images or frame_paths: {item.path}")
                            images = load_and_preprocess_images(
                                frame_paths,
                                image_resolution=image_resolution,
                            ).to(device, non_blocking=True)
                        else:
                            images = images_cpu.to(device, non_blocking=True).float()
                        teacher_pose = teacher_pose_cpu.to(device, non_blocking=True)
                        teacher_tokens = (
                            teacher_tokens_cpu.to(device, non_blocking=True).float()
                            if teacher_tokens_cpu is not None
                            else None
                        )
                        current_fixed_frames = current_clip_length - 1 if stage.fixed_frames is None else stage.fixed_frames
                        if current_fixed_frames < 1 or current_fixed_frames >= current_clip_length:
                            raise ValueError(
                                f"fixed_frames={current_fixed_frames} is invalid for cached clip "
                                f"length={current_clip_length}: {item.path}"
                            )
                        item_teacher_s = 0.0
                    else:
                        assert teacher is not None
                        sample = item
                        sequence = frame_sequences[sample.sequence_index]
                        sequence_name = sequence.name
                        start = sample.start
                        current_clip_length = stage.clip_length
                        clip_paths = sequence.frames[start : start + current_clip_length]
                        images = load_and_preprocess_images(clip_paths, image_resolution=image_resolution).to(
                            device,
                            non_blocking=True,
                        )
                        current_fixed_frames = current_clip_length - 1 if stage.fixed_frames is None else stage.fixed_frames
                        teacher_tokens = None
                        item_teacher_s = 0.0
                    _sync_cuda(device)
                    item_load_s = time.perf_counter() - item_load_start
                    load_s += item_load_s

                    if not use_cache:
                        teacher_start = time.perf_counter()
                        with torch.no_grad():
                            teacher_predictions = teacher(images)
                            teacher_pose = teacher_predictions["pose_enc"].detach().clone()
                            teacher_tokens = teacher_predictions["camera_and_register_tokens"].detach().clone()
                        _sync_cuda(device)
                        item_teacher_s = time.perf_counter() - teacher_start
                        teacher_s += item_teacher_s

                    fixed_pose_mask = torch.zeros(teacher_pose.shape[:2], dtype=torch.bool, device=device)
                    fixed_pose_mask[:, :current_fixed_frames] = True

                    student_start = time.perf_counter()
                    predictions, _ = student.forward_incremental(
                        images,
                        None,
                        fixed_pose_enc=teacher_pose,
                        fixed_pose_mask=fixed_pose_mask,
                    )
                    raw_pose = predictions.get("pose_enc_model", predictions["pose_enc"])
                    target_pose_losses = _pose_loss(
                        raw_pose[:, current_fixed_frames : current_fixed_frames + 1],
                        teacher_pose[:, current_fixed_frames : current_fixed_frames + 1],
                        pose_weight=pose_weight,
                        translation_weight=translation_weight,
                        rotation_weight=rotation_weight,
                        fov_weight=fov_weight,
                    )
                    fixed_raw_pose_losses = _pose_loss(
                        raw_pose[:, :current_fixed_frames],
                        teacher_pose[:, :current_fixed_frames],
                        pose_weight=fixed_raw_pose_weight,
                        translation_weight=translation_weight,
                        rotation_weight=rotation_weight,
                        fov_weight=fov_weight,
                    )
                    token_losses = _token_loss(
                        predictions.get("camera_and_register_tokens"),
                        teacher_tokens,
                        fixed_frames=current_fixed_frames,
                        token_weight=token_weight,
                        fixed_token_weight=fixed_token_weight,
                        target_token_weight=target_token_weight,
                        device=device,
                    )
                    loss = target_pose_losses.total + fixed_raw_pose_losses.total + token_losses.total
                    _sync_cuda(device)
                    item_student_s = time.perf_counter() - student_start
                    student_s += item_student_s
                    if wandb_run is not None and wandb_sample_images is None and wandb_log_samples > 0:
                        wandb_sample_images = images.detach().cpu()
                        wandb_sample_raw_pose = raw_pose.detach().cpu()
                        wandb_sample_teacher_pose = teacher_pose.detach().cpu()
                        wandb_sample_fixed_frames = current_fixed_frames
                        wandb_sample_caption = f"{sequence_name} start={start} fixed={current_fixed_frames}"

                    backward_start = time.perf_counter()
                    (loss / len(batch_items)).backward()
                    _sync_cuda(device)
                    item_backward_s = time.perf_counter() - backward_start
                    backward_s += item_backward_s

                    loss_value = float(loss.detach().cpu())
                    target_pose_loss_value = float(target_pose_losses.total.detach().cpu())
                    fixed_raw_pose_loss_value = float(fixed_raw_pose_losses.total.detach().cpu())
                    token_loss_value = float(token_losses.total.detach().cpu())
                    fixed_token_loss_value = float(token_losses.fixed.detach().cpu())
                    target_token_loss_value = float(token_losses.target.detach().cpu())
                    translation_loss_value = float(
                        (target_pose_losses.translation + fixed_raw_pose_losses.translation).detach().cpu()
                    )
                    rotation_loss_value = float(
                        (target_pose_losses.rotation + fixed_raw_pose_losses.rotation).detach().cpu()
                    )
                    fov_loss_value = float((target_pose_losses.fov + fixed_raw_pose_losses.fov).detach().cpu())
                    batch_losses.append(loss_value)
                    batch_target_pose_losses.append(target_pose_loss_value)
                    batch_fixed_raw_pose_losses.append(fixed_raw_pose_loss_value)
                    batch_token_losses.append(token_loss_value)
                    batch_fixed_token_losses.append(fixed_token_loss_value)
                    batch_target_token_losses.append(target_token_loss_value)
                    batch_translation_losses.append(translation_loss_value)
                    batch_rotation_losses.append(rotation_loss_value)
                    batch_fov_losses.append(fov_loss_value)
                    batch_records.append(
                        {
                            "sequence": sequence_name,
                            "source_family": _source_family(sequence_name),
                            "start": start,
                            "clip_length": current_clip_length,
                            "fixed_frames": current_fixed_frames,
                            "load_s": item_load_s,
                            "teacher_s": item_teacher_s,
                            "student_s": item_student_s,
                            "backward_s": item_backward_s,
                            "pose_loss": loss_value,
                            "target_pose_loss": target_pose_loss_value,
                            "fixed_raw_pose_loss": fixed_raw_pose_loss_value,
                            "token_loss": token_loss_value,
                            "fixed_token_loss": fixed_token_loss_value,
                            "target_token_loss": target_token_loss_value,
                            "translation_loss": translation_loss_value,
                            "rotation_loss": rotation_loss_value,
                            "fov_loss": fov_loss_value,
                        }
                    )

                step_start = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()
                _sync_cuda(device)
                backward_s += time.perf_counter() - step_start

                loss_value = sum(batch_losses) / len(batch_losses)
                target_pose_loss_value = sum(batch_target_pose_losses) / len(batch_target_pose_losses)
                fixed_raw_pose_loss_value = sum(batch_fixed_raw_pose_losses) / len(batch_fixed_raw_pose_losses)
                token_loss_value = sum(batch_token_losses) / len(batch_token_losses)
                fixed_token_loss_value = sum(batch_fixed_token_losses) / len(batch_fixed_token_losses)
                target_token_loss_value = sum(batch_target_token_losses) / len(batch_target_token_losses)
                translation_loss_value = sum(batch_translation_losses) / len(batch_translation_losses)
                rotation_loss_value = sum(batch_rotation_losses) / len(batch_rotation_losses)
                fov_loss_value = sum(batch_fov_losses) / len(batch_fov_losses)
                epoch_losses.append(loss_value)
                loss_history.append(loss_value)
                total_s = time.perf_counter() - total_start
                record = {
                    "step": global_step,
                    "stage": stage.name,
                    "stage_index": stage_index,
                    "stage_step": stage_step,
                    "epoch": epoch,
                    "sequence": ",".join(str(item["sequence"]) for item in batch_records),
                    "start": ",".join(str(item["start"]) for item in batch_records),
                    "batch_size": len(batch_items),
                    "clip_length": batch_records[0]["clip_length"],
                    "fixed_frames": batch_records[0]["fixed_frames"],
                    "load_s": load_s,
                    "teacher_s": teacher_s,
                    "student_s": student_s,
                    "backward_s": backward_s,
                    "total_s": total_s,
                    "pose_loss": loss_value,
                    "target_pose_loss": target_pose_loss_value,
                    "fixed_raw_pose_loss": fixed_raw_pose_loss_value,
                    "token_loss": token_loss_value,
                    "fixed_token_loss": fixed_token_loss_value,
                    "target_token_loss": target_token_loss_value,
                    "translation_loss": translation_loss_value,
                    "rotation_loss": rotation_loss_value,
                    "fov_loss": fov_loss_value,
                    "samples": batch_records,
                    "source_loss": _summarize_batch_records_by_source(batch_records),
                }
                profiles.append(record)
                if wandb_run is not None:
                    wandb_metrics = _wandb_scalar_metrics(record, lr=lr)
                    wandb_run.log(wandb_metrics, step=global_step)
                    should_log_wandb_samples = (
                        global_step == 0
                        or (global_step + 1) % max(wandb_log_every, 1) == 0
                        or global_step + 1 == max_steps
                    )
                    if should_log_wandb_samples and wandb_sample_images is not None:
                        sample_payload = _wandb_intermediate_payload(
                            wandb_module,
                            images=wandb_sample_images,
                            raw_pose=wandb_sample_raw_pose,
                            teacher_pose=wandb_sample_teacher_pose,
                            fixed_frames=int(wandb_sample_fixed_frames),
                            caption=str(wandb_sample_caption),
                        )
                        if sample_payload:
                            wandb_run.log(sample_payload, step=global_step)
                should_log = (
                    global_step == 0
                    or (global_step + 1) % max(log_every, 1) == 0
                    or global_step + 1 == max_steps
                )
                if should_log:
                    window = loss_history[-max(log_every, 1) :]
                    print(
                        f"step={global_step:04d} epoch={epoch} stage={record['stage']} "
                        f"stage_step={stage_step} seq={record['sequence']} start={record['start']} "
                        f"batch={len(batch_items)} clip={record['clip_length']} fixed={record['fixed_frames']} "
                        f"loss={loss_value:.6f} loss_avg{len(window)}={sum(window) / len(window):.6f} "
                        f"target={target_pose_loss_value:.6f} fixed_raw={fixed_raw_pose_loss_value:.6f} "
                        f"token={token_loss_value:.6f} token_fixed={fixed_token_loss_value:.6f} "
                        f"token_target={target_token_loss_value:.6f} "
                        f"t={translation_loss_value:.6f} r={rotation_loss_value:.6f} fov={fov_loss_value:.6f} "
                        f"load={load_s:.3f}s teacher={teacher_s:.3f}s "
                        f"student={student_s:.3f}s backward={backward_s:.3f}s total={total_s:.3f}s"
                    )
                global_step += 1
                stage_step += 1

            if epoch_losses:
                print(
                    f"stage={stage.name} epoch={epoch} steps={len(epoch_losses)} "
                    f"loss_mean={sum(epoch_losses) / len(epoch_losses):.6f} "
                    f"loss_first={epoch_losses[0]:.6f} loss_last={epoch_losses[-1]:.6f}"
                )
            stage_losses.extend(epoch_losses)
            if global_step >= max_steps or (stage.max_steps is not None and stage_step >= stage.max_steps):
                break

        if stage_losses:
            print(
                f"stage={stage.name} total_steps={len(stage_losses)} "
                f"loss_mean={sum(stage_losses) / len(stage_losses):.6f} "
                f"loss_first={stage_losses[0]:.6f} loss_last={stage_losses[-1]:.6f}"
            )
        if global_step >= max_steps:
            break

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), output_path)
    print(f"Saved fixed-pose student checkpoint: {output_path}")

    summary = _summarize_profile(profiles)
    loss_summary = _summarize_loss(loss_history)
    print("Profile summary:")
    for key in ("load_s", "teacher_s", "student_s", "backward_s", "total_s"):
        values = summary[key]
        print(f"  {key}: mean={values['mean']:.3f}s p50={values['p50']:.3f}s max={values['max']:.3f}s")
    print(
        f"Loss summary: first={loss_summary['first']:.6f} last={loss_summary['last']:.6f} "
        f"mean={loss_summary['mean']:.6f} min={loss_summary['min']:.6f} max={loss_summary['max']:.6f}"
    )
    if wandb_run is not None:
        for key, value in loss_summary.items():
            wandb_run.summary[f"loss/{key}"] = value
        for timer_name, timer_values in summary.items():
            for stat_name, value in timer_values.items():
                wandb_run.summary[f"profile/{timer_name}_{stat_name}"] = value
        wandb_run.summary["output_checkpoint"] = str(output_path)

    if profile_output:
        profile_path = Path(profile_output)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_payload = {
            "config": {
                "teacher_checkpoint": teacher_checkpoint,
                "student_init_checkpoint": student_init_checkpoint,
                "teacher_cache_dirs": teacher_cache_dirs,
                "image_resolution": image_resolution,
                "steps": steps,
                "epochs": epochs,
                "clip_length": clip_length,
                "fixed_frames": fixed_frames,
                "curriculum_clip_lengths": curriculum_clip_lengths,
                "curriculum_steps_per_stage": curriculum_steps_per_stage,
                "training_stages": [
                    {
                        "name": stage.name,
                        "clip_length": stage.clip_length,
                        "fixed_frames": stage.fixed_frames,
                        "items": len(stage.items),
                        "epochs": stage.epochs,
                        "max_steps": stage.max_steps,
                    }
                    for stage in training_stages
                ],
                "cache_stride": cache_stride,
                "lr": lr,
                "pose_weight": pose_weight,
                "translation_weight": translation_weight,
                "rotation_weight": rotation_weight,
                "fov_weight": fov_weight,
                "fixed_raw_pose_weight": fixed_raw_pose_weight,
                "token_weight": token_weight,
                "fixed_token_weight": fixed_token_weight,
                "target_token_weight": target_token_weight,
                "batch_size": batch_size,
                "freeze_backbone": freeze_backbone,
                "seed": seed,
                "device": device,
                "wandb": {
                    "enabled": wandb_enabled,
                    "project": wandb_project,
                    "entity": wandb_entity,
                    "run_name": wandb_run_name,
                    "tags": wandb_tags or [],
                    "mode": wandb_mode,
                    "log_every": wandb_log_every,
                    "log_samples": wandb_log_samples,
                    "api_key_env": "WANDB_API_KEY",
                },
            },
            "sequences": [{"name": seq.name, "num_frames": len(seq.frames)} for seq in frame_sequences],
            "cache_clips": len(cached_clips),
            "cache_clip_lengths": _count_by_clip_length(cached_clips),
            "steps": profiles,
            "summary": summary,
            "loss_summary": loss_summary,
        }
        profile_path.write_text(json.dumps(profile_payload, indent=2), encoding="utf-8")
        print(f"Saved profile JSON: {profile_path}")
    if wandb_run is not None:
        wandb_run.finish()


def _canonicalize_extrinsics_to_first_camera(extrinsics: torch.Tensor) -> torch.Tensor:
    """Rebase camera-from-world extrinsics so the first frame is canonical.

    VGGT's native camera gauge makes the first frame the local origin. For a
    subclip cut from a longer teacher window, preserve the long-window relative
    geometry but express it in the first camera's coordinate frame:
    E_i' = E_i @ inverse(E_0).
    """
    if extrinsics.dim() != 4 or extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Expected extrinsics with shape (B, S, 3, 4), got {tuple(extrinsics.shape)}")
    batch_size, _, _, _ = extrinsics.shape
    first = extrinsics[:, :1]
    first_h = torch.eye(4, device=extrinsics.device, dtype=extrinsics.dtype).view(1, 1, 4, 4).repeat(batch_size, 1, 1, 1)
    first_h[:, :, :3, :] = first
    first_inv = torch.linalg.inv(first_h)

    extrinsics_h = torch.eye(4, device=extrinsics.device, dtype=extrinsics.dtype).view(1, 1, 4, 4).repeat(
        batch_size, extrinsics.shape[1], 1, 1
    )
    extrinsics_h[:, :, :3, :] = extrinsics
    rebased = extrinsics_h @ first_inv
    return rebased[:, :, :3, :].contiguous()


def _token_loss(
    predicted_tokens: torch.Tensor | None,
    teacher_tokens: torch.Tensor | None,
    *,
    fixed_frames: int,
    token_weight: float,
    fixed_token_weight: float = 1.0,
    target_token_weight: float = 1.0,
    device: str | torch.device,
) -> TokenLoss:
    if predicted_tokens is None or teacher_tokens is None or token_weight == 0.0:
        zero = torch.zeros((), device=device)
        return TokenLoss(total=zero, fixed=zero, target=zero)

    predicted_tokens = predicted_tokens.float()
    teacher_tokens = teacher_tokens.to(device=predicted_tokens.device, dtype=torch.float32)
    if predicted_tokens.shape != teacher_tokens.shape:
        raise ValueError(
            "Teacher camera/register tokens must match predicted tokens, got "
            f"predicted={tuple(predicted_tokens.shape)} teacher={tuple(teacher_tokens.shape)}."
        )
    fixed_loss = F.smooth_l1_loss(predicted_tokens[:, :fixed_frames], teacher_tokens[:, :fixed_frames])
    target_loss = F.smooth_l1_loss(
        predicted_tokens[:, fixed_frames : fixed_frames + 1],
        teacher_tokens[:, fixed_frames : fixed_frames + 1],
    )
    total = token_weight * (fixed_token_weight * fixed_loss + target_token_weight * target_loss)
    return TokenLoss(total=total, fixed=fixed_loss, target=target_loss)


def _pose_loss(
    predicted_pose: torch.Tensor,
    teacher_pose: torch.Tensor,
    *,
    pose_weight: float,
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
    fov_weight: float = 1.0,
) -> PoseLoss:
    predicted_pose = predicted_pose.float()
    teacher_pose = teacher_pose.float()
    translation_loss = F.smooth_l1_loss(predicted_pose[..., :3], teacher_pose[..., :3])
    rotation_loss = _quaternion_geodesic_loss(predicted_pose[..., 3:7], teacher_pose[..., 3:7])
    fov_loss = F.smooth_l1_loss(predicted_pose[..., 7:9], teacher_pose[..., 7:9])
    total = pose_weight * (
        translation_weight * translation_loss
        + rotation_weight * rotation_loss
        + fov_weight * fov_loss
    )
    return PoseLoss(total=total, translation=translation_loss, rotation=rotation_loss, fov=fov_loss)


def _quaternion_geodesic_loss(predicted_quat: torch.Tensor, teacher_quat: torch.Tensor) -> torch.Tensor:
    predicted_quat = F.normalize(predicted_quat.float(), dim=-1, eps=1e-8)
    teacher_quat = F.normalize(teacher_quat.float(), dim=-1, eps=1e-8)
    dot = torch.sum(predicted_quat * teacher_quat, dim=-1).abs().clamp(0.0, 1.0)
    # This is a stable monotonic proxy for quaternion geodesic distance that
    # preserves q/-q equivalence without the infinite acos gradient near dot=1.
    return (1.0 - dot).mean()


def _build_clip_index(frame_sequences: list[FrameSequence], clip_length: int, *, stride: int) -> list[ClipSample]:
    samples: list[ClipSample] = []
    for seq_idx, sequence in enumerate(frame_sequences):
        for start in range(0, len(sequence.frames) - clip_length + 1, stride):
            samples.append(ClipSample(seq_idx, start))
    return samples


def _max_exhaustive_teacher_window_stride(teacher_window_length: int, subclip_lengths: list[int]) -> int:
    return min(teacher_window_length - length + 1 for length in subclip_lengths)


def _build_teacher_window_starts(num_frames: int, teacher_window_length: int, stride: int) -> list[int]:
    if num_frames <= 0:
        return []
    if num_frames < teacher_window_length:
        return [0]
    last_start = num_frames - teacher_window_length
    starts = list(range(0, last_start + 1, stride))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _load_cached_clips(cache_dirs: list[str]) -> list[CachedClip]:
    clips: list[CachedClip] = []
    for cache_dir in cache_dirs:
        manifest_path = Path(cache_dir) / MANIFEST_NAME
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_clip_length = int(manifest.get("clip_length", 0))
        manifest_cache_type = str(manifest.get("cache_type", "clip"))
        for item in manifest.get("clips", []):
            path = item["path"]
            clip_length = int(item.get("clip_length", manifest_clip_length))
            if clip_length <= 0:
                raise ValueError(f"Cached clip is missing clip_length metadata: {path}")
            if Path(path).exists():
                clips.append(
                    CachedClip(
                        item["sequence"],
                        int(item["start"]),
                        path,
                        clip_length,
                        cache_dir,
                        subclip_start=int(item.get("subclip_start", 0)),
                        cache_type=str(item.get("cache_type", manifest_cache_type)),
                    )
                )
    return clips


def _load_cached_payload(
    clip: CachedClip,
) -> tuple[torch.Tensor | None, list[str], torch.Tensor, torch.Tensor | None, str, int, int]:
    payload = torch.load(clip.path, map_location="cpu", weights_only=False)
    images = payload.get("images")
    frame_paths = list(payload.get("frame_paths", []))
    teacher_pose = payload["teacher_pose"]
    teacher_tokens = payload.get("teacher_camera_and_register_tokens")
    clip_length = int(payload.get("clip_length", clip.clip_length))
    if clip.cache_type == "global_window_full" or payload.get("cache_type") == "global_window_full":
        window_length = int(payload.get("window_length", teacher_pose.shape[1]))
        clip_length = clip.clip_length
        clip_slice = slice(clip.subclip_start, clip.subclip_start + clip_length)
        if clip_slice.stop > window_length:
            raise ValueError(f"Cached virtual clip extends beyond window: {clip.path}")
        images = images[clip_slice] if images is not None else None
        frame_paths = frame_paths[clip_slice]
        teacher_pose = teacher_pose[:, clip_slice]
        teacher_tokens = teacher_tokens[:, clip_slice] if teacher_tokens is not None else None
    if images is not None and images.shape[0] != clip_length:
        raise ValueError(f"Cached image clip has unexpected length: {clip.path}")
    if len(frame_paths) != clip_length:
        raise ValueError(f"Cached frame_paths has unexpected length: {clip.path}")
    bad_frame_paths = [idx for idx, frame_path in enumerate(frame_paths) if not isinstance(frame_path, str) or not frame_path]
    if bad_frame_paths:
        raise ValueError(f"Cached frame_paths contains empty paths at positions {bad_frame_paths[:10]}: {clip.path}")
    if teacher_pose.shape[1] != clip_length:
        raise ValueError(f"Cached teacher pose has unexpected length: {clip.path}")
    if not bool(torch.isfinite(teacher_pose).all()):
        raise ValueError(f"Cached teacher pose contains non-finite values: {clip.path}")
    if teacher_tokens is not None and teacher_tokens.shape[1] != clip_length:
        raise ValueError(f"Cached teacher tokens have unexpected length: {clip.path}")
    if teacher_tokens is not None and not bool(torch.isfinite(teacher_tokens).all()):
        raise ValueError(f"Cached teacher tokens contain non-finite values: {clip.path}")
    return (
        images,
        frame_paths,
        teacher_pose,
        teacher_tokens,
        str(payload.get("sequence", clip.sequence)),
        clip.start if clip.cache_type == "global_window_full" else int(payload.get("start", clip.start)),
        clip_length,
    )


def _count_by_clip_length(clips: list[CachedClip]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for clip in clips:
        key = str(clip.clip_length)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _build_training_stages(
    items: list[Any],
    *,
    use_cache: bool,
    frame_sequences: list[FrameSequence],
    clip_length: int,
    fixed_frames: int | None,
    curriculum_clip_lengths: list[int],
    curriculum_steps_per_stage: list[int],
    epochs: int,
    batch_size: int,
) -> list[TrainingStage]:
    del batch_size
    if not curriculum_clip_lengths:
        if use_cache:
            if fixed_frames is not None:
                lengths = sorted({int(item.clip_length) for item in items})
                invalid = [length for length in lengths if fixed_frames < 1 or fixed_frames >= length]
                if invalid:
                    raise ValueError(f"fixed_frames={fixed_frames} is invalid for cached clip lengths {invalid}.")
        else:
            if fixed_frames is not None and (fixed_frames < 1 or fixed_frames >= clip_length):
                raise ValueError("fixed_frames must be in [1, clip_length - 1].")
        return [
            TrainingStage(
                name=f"L{clip_length}_fixed{fixed_frames if fixed_frames is not None else 'auto'}",
                clip_length=clip_length,
                fixed_frames=fixed_frames,
                items=items,
                epochs=epochs,
                max_steps=None,
            )
        ]

    stages: list[TrainingStage] = []
    for index, length in enumerate(curriculum_clip_lengths):
        stage_fixed_frames = fixed_frames if fixed_frames is not None else length - 1
        if stage_fixed_frames < 1 or stage_fixed_frames >= length:
            raise ValueError(f"Curriculum stage length={length} has invalid fixed_frames={stage_fixed_frames}.")
        if use_cache:
            stage_items = [item for item in items if int(item.clip_length) == length]
        else:
            stage_items = _build_clip_index(frame_sequences, length, stride=1)
        if not stage_items:
            raise ValueError(f"No training items are available for curriculum clip length {length}.")
        stages.append(
            TrainingStage(
                name=f"{stage_fixed_frames}+1_L{length}",
                clip_length=length,
                fixed_frames=stage_fixed_frames,
                items=stage_items,
                epochs=epochs,
                max_steps=curriculum_steps_per_stage[index] if curriculum_steps_per_stage else None,
            )
        )
    return stages


def _planned_stage_steps(stage: TrainingStage, batch_size: int) -> int:
    full_steps = stage.epochs * ((len(stage.items) + batch_size - 1) // batch_size)
    return min(full_steps, stage.max_steps) if stage.max_steps is not None else full_steps


def _source_family(sequence_name: str) -> str:
    if sequence_name.startswith("MH_"):
        return "eth"
    if sequence_name.startswith("rgbd_dataset_"):
        return "tum"
    if sequence_name in {"office", "apartment_images", "building_images"}:
        return "mit"
    if sequence_name.startswith("DJI_"):
        return "dji"
    return "other"


def _summarize_batch_records_by_source(batch_records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in batch_records:
        grouped.setdefault(str(record["source_family"]), []).append(record)
    summary: dict[str, dict[str, float]] = {}
    for source, records in grouped.items():
        source_summary: dict[str, float] = {"count": float(len(records))}
        for key in (
            "pose_loss",
            "target_pose_loss",
            "fixed_raw_pose_loss",
            "token_loss",
            "translation_loss",
            "rotation_loss",
            "fov_loss",
        ):
            source_summary[key] = sum(float(record[key]) for record in records) / len(records)
        summary[source] = source_summary
    return summary


def _load_frame_sequences(inputs: list[str]) -> list[FrameSequence]:
    pending: list[tuple[str, str, list[str]]] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            frames = _list_images(path)
            name = path.name
        else:
            matches = _natural_sorted_paths(glob.glob(item))
            frames = [match for match in matches if _is_image(match)]
            if glob.has_magic(item):
                name = Path(os.path.commonpath(matches)).name if matches else item
            else:
                name = path.stem
        if frames:
            pending.append((name, item, frames))

    name_counts: dict[str, int] = {}
    paths_by_name: dict[str, list[str]] = {}
    for name, item, _ in pending:
        name_counts[name] = name_counts.get(name, 0) + 1
        paths_by_name.setdefault(name, []).append(item)
    duplicate_name_maps = {
        name: _unique_sequence_names_for_items(items)
        for name, items in paths_by_name.items()
        if len(items) > 1
    }

    sequences: list[FrameSequence] = []
    used_names: dict[str, int] = {}
    for name, item, frames in pending:
        if name_counts[name] > 1:
            sequence_name = duplicate_name_maps[name][item]
        else:
            sequence_name = _sanitize_sequence_name(name)
        duplicate_count = used_names.get(sequence_name, 0)
        used_names[sequence_name] = duplicate_count + 1
        if duplicate_count:
            sequence_name = f"{sequence_name}_{duplicate_count + 1}"
        sequences.append(FrameSequence(name=sequence_name, frames=frames))
    return sequences


def _unique_sequence_names_for_items(items: list[str]) -> dict[str, str]:
    paths = [_sequence_name_path(item) for item in items]
    split_parts = [[part for part in path.parts if part not in ("", "/")] for path in paths]
    max_len = max((len(parts) for parts in split_parts), default=1)
    for suffix_len in range(2, max_len + 1):
        names = [_sanitize_sequence_name("_".join(parts[-suffix_len:])) for parts in split_parts]
        if len(set(names)) == len(names):
            return dict(zip(items, names))
    names = [_sanitize_sequence_name("_".join(parts)) for parts in split_parts]
    return dict(zip(items, names))


def _sequence_name_path(item: str) -> Path:
    path = Path(item)
    if glob.has_magic(item):
        matches = _natural_sorted_paths(glob.glob(item))
        if matches:
            path = Path(os.path.commonpath(matches))
    return path


def _sanitize_sequence_name(name: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name)
    sanitized = sanitized.strip("_")
    return sanitized or "sequence"


def _list_images(directory: Path) -> list[str]:
    return [
        str(path)
        for path in _natural_sorted_paths(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def _natural_sorted_paths(paths):
    return sorted(paths, key=_natural_path_sort_key)


def _natural_path_sort_key(path: str | Path) -> tuple[tuple[tuple[int, int | str], ...], ...]:
    return tuple(_natural_string_sort_key(part) for part in Path(path).parts)


def _natural_string_sort_key(text: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(token)) if token.isdigit() else (1, token.lower())
        for token in re.split(r"(\d+)", text)
        if token
    )


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def _sync_cuda(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _init_wandb(
    *,
    project: str,
    entity: str | None,
    run_name: str | None,
    tags: list[str] | None,
    mode: str,
    config: dict[str, Any],
) -> tuple[Any, Any]:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Install wandb or disable --wandb. The Docker image reads requirements.txt.") from exc

    normalized_mode = mode or "online"
    if normalized_mode not in {"online", "offline", "disabled"}:
        raise ValueError("--wandb-mode must be one of: online, offline, disabled.")
    if normalized_mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required when --wandb is enabled in online mode.")
    if normalized_mode == "online":
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        tags=tags or None,
        mode=normalized_mode,
        config=config,
    )
    return wandb, run


def _wandb_scalar_metrics(record: dict[str, Any], *, lr: float) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "train/loss": float(record["pose_loss"]),
        "train/target_pose_loss": float(record["target_pose_loss"]),
        "train/fixed_raw_pose_loss": float(record["fixed_raw_pose_loss"]),
        "train/token_loss": float(record["token_loss"]),
        "train/fixed_token_loss": float(record["fixed_token_loss"]),
        "train/target_token_loss": float(record["target_token_loss"]),
        "train/translation_loss": float(record["translation_loss"]),
        "train/rotation_loss": float(record["rotation_loss"]),
        "train/fov_loss": float(record["fov_loss"]),
        "profile/load_s": float(record["load_s"]),
        "profile/teacher_s": float(record["teacher_s"]),
        "profile/student_s": float(record["student_s"]),
        "profile/backward_s": float(record["backward_s"]),
        "profile/total_s": float(record["total_s"]),
        "train/epoch": int(record["epoch"]),
        "train/stage_index": int(record["stage_index"]),
        "train/stage_step": int(record["stage_step"]),
        "train/stage_clip_length": int(record["clip_length"]),
        "train/stage_fixed_frames": int(record["fixed_frames"]),
        "train/batch_size": int(record["batch_size"]),
        "train/lr": float(lr),
    }
    for source, source_values in record.get("source_loss", {}).items():
        for key, value in source_values.items():
            metrics[f"train/source/{source}_{key}"] = float(value)
    return metrics


def _wandb_intermediate_payload(
    wandb_module: Any,
    *,
    images: torch.Tensor | None,
    raw_pose: torch.Tensor | None,
    teacher_pose: torch.Tensor | None,
    fixed_frames: int,
    caption: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    contact_sheet = _clip_contact_sheet_uint8(images)
    if contact_sheet is not None:
        payload["samples/clip_contact_sheet"] = wandb_module.Image(contact_sheet, caption=caption)
    pose_table = _pose_comparison_table(wandb_module, raw_pose, teacher_pose, fixed_frames=fixed_frames)
    if pose_table is not None:
        payload["samples/pose_comparison"] = pose_table
    return payload


def _clip_contact_sheet_uint8(images: torch.Tensor | None, *, max_frames: int = 5) -> Any | None:
    if images is None:
        return None
    tensor = images.detach().cpu().float()
    if tensor.dim() == 5:
        tensor = tensor[0]
    if tensor.dim() != 4:
        return None
    if tensor.shape[1] in (1, 3):
        tensor = tensor[:max_frames]
    elif tensor.shape[-1] in (1, 3):
        tensor = tensor[:max_frames].permute(0, 3, 1, 2)
    else:
        return None
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    tensor = tensor.clamp(0.0, 1.0)
    sheet = torch.cat([frame for frame in tensor], dim=2)
    return (sheet.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")


def _pose_comparison_table(
    wandb_module: Any,
    raw_pose: torch.Tensor | None,
    teacher_pose: torch.Tensor | None,
    *,
    fixed_frames: int,
) -> Any | None:
    if raw_pose is None or teacher_pose is None:
        return None
    pred = raw_pose.detach().cpu().float()
    teacher = teacher_pose.detach().cpu().float()
    if pred.dim() == 3:
        pred = pred[0]
    if teacher.dim() == 3:
        teacher = teacher[0]
    if pred.dim() != 2 or teacher.dim() != 2 or pred.shape[-1] < 9 or teacher.shape[-1] < 9:
        return None
    rows = []
    frame_count = min(pred.shape[0], teacher.shape[0], 5)
    for frame_idx in range(frame_count):
        pred_t = pred[frame_idx, :3]
        teacher_t = teacher[frame_idx, :3]
        pred_q = F.normalize(pred[frame_idx, 3:7], dim=-1, eps=1e-8)
        teacher_q = F.normalize(teacher[frame_idx, 3:7], dim=-1, eps=1e-8)
        q_dot = float(torch.sum(pred_q * teacher_q).abs().clamp(0.0, 1.0))
        trans_l1 = float(F.smooth_l1_loss(pred_t, teacher_t))
        fov_l1 = float(F.smooth_l1_loss(pred[frame_idx, 7:9], teacher[frame_idx, 7:9]))
        rows.append(
            [
                frame_idx,
                frame_idx < fixed_frames,
                trans_l1,
                1.0 - q_dot,
                fov_l1,
                float(pred_t[0]),
                float(pred_t[1]),
                float(pred_t[2]),
                float(teacher_t[0]),
                float(teacher_t[1]),
                float(teacher_t[2]),
            ]
        )
    table = wandb_module.Table(
        columns=[
            "frame",
            "is_fixed",
            "translation_smooth_l1",
            "rotation_1_minus_abs_qdot",
            "fov_smooth_l1",
            "pred_tx",
            "pred_ty",
            "pred_tz",
            "teacher_tx",
            "teacher_ty",
            "teacher_tz",
        ],
        data=rows,
    )
    return table


def _summarize_profile(profiles: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in ("load_s", "teacher_s", "student_s", "backward_s", "total_s"):
        values = sorted(float(item[key]) for item in profiles)
        if not values:
            summary[key] = {"mean": 0.0, "p50": 0.0, "max": 0.0}
            continue
        middle = len(values) // 2
        p50 = values[middle] if len(values) % 2 else 0.5 * (values[middle - 1] + values[middle])
        summary[key] = {
            "mean": sum(values) / len(values),
            "p50": p50,
            "max": values[-1],
        }
    return summary


def _summarize_loss(losses: list[float]) -> dict[str, float]:
    if not losses:
        return {"first": 0.0, "last": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "first": losses[0],
        "last": losses[-1],
        "mean": sum(losses) / len(losses),
        "min": min(losses),
        "max": max(losses),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the fixed-pose CausalVGGTOmega path. Cache mode defaults "
            "to 5-frame clips for repeatable fixed-history training."
        )
    )
    parser.add_argument("frame_sequences", nargs="+", help="2FPS frame directories or image glob patterns.")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--student-init-checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile-output")
    parser.add_argument(
        "--teacher-cache-dir",
        action="append",
        default=[],
        help="Teacher cache directory. Repeat this option to mix multiple clip lengths during training.",
    )
    parser.add_argument("--prepare-teacher-cache", action="store_true")
    parser.add_argument("--prepare-global-window-cache", action="store_true")
    parser.add_argument("--convert-subclip-cache-to-full-window-cache", action="store_true")
    parser.add_argument("--source-teacher-cache-dir")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--clip-length", type=int, default=5)
    parser.add_argument("--fixed-frames", type=int, help="Number of fixed history frames. Defaults to clip_length - 1 for each clip.")
    parser.add_argument(
        "--curriculum-clip-lengths",
        type=int,
        nargs="+",
        default=[],
        help="Train sequential fixed-pose stages, e.g. 2 3 4 5 means 1+1, 2+1, 3+1, 4+1.",
    )
    parser.add_argument(
        "--curriculum-steps-per-stage",
        type=int,
        nargs="+",
        default=[],
        help="Optional cap for each curriculum stage; must match --curriculum-clip-lengths.",
    )
    parser.add_argument("--cache-stride", type=int, default=1)
    parser.add_argument("--teacher-window-length", type=int, default=100)
    parser.add_argument(
        "--teacher-window-stride",
        type=int,
        help="Teacher window start stride. Defaults to the largest exhaustive stride for the requested subclip lengths.",
    )
    parser.add_argument("--subclip-lengths", type=int, nargs="+", default=[5])
    parser.add_argument("--subclip-stride", type=int, default=1)
    parser.add_argument("--canonicalize-subclips", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache-full-windows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--cache-teacher-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--pose-weight", type=float, default=1.0)
    parser.add_argument("--translation-weight", type=float, default=1.0)
    parser.add_argument("--rotation-weight", type=float, default=1.0)
    parser.add_argument("--fov-weight", type=float, default=1.0)
    parser.add_argument(
        "--fixed-raw-pose-weight",
        type=float,
        default=1.0,
        help="Weight for supervising raw camera-head pose on fixed frames before pass-through replacement.",
    )
    parser.add_argument(
        "--token-weight",
        type=float,
        default=0.1,
        help="Weight for distilling final camera/register tokens when the teacher cache contains them.",
    )
    parser.add_argument("--fixed-token-weight", type=float, default=1.0)
    parser.add_argument("--target-token-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False, help="Log training metrics to wandb.ai.")
    parser.add_argument("--wandb-project", default="vggt-fixed-pose")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-log-every", type=int, help="Step interval for wandb image/table samples. Scalars log every step.")
    parser.add_argument("--wandb-log-samples", type=int, default=1, help="Number of sample clips to log per sample step; currently capped at 1.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    sequences = _load_frame_sequences(args.frame_sequences)
    cache_modes = [
        args.prepare_teacher_cache,
        args.prepare_global_window_cache,
        args.convert_subclip_cache_to_full_window_cache,
    ]
    if sum(bool(mode) for mode in cache_modes) > 1:
        raise ValueError("Choose only one cache preparation/conversion mode.")

    if args.prepare_teacher_cache:
        if not args.teacher_cache_dir:
            raise ValueError("--teacher-cache-dir is required with --prepare-teacher-cache")
        if len(args.teacher_cache_dir) != 1:
            raise ValueError("Pass exactly one --teacher-cache-dir when preparing a cache.")
        prepare_teacher_cache(
            sequences,
            teacher_checkpoint=args.teacher_checkpoint,
            cache_dir=args.teacher_cache_dir[0],
            image_resolution=args.image_resolution,
            clip_length=args.clip_length,
            stride=args.cache_stride,
            overwrite_cache=args.overwrite_cache,
            cache_teacher_tokens=args.cache_teacher_tokens,
            device=args.device,
            log_every=args.log_every,
        )
        if args.cache_only:
            return

    if args.prepare_global_window_cache:
        if not args.teacher_cache_dir:
            raise ValueError("--teacher-cache-dir is required with --prepare-global-window-cache")
        if len(args.teacher_cache_dir) != 1:
            raise ValueError("Pass exactly one --teacher-cache-dir when preparing a cache.")
        prepare_global_window_teacher_cache(
            sequences,
            teacher_checkpoint=args.teacher_checkpoint,
            cache_dir=args.teacher_cache_dir[0],
            image_resolution=args.image_resolution,
            teacher_window_length=args.teacher_window_length,
            teacher_window_stride=args.teacher_window_stride,
            subclip_lengths=args.subclip_lengths,
            subclip_stride=args.subclip_stride,
            canonicalize_subclips=args.canonicalize_subclips,
            cache_full_windows=args.cache_full_windows,
            overwrite_cache=args.overwrite_cache,
            cache_images=args.cache_images,
            cache_teacher_tokens=args.cache_teacher_tokens,
            device=args.device,
            log_every=args.log_every,
        )
        if args.cache_only:
            return

    if args.convert_subclip_cache_to_full_window_cache:
        if not args.source_teacher_cache_dir:
            raise ValueError("--source-teacher-cache-dir is required with --convert-subclip-cache-to-full-window-cache")
        if not args.teacher_cache_dir:
            raise ValueError("--teacher-cache-dir is required with --convert-subclip-cache-to-full-window-cache")
        if len(args.teacher_cache_dir) != 1:
            raise ValueError("Pass exactly one --teacher-cache-dir when converting a cache.")
        convert_subclip_cache_to_full_window_cache(
            source_cache_dir=args.source_teacher_cache_dir,
            output_cache_dir=args.teacher_cache_dir[0],
            subclip_lengths=args.subclip_lengths,
            subclip_stride=args.subclip_stride,
            overwrite_cache=args.overwrite_cache,
            log_every=args.log_every,
        )
        if args.cache_only:
            return

    train_fixed_pose_student(
        sequences,
        teacher_checkpoint=args.teacher_checkpoint,
        student_init_checkpoint=args.student_init_checkpoint,
        output=args.output,
        profile_output=args.profile_output,
        teacher_cache_dirs=args.teacher_cache_dir,
        image_resolution=args.image_resolution,
        steps=args.steps,
        epochs=args.epochs,
        clip_length=args.clip_length,
        fixed_frames=args.fixed_frames,
        curriculum_clip_lengths=args.curriculum_clip_lengths,
        curriculum_steps_per_stage=args.curriculum_steps_per_stage,
        cache_stride=args.cache_stride,
        lr=args.lr,
        pose_weight=args.pose_weight,
        translation_weight=args.translation_weight,
        rotation_weight=args.rotation_weight,
        fov_weight=args.fov_weight,
        fixed_raw_pose_weight=args.fixed_raw_pose_weight,
        token_weight=args.token_weight,
        fixed_token_weight=args.fixed_token_weight,
        target_token_weight=args.target_token_weight,
        batch_size=args.batch_size,
        freeze_backbone=args.freeze_backbone,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
        wandb_mode=args.wandb_mode,
        wandb_log_every=args.wandb_log_every,
        wandb_log_samples=args.wandb_log_samples,
    )


if __name__ == "__main__":
    main()

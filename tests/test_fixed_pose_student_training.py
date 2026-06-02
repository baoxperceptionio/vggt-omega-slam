import inspect
from pathlib import Path

import pytest
import torch

from scripts.train_fixed_pose_student import (
    _build_clip_index,
    _build_teacher_window_starts,
    _build_training_stages,
    _load_cached_clips,
    _load_cached_payload,
    _load_frame_sequences,
    _max_exhaustive_teacher_window_stride,
    _pose_loss,
    _token_loss,
    FrameSequence,
    build_arg_parser,
    convert_subclip_cache_to_full_window_cache,
    prepare_global_window_teacher_cache,
    prepare_teacher_cache,
    train_fixed_pose_student,
)


def _pose_with_quaternion(quaternion):
    pose = torch.zeros(1, 1, 9)
    pose[..., 3:7] = torch.tensor(quaternion, dtype=torch.float32)
    return pose


def test_pose_loss_is_quaternion_sign_invariant():
    teacher = _pose_with_quaternion([0.0, 0.0, 0.0, 1.0])
    predicted = _pose_with_quaternion([0.0, 0.0, 0.0, -1.0])

    losses = _pose_loss(predicted, teacher, pose_weight=1.0)

    assert losses.rotation.item() == pytest.approx(0.0, abs=1e-6)
    assert losses.total.item() == pytest.approx(0.0, abs=1e-6)


def test_pose_loss_component_weights_can_disable_terms():
    teacher = _pose_with_quaternion([0.0, 0.0, 0.0, 1.0])
    predicted = _pose_with_quaternion([1.0, 0.0, 0.0, 0.0])
    predicted[..., :3] = torch.tensor([10.0, -2.0, 1.0])
    predicted[..., 7:9] = torch.tensor([0.2, -0.4])

    zeroed = _pose_loss(
        predicted,
        teacher,
        pose_weight=1.0,
        translation_weight=0.0,
        rotation_weight=0.0,
        fov_weight=0.0,
    )
    rotation_only = _pose_loss(
        predicted,
        teacher,
        pose_weight=1.0,
        translation_weight=0.0,
        rotation_weight=1.0,
        fov_weight=0.0,
    )

    assert zeroed.translation.item() > 0.0
    assert zeroed.rotation.item() > 0.0
    assert zeroed.fov.item() > 0.0
    assert zeroed.total.item() == pytest.approx(0.0, abs=1e-6)
    assert rotation_only.total.item() == pytest.approx(rotation_only.rotation.item(), abs=1e-6)


def test_token_loss_skips_missing_teacher_tokens_and_applies_weight():
    predicted = torch.zeros(1, 5, 2, 4)
    teacher = torch.zeros(1, 5, 2, 4)
    teacher[:, :4] = 2.0
    teacher[:, 4:] = 4.0

    skipped = _token_loss(
        predicted,
        None,
        fixed_frames=4,
        token_weight=0.1,
        fixed_token_weight=1.0,
        target_token_weight=1.0,
        device="cpu",
    )
    weighted = _token_loss(
        predicted,
        teacher,
        fixed_frames=4,
        token_weight=0.5,
        fixed_token_weight=0.0,
        target_token_weight=1.0,
        device="cpu",
    )

    assert skipped.total.item() == pytest.approx(0.0, abs=1e-6)
    assert weighted.fixed.item() > 0.0
    assert weighted.target.item() > 0.0
    assert weighted.total.item() == pytest.approx(0.5 * weighted.target.item(), abs=1e-6)


def test_exhaustive_5_frame_cache_indices_cover_every_start():
    sequence = FrameSequence(name="seq", frames=[f"frame_{idx:03d}.jpg" for idx in range(205)])

    simple_samples = _build_clip_index([sequence], clip_length=5, stride=1)
    assert [sample.start for sample in simple_samples] == list(range(201))

    stride = _max_exhaustive_teacher_window_stride(teacher_window_length=100, subclip_lengths=[5])
    assert stride == 96
    window_starts = _build_teacher_window_starts(
        num_frames=len(sequence.frames),
        teacher_window_length=100,
        stride=stride,
    )

    seen_starts = []
    seen_keys = set()
    for window_start in window_starts:
        for subclip_start in range(0, 100 - 5 + 1, 1):
            global_start = window_start + subclip_start
            if global_start > len(sequence.frames) - 5:
                continue
            key = (5, global_start)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            seen_starts.append(global_start)

    assert window_starts == [0, 96, 105]
    assert seen_starts == list(range(201))



def test_short_sequence_uses_one_teacher_window_and_slides_all_5_frame_clips():
    window_starts = _build_teacher_window_starts(num_frames=7, teacher_window_length=100, stride=96)
    assert window_starts == [0]

    seen_starts = []
    current_window_length = 7
    for window_start in window_starts:
        for subclip_start in range(0, current_window_length - 5 + 1, 1):
            seen_starts.append(window_start + subclip_start)

    assert seen_starts == [0, 1, 2]


def test_duplicate_rgb_directory_names_are_disambiguated(tmp_path):
    first = tmp_path / "seq_a" / "rgb"
    second = tmp_path / "seq_b" / "rgb"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "000001.jpg").write_bytes(b"")
    (second / "000001.jpg").write_bytes(b"")

    sequences = _load_frame_sequences([str(first), str(second)])

    assert [sequence.name for sequence in sequences] == ["seq_a_rgb", "seq_b_rgb"]
    assert len({sequence.name for sequence in sequences}) == 2


def test_deep_duplicate_data_directory_names_use_unique_suffix(tmp_path):
    first = tmp_path / "machine_hall" / "MH_01_easy" / "mav0" / "cam0" / "data"
    second = tmp_path / "machine_hall" / "MH_02_easy" / "mav0" / "cam0" / "data"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "000001.png").write_bytes(b"")
    (second / "000001.png").write_bytes(b"")

    sequences = _load_frame_sequences([str(first), str(second)])

    assert [sequence.name for sequence in sequences] == [
        "MH_01_easy_mav0_cam0_data",
        "MH_02_easy_mav0_cam0_data",
    ]
    assert len({sequence.name for sequence in sequences}) == 2


def test_load_frame_sequences_uses_natural_numeric_image_order(tmp_path):
    frames = tmp_path / "mit" / "office"
    frames.mkdir(parents=True)
    for name in ("rgb_0.jpg", "rgb_1.jpg", "rgb_10.jpg", "rgb_2.jpg", "rgb_29.jpg", "rgb_290.jpg", "rgb_3.jpg"):
        (frames / name).write_bytes(b"")

    sequences = _load_frame_sequences([str(frames)])

    assert [Path(path).name for path in sequences[0].frames] == [
        "rgb_0.jpg",
        "rgb_1.jpg",
        "rgb_2.jpg",
        "rgb_3.jpg",
        "rgb_10.jpg",
        "rgb_29.jpg",
        "rgb_290.jpg",
    ]


def test_load_frame_sequences_uses_natural_numeric_order_for_globs(tmp_path):
    frames = tmp_path / "mit" / "office"
    frames.mkdir(parents=True)
    for name in ("rgb_0.jpg", "rgb_1.jpg", "rgb_10.jpg", "rgb_2.jpg"):
        (frames / name).write_bytes(b"")

    sequences = _load_frame_sequences([str(frames / "*.jpg")])

    assert [Path(path).name for path in sequences[0].frames] == [
        "rgb_0.jpg",
        "rgb_1.jpg",
        "rgb_2.jpg",
        "rgb_10.jpg",
    ]



def test_fixed_pose_training_defaults_to_5_frame_noncanonical_cache():
    parser = build_arg_parser()
    args = parser.parse_args([
        "frames",
        "--teacher-checkpoint",
        "teacher.pt",
        "--output",
        "student.pt",
    ])

    assert args.clip_length == 5
    assert args.cache_stride == 1
    assert args.teacher_window_stride is None
    assert args.subclip_lengths == [5]
    assert args.canonicalize_subclips is False
    assert args.cache_full_windows is True
    assert args.batch_size == 1
    assert args.translation_weight == 1.0
    assert args.rotation_weight == 1.0
    assert args.fov_weight == 1.0
    assert args.fixed_raw_pose_weight == 1.0
    assert args.cache_teacher_tokens is True
    assert args.token_weight == 0.1
    assert args.fixed_token_weight == 1.0
    assert args.target_token_weight == 1.0
    assert args.curriculum_clip_lengths == []
    assert args.curriculum_steps_per_stage == []
    assert args.wandb is False
    assert args.wandb_project == "vggt-fixed-pose"
    assert args.wandb_mode == "online"
    assert args.wandb_log_every is None
    assert args.wandb_log_samples == 1

    global_cache_signature = inspect.signature(prepare_global_window_teacher_cache)
    cache_signature = inspect.signature(prepare_teacher_cache)
    train_signature = inspect.signature(train_fixed_pose_student)
    assert global_cache_signature.parameters["canonicalize_subclips"].default is False
    assert global_cache_signature.parameters["cache_full_windows"].default is True
    assert global_cache_signature.parameters["subclip_lengths"].default is None
    assert global_cache_signature.parameters["teacher_window_stride"].default is None
    assert cache_signature.parameters["clip_length"].default == 5
    assert cache_signature.parameters["stride"].default == 1
    assert train_signature.parameters["clip_length"].default == 5
    assert train_signature.parameters["curriculum_clip_lengths"].default is None
    assert train_signature.parameters["wandb_enabled"].default is False


def test_curriculum_training_stages_use_last_frame_as_target():
    sequence = FrameSequence(name="seq", frames=[f"frame_{idx:03d}.jpg" for idx in range(8)])

    stages = _build_training_stages(
        [],
        use_cache=False,
        frame_sequences=[sequence],
        clip_length=5,
        fixed_frames=None,
        curriculum_clip_lengths=[2, 3, 4, 5],
        curriculum_steps_per_stage=[1, 2, 3, 4],
        epochs=1,
        batch_size=2,
    )

    assert [stage.name for stage in stages] == ["1+1_L2", "2+1_L3", "3+1_L4", "4+1_L5"]
    assert [stage.clip_length for stage in stages] == [2, 3, 4, 5]
    assert [stage.fixed_frames for stage in stages] == [1, 2, 3, 4]
    assert [stage.max_steps for stage in stages] == [1, 2, 3, 4]
    assert [len(stage.items) for stage in stages] == [7, 6, 5, 4]


def test_full_window_cache_manifest_slices_virtual_subclips(tmp_path):
    cache_dir = tmp_path / "cache"
    window_path = cache_dir / "seq" / "window_000000.pt"
    window_path.parent.mkdir(parents=True)
    torch.save(
        {
            "cache_type": "global_window_full",
            "sequence": "seq",
            "start": 0,
            "window_start": 0,
            "window_length": 6,
            "frame_paths": [f"frame_{idx}.jpg" for idx in range(6)],
            "teacher_pose": torch.arange(1 * 6 * 9, dtype=torch.float32).view(1, 6, 9),
            "teacher_camera_and_register_tokens": torch.arange(1 * 6 * 2, dtype=torch.float32).view(1, 6, 2),
        },
        window_path,
    )
    (cache_dir / "manifest.json").write_text(
        """{
  "version": 3,
  "cache_type": "global_window_full",
  "clips": [
    {
      "sequence": "seq",
      "start": 1,
      "path": "__PATH__",
      "clip_length": 3,
      "cache_dir": "__CACHE__",
      "subclip_start": 1,
      "cache_type": "global_window_full"
    },
    {
      "sequence": "seq",
      "start": 2,
      "path": "__PATH__",
      "clip_length": 5,
      "cache_dir": "__CACHE__",
      "subclip_start": 1,
      "cache_type": "global_window_full"
    }
  ]
}""".replace("__PATH__", str(window_path)).replace("__CACHE__", str(cache_dir)),
        encoding="utf-8",
    )

    clips = _load_cached_clips([str(cache_dir)])
    assert len(clips) == 2
    _, frame_paths, teacher_pose, teacher_tokens, sequence, start, clip_length = _load_cached_payload(clips[0])

    assert sequence == "seq"
    assert start == 1
    assert clip_length == 3
    assert frame_paths == ["frame_1.jpg", "frame_2.jpg", "frame_3.jpg"]
    assert teacher_pose.shape == (1, 3, 9)
    assert teacher_tokens.shape == (1, 3, 2)
    assert torch.equal(teacher_pose, torch.arange(1 * 6 * 9, dtype=torch.float32).view(1, 6, 9)[:, 1:4])


def test_convert_subclip_cache_to_full_window_cache_reuses_existing_teacher_outputs(tmp_path):
    source_cache = tmp_path / "source"
    output_cache = tmp_path / "output"
    window_dir = source_cache / "seq" / "window_000000"
    window_dir.mkdir(parents=True)
    clips = []
    pose_source = torch.arange(1 * 6 * 9, dtype=torch.float32).view(1, 6, 9)
    token_source = torch.arange(1 * 6 * 2, dtype=torch.float32).view(1, 6, 2).to(torch.float16)
    for start in (0, 1):
        path = window_dir / f"clip_L005_{start:06d}.pt"
        torch.save(
            {
                "sequence": "seq",
                "start": start,
                "window_start": 0,
                "window_length": 6,
                "subclip_start": start,
                "frame_paths": [f"frame_{idx}.jpg" for idx in range(start, start + 5)],
                "image_resolution": 512,
                "clip_length": 5,
                "coordinate_frame": "full_vggt_teacher_window",
                "canonicalized_to_first_frame": False,
                "teacher_pose": pose_source[:, start : start + 5],
                "teacher_camera_and_register_tokens": token_source[:, start : start + 5],
            },
            path,
        )
        clips.append(
            {
                "sequence": "seq",
                "start": start,
                "path": str(path),
                "clip_length": 5,
                "cache_dir": str(source_cache),
            }
        )
    (source_cache / "manifest.json").write_text(
        __import__("json").dumps(
            {
                "version": 2,
                "cache_type": "global_window_subclips",
                "image_resolution": 512,
                "teacher_window_length": 6,
                "teacher_window_stride": 1,
                "subclip_lengths": [5],
                "subclip_stride": 1,
                "canonicalize_subclips": False,
                "cache_teacher_tokens": True,
                "clips": clips,
            }
        ),
        encoding="utf-8",
    )

    converted = convert_subclip_cache_to_full_window_cache(
        source_cache_dir=str(source_cache),
        output_cache_dir=str(output_cache),
        subclip_lengths=[2, 3],
        subclip_stride=1,
    )

    assert (output_cache / "seq" / "window_000000.pt").exists()
    assert len(converted) == 9
    payload = torch.load(output_cache / "seq" / "window_000000.pt", map_location="cpu", weights_only=False)
    assert payload["cache_type"] == "global_window_full"
    assert torch.equal(payload["teacher_pose"], pose_source)
    assert torch.equal(payload["teacher_camera_and_register_tokens"], token_source)


def test_convert_subclip_cache_to_full_window_cache_handles_sparse_tail_range(tmp_path):
    source_cache = tmp_path / "source"
    output_cache = tmp_path / "output"
    window_dir = source_cache / "seq" / "window_000000"
    window_dir.mkdir(parents=True)
    path = window_dir / "clip_L002_000008.pt"
    pose_source = torch.arange(1 * 10 * 9, dtype=torch.float32).view(1, 10, 9)
    token_source = torch.arange(1 * 10 * 2, dtype=torch.float32).view(1, 10, 2).to(torch.float16)
    torch.save(
        {
            "sequence": "seq",
            "start": 8,
            "window_start": 0,
            "window_length": 10,
            "subclip_start": 8,
            "frame_paths": ["frame_8.jpg", "frame_9.jpg"],
            "image_resolution": 512,
            "clip_length": 2,
            "coordinate_frame": "full_vggt_teacher_window",
            "canonicalized_to_first_frame": False,
            "teacher_pose": pose_source[:, 8:10],
            "teacher_camera_and_register_tokens": token_source[:, 8:10],
        },
        path,
    )
    (source_cache / "manifest.json").write_text(
        __import__("json").dumps(
            {
                "version": 2,
                "cache_type": "global_window_subclips",
                "image_resolution": 512,
                "teacher_window_length": 10,
                "teacher_window_stride": 1,
                "subclip_lengths": [2],
                "subclip_stride": 1,
                "canonicalize_subclips": False,
                "cache_teacher_tokens": True,
                "clips": [
                    {
                        "sequence": "seq",
                        "start": 8,
                        "window_start": 0,
                        "path": str(path),
                        "clip_length": 2,
                        "cache_dir": str(source_cache),
                        "subclip_start": 8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    converted = convert_subclip_cache_to_full_window_cache(
        source_cache_dir=str(source_cache),
        output_cache_dir=str(output_cache),
        subclip_lengths=[2, 3],
        subclip_stride=1,
    )

    assert len(converted) == 1
    assert converted[0].start == 8
    assert converted[0].clip_length == 2
    assert converted[0].subclip_start == 0
    compact_path = output_cache / "seq" / "window_000000_range_008.pt"
    assert compact_path.exists()
    payload = torch.load(compact_path, map_location="cpu", weights_only=False)
    assert payload["source_window_start"] == 0
    assert payload["source_range_start"] == 8
    assert payload["window_length"] == 2

    _, frame_paths, teacher_pose, teacher_tokens, sequence, start, clip_length = _load_cached_payload(converted[0])
    assert sequence == "seq"
    assert start == 8
    assert clip_length == 2
    assert frame_paths == ["frame_8.jpg", "frame_9.jpg"]
    assert torch.equal(teacher_pose, pose_source[:, 8:10])
    assert torch.equal(teacher_tokens, token_source[:, 8:10])


def test_load_cached_payload_rejects_bad_frame_paths_and_teacher_lengths(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    window_path = cache_dir / "window_000000.pt"
    manifest_clip = {
        "sequence": "seq",
        "start": 0,
        "path": str(window_path),
        "clip_length": 3,
        "cache_dir": str(cache_dir),
        "subclip_start": 0,
        "cache_type": "global_window_full",
    }
    base_payload = {
        "cache_type": "global_window_full",
        "sequence": "seq",
        "start": 0,
        "window_start": 0,
        "window_length": 3,
        "frame_paths": ["frame_0.jpg", "frame_1.jpg", "frame_2.jpg"],
        "teacher_pose": torch.zeros(1, 3, 9),
        "teacher_camera_and_register_tokens": torch.zeros(1, 3, 2),
    }
    torch.save(base_payload, window_path)
    (cache_dir / "manifest.json").write_text(
        __import__("json").dumps({"version": 3, "cache_type": "global_window_full", "clips": [manifest_clip]}),
        encoding="utf-8",
    )
    clip = _load_cached_clips([str(cache_dir)])[0]

    bad_paths = dict(base_payload)
    bad_paths["frame_paths"] = ["frame_0.jpg", "", "frame_2.jpg"]
    torch.save(bad_paths, window_path)
    with pytest.raises(ValueError, match="empty paths"):
        _load_cached_payload(clip)

    bad_pose = dict(base_payload)
    bad_pose["teacher_pose"] = torch.zeros(1, 2, 9)
    torch.save(bad_pose, window_path)
    with pytest.raises(ValueError, match="teacher pose"):
        _load_cached_payload(clip)

    bad_tokens = dict(base_payload)
    bad_tokens["teacher_camera_and_register_tokens"] = torch.zeros(1, 2, 2)
    torch.save(bad_tokens, window_path)
    with pytest.raises(ValueError, match="teacher tokens"):
        _load_cached_payload(clip)

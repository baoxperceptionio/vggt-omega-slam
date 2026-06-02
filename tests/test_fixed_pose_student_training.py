import inspect

import pytest
import torch

from scripts.train_fixed_pose_student import (
    _build_clip_index,
    _build_teacher_window_starts,
    _build_training_stages,
    _load_frame_sequences,
    _max_exhaustive_teacher_window_stride,
    _pose_loss,
    _token_loss,
    FrameSequence,
    build_arg_parser,
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

import torch
import numpy as np

from vggt_omega.models import CausalVGGTOmega, VGGTOmega
from vggt_omega.utils.slam import unproject_depth_map_to_point_map_torch


def _tiny_model():
    return CausalVGGTOmega(
        embed_dim=64,
        depth=1,
        cached_layer_indices=(0,),
        register_attention_block_indices=[],
        enable_camera=False,
        enable_depth=False,
    ).eval()


def test_empty_state_first_chunk_outputs_tokens():
    model = _tiny_model()
    images = torch.rand(1, 3, 32, 32)

    with torch.inference_mode():
        predictions, state = model.forward_incremental(images)

    assert "camera_and_register_tokens" in predictions
    assert state["num_frames_seen"] == 1
    assert state["image_size_hw"] == (32, 32)


def test_multi_chunk_state_and_cache_grow():
    model = _tiny_model()
    state = model.init_slam_state()

    with torch.inference_mode():
        _, state = model.forward_incremental(torch.rand(1, 3, 32, 32), state)
        first_cache_tokens = state["layer_kv_cache"][0]["k"].shape[2]
        _, state = model.forward_incremental(torch.rand(2, 3, 32, 32), state)

    assert state["num_frames_seen"] == 3
    assert state["layer_kv_cache"][0]["k"].shape[2] == first_cache_tokens * 3



def test_incremental_chunks_only_mark_global_first_frame_as_special():
    from vggt_omega.models.aggregator import slice_expand_and_flatten_from_offset

    token = torch.tensor([[[[1.0]], [[2.0]]]])

    first_chunk = slice_expand_and_flatten_from_offset(
        token,
        batch_size=1,
        num_frames=3,
        first_global_frame_idx=0,
    )
    later_chunk = slice_expand_and_flatten_from_offset(
        token,
        batch_size=1,
        num_frames=3,
        first_global_frame_idx=3,
    )

    assert first_chunk.flatten().tolist() == [1.0, 2.0, 2.0]
    assert later_chunk.flatten().tolist() == [2.0, 2.0, 2.0]

def test_world_points_from_depth_matches_demo_convention():
    depth = torch.ones(1, 1, 2, 2, 1)
    extrinsic = torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]])
    intrinsic = torch.tensor([[[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]]])

    points = unproject_depth_map_to_point_map_torch(depth, extrinsic, intrinsic)

    expected = torch.tensor([[[[[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0]], [[-0.5, 0.5, 1.0], [0.5, 0.5, 1.0]]]]])
    assert torch.allclose(points, expected)



def test_umeyama_sim3_recovers_scale_rotation_translation():
    import math
    import numpy as np

    from scripts.run_incremental_slam import _estimate_sim3_umeyama, _transform_points_sim3

    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    angle = math.radians(30.0)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    scale = 2.5
    translation = np.array([0.25, -0.5, 1.25], dtype=np.float64)
    target = scale * (source @ rotation.T) + translation

    transform = _estimate_sim3_umeyama(source, target)
    recovered = _transform_points_sim3(transform, source)

    assert np.allclose(recovered, target, atol=1e-6)


def test_fixed_pose_conditioning_preserves_locked_camera_output():
    model = CausalVGGTOmega(
        embed_dim=64,
        depth=1,
        cached_layer_indices=(0,),
        register_attention_block_indices=[],
        enable_camera=True,
        enable_depth=False,
    ).eval()
    images = torch.rand(1, 3, 32, 32)
    fixed_pose = torch.tensor([[[0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.7, 0.8]]])
    fixed_mask = torch.tensor([[True]])

    with torch.inference_mode():
        predictions, state = model.forward_incremental(
            images,
            fixed_pose_enc=fixed_pose,
            fixed_pose_mask=fixed_mask,
        )

    assert torch.allclose(predictions["pose_enc"], fixed_pose)
    assert predictions["pose_enc_model"].shape == fixed_pose.shape
    assert predictions["fixed_pose_mask"].tolist() == [[True]]
    assert state["num_frames_seen"] == 1
    assert state["layer_kv_cache"][0]["k"].shape[2] > 0


def test_fixed_pose_conditioner_starts_as_noop_adapter():
    model = _tiny_model()
    conditioner = model.aggregator.fixed_pose_conditioner

    assert torch.count_nonzero(conditioner.pose_proj.weight).item() == 0
    assert torch.count_nonzero(conditioner.pose_proj.bias).item() == 0
    assert torch.count_nonzero(conditioner.deep_pose_proj.weight).item() == 0
    assert torch.count_nonzero(conditioner.deep_pose_proj.bias).item() == 0
    assert torch.count_nonzero(conditioner.cached_pose_proj.weight).item() == 0
    assert torch.count_nonzero(conditioner.cached_pose_proj.bias).item() == 0


def test_noop_fixed_pose_conditioning_leaves_internal_tokens_unchanged():
    torch.manual_seed(0)
    model = _tiny_model()
    images = torch.rand(2, 3, 32, 32)
    fixed_pose = torch.randn(1, 2, 9)
    fixed_mask = torch.tensor([[True, False]])

    with torch.inference_mode():
        plain_predictions, _ = model.forward_incremental(images)
        fixed_predictions, _ = model.forward_incremental(
            images,
            fixed_pose_enc=fixed_pose,
            fixed_pose_mask=fixed_mask,
        )

    assert torch.allclose(
        plain_predictions["camera_and_register_tokens"],
        fixed_predictions["camera_and_register_tokens"],
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )

def test_original_vggt_omega_still_constructs():
    assert isinstance(VGGTOmega(), VGGTOmega)


def _dummy_slam_predictions(seq_len: int, *, base: float = 0.0) -> dict[str, torch.Tensor]:
    pose = torch.zeros(1, seq_len, 9)
    pose[0, :, 0] = torch.arange(seq_len, dtype=torch.float32) + base
    pose[0, :, 6] = 1.0
    pose[0, :, 7:9] = 0.5

    extrinsic = torch.eye(4).view(1, 1, 4, 4).repeat(1, seq_len, 1, 1)[:, :, :3]
    extrinsic[0, :, 0, 3] = pose[0, :, 0]
    intrinsic = torch.eye(3).view(1, 1, 3, 3).repeat(1, seq_len, 1, 1)
    depth = torch.ones(1, seq_len, 2, 2, 1)
    depth_conf = torch.ones(1, seq_len, 2, 2)
    images = torch.zeros(1, seq_len, 3, 2, 2)
    world_points = torch.zeros(1, seq_len, 2, 2, 3)
    world_points[..., 2] = 1.0
    return {
        "pose_enc": pose,
        "depth": depth,
        "depth_conf": depth_conf,
        "images": images,
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "world_points_from_depth": world_points,
    }


def _install_fixed_pose_runner_mocks(monkeypatch, runner, dummy_model, *, num_images: int):
    monkeypatch.setattr(runner, "CausalVGGTOmega", lambda: dummy_model)
    monkeypatch.setattr(runner, "_load_or_initialize_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "load_and_preprocess_images", lambda *args, **kwargs: torch.zeros(num_images, 3, 2, 2))
    monkeypatch.setattr(
        runner,
        "_estimate_ground_normalization",
        lambda local: (
            np.array(
                [[2.0, 0.0, 0.0, 1.0], [0.0, 2.0, 0.0, 2.0], [0.0, 0.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            {
                "scale": 2.0,
                "inliers": 4,
                "plane": [0.0, 1.0, 0.0, -1.0],
                "ransac_threshold": 0.01,
                "coarse_distance": 1.0,
            },
        ),
    )

    def fake_rebase(local, transform):
        del transform
        rebased = {
            "pose_enc": local["pose_enc"].copy(),
            "extrinsic": local["extrinsic"].copy(),
            "world_points_from_depth": local["world_points_from_depth"].copy(),
        }
        rebased["pose_enc"][..., 0] += 100.0
        rebased["extrinsic"][..., 0, 3] += 100.0
        rebased["world_points_from_depth"][..., 0] += 100.0
        return rebased

    monkeypatch.setattr(runner, "_apply_global_transform", fake_rebase)
    monkeypatch.setattr(runner, "predictions_to_point_cloud", lambda *args, **kwargs: (np.zeros((1, 3)), np.zeros((1, 3))))
    monkeypatch.setattr(runner, "save_point_cloud_ply", lambda *args, **kwargs: None)


def test_fixed_pose_kv_commits_model_gauge_pose_but_exports_world_pose(monkeypatch, tmp_path):
    from scripts import run_fixed_pose_kv_slam as runner

    class DummyModel:
        def __init__(self):
            self.fixed_pose_inputs = []
            self.commit_count = 0

        def eval(self):
            return self

        def to(self, device):
            return self

        def init_slam_state(self):
            return {"layer_kv_cache": {0: {"k": torch.zeros(1, 1, 0, 1)}}}

        def forward_incremental(self, images, state=None, fixed_pose_enc=None, fixed_pose_mask=None):
            del fixed_pose_mask
            if fixed_pose_enc is not None:
                self.fixed_pose_inputs.append(fixed_pose_enc.detach().cpu().clone())
                self.commit_count += 1
                state = {"layer_kv_cache": {0: {"k": torch.zeros(1, 1, self.commit_count, 1)}}}
            else:
                state = self.init_slam_state() if state is None else state
            return _dummy_slam_predictions(int(images.shape[0]), base=float(self.commit_count)), state

    dummy = DummyModel()
    _install_fixed_pose_runner_mocks(monkeypatch, runner, dummy, num_images=1)

    runner.run_fixed_pose_kv_slam(
        ["frame_000.png"],
        checkpoint=None,
        output_dir=str(tmp_path),
        device="cpu",
        allow_random_weights=True,
    )

    assert len(dummy.fixed_pose_inputs) == 1
    assert dummy.fixed_pose_inputs[0][0, 0, 0].item() == 0.0
    saved = np.load(tmp_path / "fixed_pose_kv_slam_predictions.npz")
    assert saved["pose_enc"][0, 0] == 100.0


def test_fixed_pose_window_uses_model_gauge_history_and_exports_world_pose(monkeypatch, tmp_path):
    from scripts import run_fixed_pose_kv_slam as runner

    class DummyModel:
        def __init__(self):
            self.fixed_pose_inputs = []
            self.call_index = 0

        def eval(self):
            return self

        def to(self, device):
            return self

        def forward_incremental(self, images, state=None, fixed_pose_enc=None, fixed_pose_mask=None):
            del state, fixed_pose_mask
            seq_len = int(images.shape[0])
            if fixed_pose_enc is not None:
                self.fixed_pose_inputs.append(fixed_pose_enc.detach().cpu().clone())
            base = 0.0 if self.call_index == 0 else 1000.0 * self.call_index
            self.call_index += 1
            return _dummy_slam_predictions(seq_len, base=base), {}

    dummy = DummyModel()
    _install_fixed_pose_runner_mocks(monkeypatch, runner, dummy, num_images=6)

    runner.run_fixed_pose_window_slam(
        [f"frame_{idx:03d}.png" for idx in range(6)],
        checkpoint=None,
        output_dir=str(tmp_path),
        device="cpu",
        allow_random_weights=True,
    )

    assert len(dummy.fixed_pose_inputs) == 2
    first_fixed = dummy.fixed_pose_inputs[0][0, :4, 0].tolist()
    second_fixed = dummy.fixed_pose_inputs[1][0, :4, 0].tolist()
    assert first_fixed == [0.0, 1.0, 2.0, 3.0]
    assert second_fixed == [1.0, 2.0, 3.0, 1004.0]
    assert all(value < 1100.0 for value in second_fixed)

    saved = np.load(tmp_path / "fixed_pose_kv_slam_predictions.npz")
    assert "pose_enc_model" in saved
    assert saved["pose_enc"][0, 0] == saved["pose_enc_model"][0, 0] + 100.0
    assert saved["pose_enc"][4, 0] == saved["pose_enc_model"][4, 0] + 100.0

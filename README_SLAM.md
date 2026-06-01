# VGGT-Omega Incremental SLAM Mode

This repository now contains a first-pass offline incremental SLAM path built around `CausalVGGTOmega`. It keeps the original `VGGTOmega` teacher/baseline intact and adds a separate causal student API for ordered video/image sequences.

## Does This Version Implement KV Cache?

Yes. The current SLAM version implements an append-only KV cache for the aggregator inter-frame attention blocks.

The cache lives in the public SLAM state:

```python
state = {
    "num_frames_seen": 0,
    "layer_kv_cache": {},
    "keyframe_poses": [],
    "keyframe_points": [],
    "image_size_hw": None,
}
```

`layer_kv_cache` is keyed by aggregator layer index. Each cached entry stores:

```python
{
    "k": Tensor,              # [B, heads, history_tokens, head_dim]
    "v": Tensor,              # [B, heads, history_tokens, head_dim]
    "tokens_per_frame": int,
}
```

For a new chunk, the model computes Q/K/V only for the new frames. It concatenates the cached historical K/V with the current chunk K/V, then runs scaled dot-product attention with a frame-causal mask. Queries from the current chunk can attend to all past frames and to earlier/equal frames inside the current chunk, but not to future frames inside that chunk.

Important implementation details:

- The cache is implemented in `vggt_omega/models/aggregator.py` by `CausalInterFrameSelfAttention`, `CausalSelfAttentionBlockAdapter`, and `CausalAggregator`.
- Global inter-frame blocks cache all camera/register/patch tokens per frame.
- Register inter-frame blocks cache only camera/register tokens. Patch tokens pass through for the current chunk, matching the original register-block design.
- Cached K/V tensors are detached before storage. This is correct for inference and keeps memory bounded by stored activations rather than autograd graphs.
- `num_frames_seen` is incremented once per processed chunk.
- The first implementation supports `batch_size=1` SLAM state. Multiple independent sequences should use separate states.
- All chunks in one state must use the same preprocessed image size.

What is not cached:

- Frame blocks are not cached for historical frames because they are single-frame operations. They are only run on the new chunk.
- Heads do not cache their internal attention. Camera/depth heads run on the current chunk outputs.
- This version does not do bundle adjustment, loop closure, pose graph optimization, or map fusion beyond confidence-filtered point accumulation.

## Public API

```python
import torch
from vggt_omega.models import CausalVGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images

model = CausalVGGTOmega().to("cuda").eval()
model.load_state_dict(torch.load("checkpoints/VGGT-Omega-1B-512/model.pt", map_location="cpu"), strict=False)

state = model.init_slam_state()
images = load_and_preprocess_images(image_paths, image_resolution=512).to("cuda")

with torch.inference_mode():
    for start in range(0, images.shape[0], 4):
        chunk = images[start:start + 4]
        predictions, state = model.step(chunk, state, keyframe_stride=4)
```

Each `predictions` dictionary contains outputs for the new chunk only:

- `pose_enc`
- `depth`
- `depth_conf`
- `extrinsic`
- `intrinsic`
- `world_points_from_depth`
- `camera_and_register_tokens`
- `images` when the model is in eval mode


## Sliding-Window SLAM Mode

For quality-sensitive inference, use `--mode sliding-window` instead of the append-only causal cache path. This mode runs overlapping full VGGT-Omega windows, for example `1 2 3 4 5 -> 2 3 4 5 6 -> 3 4 5 6 7`. Each new frame is predicted in a window that contains as many already-seen frames as possible.

Each window has its own local coordinate system whose origin is that window's first frame, and its scale can also differ. The script therefore estimates a Sim(3) similarity alignment from the overlapping camera centers before saving any new pose or points:

```text
camera_center_global ~= scale * rotation * camera_center_window + translation
global_from_window = umeyama_sim3(window_overlap_centers, global_overlap_centers)
points_global = global_from_window(points_window)
```

For camera extrinsics, where VGGT uses camera-from-world matrices, the same Sim(3) is applied with the corresponding inverse relation so the saved `extrinsic`, `pose_enc`, and `world_points_from_depth` all live in the global coordinate system of the first input frame. The first window stores all frames in that window. Each later window stores only its newest frame.

Example:

```bash
python scripts/run_incremental_slam.py \
  'outputs/dji_0005_10s_2fps/frames/*.jpeg' \
  --mode sliding-window \
  --window-size 5 \
  --checkpoint checkpoints/VGGT-Omega-1B-512/model.pt \
  --output-dir outputs/dji_0005_10s_2fps/sliding_window_w5 \
  --image-resolution 512 \
  --max-points 300000 \
  --conf-percentile 20
```

Outputs are named `sliding_window_slam_points.ply` and `sliding_window_slam_predictions.npz`.

## Running Offline Incremental SLAM

Use the provided script:

```bash
python scripts/run_incremental_slam.py \
  '/home/ubuntu/resplat/users/familyroom0526/*.jpeg' \
  --checkpoint checkpoints/VGGT-Omega-1B-512/model.pt \
  --output-dir outputs/familyroom0526_incremental_slam_correct_weights \
  --chunk-size 4 \
  --keyframe-stride 4 \
  --image-resolution 512 \
  --max-points 300000 \
  --conf-percentile 20
```

Outputs:

- `incremental_slam_points.ply`: filtered point cloud for inspection.
- `incremental_slam_predictions.npz`: saved `pose_enc`, `extrinsic`, `intrinsic`, and image paths.

The latest verified run used the real 512 checkpoint and processed 24 familyroom frames with `chunk_size=4`. The output PLY had 300,000 points.

## Docker Checkpoints

The Dockerfile downloads both released checkpoints during image build using a BuildKit secret named `hf_token`:

- `/app/checkpoints/VGGT-Omega-1B-512/model.pt`
- `/app/checkpoints/VGGT-Omega-1B-256-Text-Alignment/model.pt`

Build example:

```bash
printf '%s' "$HF_TOKEN" > /tmp/vggt_omega_hf_token
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,src=/tmp/vggt_omega_hf_token \
  -t vggt-omega:local .
rm -f /tmp/vggt_omega_hf_token
```

Run inside Docker:

```bash
docker run --rm --gpus all --ipc=host \
  -v /home/ubuntu/resplat/users/familyroom0526:/data/familyroom0526:ro \
  -v /home/ubuntu/vggt-omega/outputs:/outputs \
  vggt-omega:local \
  python scripts/run_incremental_slam.py \
    '/data/familyroom0526/*.jpeg' \
    --checkpoint /app/checkpoints/VGGT-Omega-1B-512/model.pt \
    --output-dir /outputs/familyroom0526_incremental_slam_docker_real \
    --chunk-size 4 \
    --keyframe-stride 4 \
    --image-resolution 512 \
    --max-points 300000 \
    --conf-percentile 20
```

Do not hard-code a Hugging Face token into the Dockerfile. Use the secret mount so the token is available only during the build step.

## Why Fine-Tuning Is Needed

The released VGGT-Omega checkpoint was trained with full-sequence bidirectional inter-frame attention. The incremental model changes the attention pattern to causal attention plus KV cache. Loading the teacher checkpoint into `CausalVGGTOmega` is useful for initialization and smoke testing, but it is not expected to be numerically identical to full VGGT-Omega without distillation.

Fine-tuning should teach the causal student to match the full teacher when it only sees prefixes and chunks.

## Teacher Label Generation

Generate full-sequence pseudo-labels with the original `VGGTOmega` teacher:

```bash
python scripts/generate_teacher_labels.py \
  '/path/to/sequence/*.jpeg' \
  --checkpoint checkpoints/VGGT-Omega-1B-512/model.pt \
  --output outputs/teacher_labels/sequence_0001.npz \
  --image-resolution 512 \
  --device cuda
```

The label file stores:

- `pose_enc`
- `depth`
- `depth_conf`
- `camera_and_register_tokens`
- cached teacher tokens from layers `4`, `11`, `17`, and `23`
- `patch_token_start`
- `image_paths`

Use complete video clips or image sequences for teacher generation. The teacher should see the full sequence so the labels represent the original full-context VGGT-Omega behavior.

## Fine-Tuning the Causal Student

Start from the released checkpoint and train the causal model with prefix/chunk simulation:

```bash
python scripts/train_causal_student.py \
  '/path/to/sequence/*.jpeg' \
  --labels outputs/teacher_labels/sequence_0001.npz \
  --init-checkpoint checkpoints/VGGT-Omega-1B-512/model.pt \
  --output checkpoints/Causal-VGGT-Omega-512/student.pt \
  --image-resolution 512 \
  --chunk-size 4 \
  --epochs 5 \
  --lr 1e-5 \
  --device cuda
```

The current training script is intentionally minimal. It trains one sequence invocation at a time and uses these distillation terms:

- pose Smooth L1 loss against teacher `pose_enc`
- depth L1 loss
- log-depth L1 loss
- confidence-weighted depth loss using teacher `depth_conf`
- final camera/register token MSE against teacher layer 23 tokens

Recommended fine-tuning schedule:

1. Generate teacher labels for many complete sequences with the full `VGGTOmega` teacher.
2. Initialize `CausalVGGTOmega` from `VGGT-Omega-1B-512/model.pt` with `strict=False`.
3. Train with random chunk sizes, such as `1`, `2`, `4`, and `8`, so the cache path learns both single-frame and batched streaming behavior.
4. Mix short and long prefixes. Long prefixes are important because cache length changes the attention distribution.
5. Keep image resolution consistent with the checkpoint during the first pass. Use `512` for the 512 checkpoint and `256` for the text-aligned checkpoint.
6. Start with a low learning rate, around `1e-5`, then reduce to `1e-6` for stabilization.
7. Validate by comparing full teacher inference against incremental student inference on held-out sequences.

For larger training, extend `scripts/train_causal_student.py` into a dataset loader that iterates over many `.npz` label files and image lists. Keep each sequence state separate; do not reuse one `state` across unrelated videos.

## Validation Checklist

Run these checks after fine-tuning:

```bash
python -m pytest -q tests/test_incremental_slam_api.py
```

Then run incremental SLAM on held-out sequences with different chunk sizes:

```bash
for chunk in 1 2 4; do
  python scripts/run_incremental_slam.py \
    '/path/to/heldout/*.jpeg' \
    --checkpoint checkpoints/Causal-VGGT-Omega-512/student.pt \
    --output-dir outputs/heldout_chunk_${chunk} \
    --chunk-size ${chunk} \
    --image-resolution 512
 done
```

Compare against full teacher outputs:

- pose error from `pose_enc` or decoded extrinsics
- depth L1 and log-depth L1
- confidence-weighted depth error
- token MSE at layers `4`, `11`, `17`, and `23`
- point-cloud visual quality and temporal consistency

## Current Limitations

- This is an offline incremental mode, not a real-time SLAM system.
- No loop closure or global bundle adjustment is implemented.
- No keyframe culling beyond fixed `keyframe_stride` is implemented.
- The map is a lightweight point cloud assembled from chunk predictions.
- Cache memory grows linearly with processed frames and token count.
- The first state implementation is for `batch_size=1`.
- The causal model should be distilled before quality-sensitive use.

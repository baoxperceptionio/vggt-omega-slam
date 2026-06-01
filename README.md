# VGGT-Omega SLAM with Ground-Tracking

This project is an experimental SLAM-style inference pipeline built on top of
[VGGT-Omega](https://github.com/facebookresearch/vggt-omega). The original
VGGT-Omega model remains the geometry engine: it predicts camera pose, depth,
confidence, and point maps for a short ordered image sequence. This repository
wraps that model with a ground-normalized sliding-window tracker so longer
videos can be processed in one consistent, interpretable coordinate frame.

The current public runner is `scripts/run_incremental_slam.py`. Despite the
historical filename, the supported mode is now ground-tracking sliding-window
SLAM. The older append-only KV-cache mode and the plain sliding-window mode were
removed from the command-line interface to avoid maintaining several coordinate
systems at once.

## Improvements Over VGGT-Omega

VGGT-Omega is designed for strong geometry prediction inside one local image
set. This repository improves the practical long-video workflow around that
checkpoint without requiring a retrained model:

- **Longer sequence handling**: videos are processed through overlapping
  windows instead of one monolithic inference call.
- **Consistent global frame**: the first window defines a ground-normalized
  coordinate system, then later windows are aligned back into it with Sim(3).
- **Motion-aware mapping**: frames are accepted into the map only after enough
  translated motion, which reduces redundant point-cloud export from nearly
  identical views.
- **Checkpoint compatibility**: the released VGGT-Omega checkpoint can be used
  directly because the change is an inference wrapper, not a new learned
  recurrent architecture.
- **Operational outputs**: the runner exports both a filtered colored PLY point
  cloud and an NPZ package with poses, intrinsics, accepted-frame diagnostics,
  ground-plane metadata, and per-window transforms.

This is not a KV-cache streaming model. Each tracking step still runs
VGGT-Omega on a bounded overlap window, then aligns that window into the
ground-normalized global frame. Fine-tuning is therefore not required to use the
current pipeline, although domain-specific data can still help if the input
videos differ substantially from the checkpoint's training distribution.


## Improvement over VGGT-omega

This repository also includes an experimental incremental SLAM model path that moves beyond an output-side pose wrapper. The goal is to let the model understand fixed historical poses internally, so registered frames act as SLAM anchors while new frames are estimated against cached history.

The intended model interface is:

```text
history_images, history_fixed_poses, current_images
    -> pose-conditioned fixed memory / KV cache
    -> current-frame predictions
```

Key ideas:

- **Fixed pose conditioning**: known history poses are encoded and injected into camera/register tokens before the causal transformer path, so attention can treat those frames as geometric anchors.
- **Frozen history semantics**: fixed poses are immutable inputs. When a pose is marked fixed, the model preserves it in `predictions["pose_enc"]` instead of replacing it with a newly predicted value.
- **Current-frame prediction**: unknown current frames still flow through the original camera/depth heads, while fixed history frames can be masked out or passed through.
- **KV-cache reuse**: fixed history tokens are encoded once and appended to the causal inter-frame KV cache, allowing later frames to query registered memory without recomputing the entire prefix.
- **No-op initialization**: the new fixed-pose adapter starts at zero, so original VGGT-Omega weights remain useful and the incremental model initially behaves close to the original checkpoint.
- **Checkpoint compatibility**: the original encoder, transformer, and head weights are reused. Newly introduced pose-conditioning parameters can be loaded with `strict=False` and fine-tuned later.

This stage is designed to pass smoke tests before retraining: fixed-pose inputs flow into the model, locked poses are preserved in the output, and the KV cache still grows across incremental chunks. Fine-tuning is recommended for final quality because the released VGGT-Omega checkpoint was not trained to interpret fixed poses as SLAM anchors.

Minimal API example:

```python
import torch
from vggt_omega.models import CausalVGGTOmega

model = CausalVGGTOmega().eval()
state = model.init_slam_state()
images = torch.rand(1, 2, 3, 512, 512)
fixed_pose_enc = torch.zeros(1, 2, 9)
fixed_pose_mask = torch.tensor([[True, False]])

with torch.inference_mode():
    predictions, state = model.forward_incremental(
        images,
        state,
        fixed_pose_enc=fixed_pose_enc,
        fixed_pose_mask=fixed_pose_mask,
    )

assert torch.equal(predictions["fixed_pose_mask"], fixed_pose_mask)
assert torch.allclose(predictions["pose_enc"][:, 0], fixed_pose_enc[:, 0])
```

`fixed_pose_mask=True` means the corresponding input pose is a locked SLAM anchor. The raw camera-head output is still available as `predictions["pose_enc_model"]` for debugging or training losses.

## What Changed From VGGT-Omega

VGGT-Omega is normally a batch model: it receives a set or short sequence of
images and predicts geometry in that sequence's local coordinate system. Each
new window can have a different origin, orientation, and scale. This project
does not retrain the released checkpoint. Instead, it changes the inference
system around VGGT-Omega:

- run VGGT-Omega repeatedly on overlapping windows;
- estimate a ground plane from the first window;
- rotate the first window so the estimated ground normal becomes global up;
- scale the scene so the first camera-to-ground distance is `1`;
- align later windows into that global frame with Sim(3);
- add only sufficiently translated frames to the map and to the future window
  anchor set.

```mermaid
flowchart LR
    A["Input images / video frames"] --> B["VGGT-Omega window inference"]
    B --> C["Pose, depth, confidence, point map"]
    C --> D["Ground-plane initialization"]
    D --> E["Ground-normalized global frame"]
    E --> F["Sliding-window tracking"]
    F --> G["Sim(3) alignment from overlap cameras"]
    G --> H{"Motion above threshold?"}
    H -- "yes" --> I["Accept frame into map and future anchors"]
    H -- "no" --> J["Save pose only"]
    I --> K["PLY point cloud + NPZ predictions"]
    J --> K
```

## Model Adaptation

The model itself is still `VGGTOmega`. The adaptation is a tracking wrapper,
not a learned recurrent model:

```mermaid
flowchart TB
    subgraph Original["Original VGGT-Omega"]
        O1["Images in one local window"] --> O2["Full-attention VGGT-Omega"]
        O2 --> O3["Local camera extrinsics"]
        O2 --> O4["Depth and confidence"]
    end

    subgraph Wrapper["Ground-tracking wrapper in this repository"]
        W1["Choose initial window"] --> W2["Estimate ground transform"]
        W2 --> W3["Store global frame"]
        W3 --> W4["For each candidate frame: run overlap window"]
        W4 --> W5["Estimate global_from_window with Umeyama Sim(3)"]
        W5 --> W6["Rebase extrinsics and points"]
        W6 --> W7["Motion gate"]
    end

    O3 --> W2
    O4 --> W2
    O3 --> W5
    O4 --> W6
```

Important consequences:

- The released VGGT-Omega checkpoint can be used directly.
- There is no learned temporal state, bundle adjustment, loop closure, or pose
  graph optimization.
- The output coordinate system is defined by the first window's estimated
  ground plane, not by GPS, IMU, COLMAP, or a metric calibration target.

## How Ground Is Estimated

The first VGGT-Omega window initializes the global coordinate frame. The code
uses the first frame's predicted point map and confidence map:

1. Select candidate ground points:
   - keep finite points only;
   - prefer points in the lower part of the image;
   - prefer points with confidence at or above the median confidence;
   - prefer points whose model-space vertical coordinate suggests they are
     below the camera.
2. Fit a coarse plane with SVD over those candidates.
3. Compute the coarse camera-to-plane distance `d`.
4. Run deterministic RANSAC with threshold `max(d / 10, 1e-4)`.
5. Refit the plane with SVD on the RANSAC inliers.
6. Orient the normal so it points upward relative to the candidate centroid.
7. Rotate that normal onto the global up axis `[0, 1, 0]`.
8. Scale the scene by `1 / d`, so the first camera is one normalized unit above
   the estimated ground plane.

The initial ground transform is therefore:

```text
global_point = scale * R_ground * local_point
scale = 1 / camera_to_ground_distance
R_ground * ground_normal = [0, 1, 0]
```

This gives the tracker a practical normalized unit. With the default motion
threshold of `0.1`, a frame is accepted after moving roughly one tenth of the
first camera height above the estimated ground.

## Window Alignment

After initialization, each tracking step builds a window from the latest
accepted frames plus one candidate frame:

```text
accepted anchors: 0 1 2 3
candidate:        4
window:           0 1 2 3 4
```

If frame `4` is accepted, it becomes a future anchor. If it is rejected, it
still receives a global pose, but it is not added to the map and it is not used
as a future alignment anchor.

Every new window has its own VGGT-Omega local coordinates. The overlap cameras
are used to estimate a similarity transform:

```text
p_global ~= s * R * p_window + t
global_from_window = [sR, t]
```

The implementation uses Umeyama alignment on overlapping camera centers. The
same Sim(3) is then applied to:

- predicted point maps;
- camera-from-world extrinsics;
- regenerated `pose_enc` values.

This keeps `extrinsic`, `pose_enc`, and `world_points_from_depth` in the same
ground-normalized global frame.

## Runtime Flow

```mermaid
sequenceDiagram
    participant User
    participant Runner as run_incremental_slam.py
    participant VGGT as VGGT-Omega
    participant Tracker as Ground tracker
    participant Output as Output files

    User->>Runner: image glob, checkpoint, window size, threshold
    Runner->>VGGT: first image window
    VGGT-->>Runner: pose, depth, confidence, points
    Runner->>Tracker: estimate ground transform
    Tracker-->>Runner: normalized global frame
    loop each later frame
        Runner->>VGGT: accepted overlap frames + candidate
        VGGT-->>Runner: local window geometry
        Runner->>Tracker: Sim(3) align and motion-gate candidate
    end
    Runner->>Output: ground_tracking_slam_points.ply
    Runner->>Output: ground_tracking_slam_predictions.npz
```

## Running

Example:

```bash
python scripts/run_incremental_slam.py \
  'outputs/dji_0005_10s_2fps/frames/*.jpeg' \
  --checkpoint checkpoints/VGGT-Omega-1B-512/model.pt \
  --output-dir outputs/dji_0005_10s_2fps/ground_tracking_w5_thresh0p1 \
  --window-size 5 \
  --displacement-threshold 0.1 \
  --image-resolution 512 \
  --max-points 300000 \
  --conf-percentile 20
```

Common options:

| Option | Meaning |
| :--- | :--- |
| `--checkpoint` | Path to the released VGGT-Omega checkpoint. |
| `--window-size` | Number of frames per VGGT-Omega tracking window. The candidate frame is appended after the latest accepted anchors. |
| `--displacement-threshold` | Minimum normalized translation required before a candidate is accepted into the map and future anchor set. |
| `--image-resolution` | Preprocessing resolution passed to VGGT-Omega. Use `512` for the `VGGT-Omega-1B-512` checkpoint. |
| `--conf-percentile` | Confidence percentile used when exporting the point cloud. |
| `--max-points` | Maximum number of exported PLY points after filtering. |
| `--device` | `cuda` or `cpu`. CUDA is expected for practical runs. |

Outputs:

- `ground_tracking_slam_points.ply`: filtered colored point cloud containing
  accepted map frames.
- `ground_tracking_slam_predictions.npz`: poses and diagnostics for all input
  frames.

The `.npz` file contains:

| Key | Description |
| :--- | :--- |
| `pose_enc` | Global pose encoding for every input frame. |
| `extrinsic` | Ground-normalized camera-from-world extrinsics for every frame. |
| `intrinsic` | Predicted intrinsics for every frame. |
| `image_paths` | Input image paths after glob expansion. |
| `accepted_mask` | Boolean mask showing which frames entered the map and future anchor set. |
| `accepted_indices` | Integer indices of accepted frames. |
| `displacements` | Candidate displacement from the latest accepted frame. |
| `ground_transform` | Initial Sim(3)-style ground normalization transform. |
| `ground_plane` | Estimated first-frame ground plane. |
| `ground_inliers` | Number of RANSAC inliers for the ground plane. |
| `ground_ransac_threshold` | RANSAC plane threshold derived from coarse distance. |
| `ground_coarse_distance` | Coarse camera-to-plane distance before RANSAC refinement. |
| `global_from_window` | Per-window transforms into the global frame. |

## Docker Compose

The Dockerfile now builds the dependency/runtime image only. Repository code is not baked into the image; `docker-compose.yml` bind-mounts the host checkout into `/app` so code changes are picked up dynamically. The compose service also mounts the host `/tmp` into the container `/tmp`.

By default, compose uses checkpoints from the mounted host checkout:

- `/app/checkpoints/VGGT-Omega-1B-512/model.pt`
- `/app/checkpoints/VGGT-Omega-1B-256-Text-Alignment/model.pt`

The Dockerfile can optionally bake checkpoints into `/opt/vggt-omega/checkpoints` if an `hf_token` BuildKit secret is provided, but `docker compose build vggt-omega` no longer requires that token file.

Build and run the demo with compose:

```bash
docker compose build vggt-omega
docker compose up vggt-omega
```

Run the SLAM script through the same service:

```bash
docker compose run --rm vggt-omega \
  python scripts/run_incremental_slam.py \
    '/tmp/frames/*.jpeg' \
    --checkpoint /app/checkpoints/VGGT-Omega-1B-512/model.pt \
    --output-dir /tmp/vggt_omega_slam \
    --window-size 5 \
    --displacement-threshold 0.1 \
    --image-resolution 512
```

Set `VGGT_OMEGA_PORT` to change the exposed Gradio port. Do not hard-code a Hugging Face token into the Dockerfile; use a BuildKit secret only when intentionally baking checkpoints into the image.

## Current Limitations

- Ground estimation is heuristic. It assumes the first view contains enough
  visible ground below the camera.
- The global up direction and normalized scale come from the first fitted plane.
  A bad first plane will affect the whole track.
- Sim(3) is estimated only from overlapping camera centers. Nearly pure rotation
  or tiny baselines can still be unstable.
- Rejected frames still have pose output, but their points are not added to the
  exported map.
- There is no loop closure, bundle adjustment, dense fusion, or global pose
  graph optimization.
- The tracker uses the released full-attention VGGT-Omega checkpoint directly.
  It is not a trained recurrent, causal, or metric SLAM system.

## Verification

Run the focused SLAM API tests:

```bash
python -m pytest -q tests/test_incremental_slam_api.py
```

The tests include point-map convention checks and Sim(3) recovery checks for
scale, rotation, and translation.

#!/usr/bin/env bash
set -euo pipefail

CONFIG_JSON="${CONFIG_JSON:-/app/configs/fixed_pose_deep_token_public_sources.json}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-/app/checkpoints/VGGT-Omega-1B-512/model.pt}"
STUDENT_INIT_CHECKPOINT="${STUDENT_INIT_CHECKPOINT:-/tmp/vggt_fixed_pose_ft/checkpoints/fixed_pose_student_deep_token_epoch1.pt}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-/tmp/vggt_fixed_pose_ft/checkpoints/fixed_pose_student_deep_token_public_epoch1.pt}"
PROFILE_OUTPUT="${PROFILE_OUTPUT:-/tmp/vggt_fixed_pose_ft/profile_deep_token_public_epoch1.json}"

DJI_CACHE="${DJI_CACHE:-/app/outputs/teacher_cache/cache_global_w100_sub5_tokens}"
PUBLIC_CACHE="${PUBLIC_CACHE:-/app/outputs/teacher_cache/cache_public_rgb_w100_sub5_tokens}"

IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
TEACHER_WINDOW_LENGTH="${TEACHER_WINDOW_LENGTH:-100}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-3}"
LR="${LR:-3e-6}"
TOKEN_WEIGHT="${TOKEN_WEIGHT:-0.1}"
LOG_EVERY="${LOG_EVERY:-25}"
DEVICE="${DEVICE:-cuda}"

DJI_FRAME_DIRS=(
  "/app/outputs/teacher_cache/frames_exhaustive5/DJI_0005"
  "/app/outputs/teacher_cache/frames_exhaustive5/DJI_0010"
)

PUBLIC_RGB_DIRS=(
  "/app/public_data/tum/rgbd_dataset_freiburg2_pioneer_360/rgb"
  "/app/public_data/tum/rgbd_dataset_freiburg2_pioneer_slam/rgb"
  "/app/public_data/tum/rgbd_dataset_freiburg2_pioneer_slam2/rgb"
  "/app/public_data/tum/rgbd_dataset_freiburg2_pioneer_slam3/rgb"
)

echo "Using config record: ${CONFIG_JSON}"
echo "Preparing public RGB teacher-token cache at ${PUBLIC_CACHE}"

python scripts/train_fixed_pose_student.py \
  "${PUBLIC_RGB_DIRS[@]}" \
  --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
  --output /tmp/vggt_fixed_pose_ft/checkpoints/public_cache_prepare_dummy.pt \
  --teacher-cache-dir "${PUBLIC_CACHE}" \
  --prepare-global-window-cache \
  --cache-only \
  --image-resolution "${IMAGE_RESOLUTION}" \
  --teacher-window-length "${TEACHER_WINDOW_LENGTH}" \
  --subclip-lengths 5 \
  --subclip-stride 1 \
  --cache-teacher-tokens \
  --device "${DEVICE}" \
  --log-every 1

echo "Continuing training from ${STUDENT_INIT_CHECKPOINT}"

python scripts/train_fixed_pose_student.py \
  "${DJI_FRAME_DIRS[@]}" \
  "${PUBLIC_RGB_DIRS[@]}" \
  --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
  --student-init-checkpoint "${STUDENT_INIT_CHECKPOINT}" \
  --output "${OUTPUT_CHECKPOINT}" \
  --profile-output "${PROFILE_OUTPUT}" \
  --teacher-cache-dir "${DJI_CACHE}" \
  --teacher-cache-dir "${PUBLIC_CACHE}" \
  --image-resolution "${IMAGE_RESOLUTION}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --token-weight "${TOKEN_WEIGHT}" \
  --fixed-token-weight 1.0 \
  --target-token-weight 1.0 \
  --fixed-raw-pose-weight 1.0 \
  --freeze-backbone \
  --device "${DEVICE}" \
  --log-every "${LOG_EVERY}"

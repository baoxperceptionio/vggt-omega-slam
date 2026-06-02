#!/usr/bin/env bash
set -euo pipefail

CONFIG_JSON="${CONFIG_JSON:-/app/configs/fixed_pose_deep_token_public_sources.json}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-/app/checkpoints/VGGT-Omega-1B-512/model.pt}"
STUDENT_INIT_CHECKPOINT="${STUDENT_INIT_CHECKPOINT:-}"
OUTPUT_CHECKPOINT="${OUTPUT_CHECKPOINT:-/tmp/vggt_fixed_pose_ft/checkpoints/fixed_pose_student_tum_eth_mit_curriculum_epoch1.pt}"
PROFILE_OUTPUT="${PROFILE_OUTPUT:-/tmp/vggt_fixed_pose_ft/profile_tum_eth_mit_curriculum_epoch1.json}"

TUM_CACHE="${TUM_CACHE:-/app/outputs/teacher_cache/cache_public_rgb_w100_sub2_3_4_5_tokens}"
ETH_CACHE="${ETH_CACHE:-/app/outputs/teacher_cache/cache_eth_cam0_w100_sub2_3_4_5_tokens}"
MIT_CACHE="${MIT_CACHE:-/app/outputs/teacher_cache/cache_mit_jpg_w100_sub2_3_4_5_tokens}"

IMAGE_RESOLUTION="${IMAGE_RESOLUTION:-512}"
TEACHER_WINDOW_LENGTH="${TEACHER_WINDOW_LENGTH:-100}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-3}"
LR="${LR:-8e-6}"
TOKEN_WEIGHT="${TOKEN_WEIGHT:-0.1}"
LOG_EVERY="${LOG_EVERY:-25}"
DEVICE="${DEVICE:-cuda}"
SUBCLIP_LENGTHS=(${SUBCLIP_LENGTHS:-2 3 4 5})
CURRICULUM_CLIP_LENGTHS=(${CURRICULUM_CLIP_LENGTHS:-2 3 4 5})
CURRICULUM_STEPS_PER_STAGE=(${CURRICULUM_STEPS_PER_STAGE:-})
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-vggt-fixed-pose}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-fixed-pose-public-only-curriculum-lr8e-6-epoch1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-25}"
WANDB_LOG_SAMPLES="${WANDB_LOG_SAMPLES:-1}"

TUM_RGBD_DIRS=(
  "/app/public_data/tum/rgbd_dataset_freiburg2_pioneer_360/rgb"
  "/app/public_data/tum/rgbd_dataset_freiburg2_pioneer_slam/rgb"
  "/app/public_data/tum/rgbd_dataset_freiburg2_pioneer_slam2/rgb"
  "/app/public_data/tum/rgbd_dataset_freiburg2_pioneer_slam3/rgb"
)

ETH_CAM0_DIRS=(
  "/app/public_data/eth/machine_hall/MH_01_easy/mav0/cam0/data"
  "/app/public_data/eth/machine_hall/MH_02_easy/mav0/cam0/data"
  "/app/public_data/eth/machine_hall/MH_03_medium/mav0/cam0/data"
  "/app/public_data/eth/machine_hall/MH_04_difficult/mav0/cam0/data"
  "/app/public_data/eth/machine_hall/MH_05_difficult/mav0/cam0/data"
)

MIT_JPG_DIRS=(
  "/app/public_data/mit/office"
  "/app/public_data/mit/apartment/images"
  "/app/public_data/mit/building/images"
)

echo "Using config record: ${CONFIG_JSON}"
echo "Curriculum subclip lengths: ${SUBCLIP_LENGTHS[*]}"
echo "Training curriculum clip lengths: ${CURRICULUM_CLIP_LENGTHS[*]}"
echo "Preparing TUM RGB-D teacher-token cache at ${TUM_CACHE}"
python scripts/train_fixed_pose_student.py \
  "${TUM_RGBD_DIRS[@]}" \
  --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
  --output /tmp/vggt_fixed_pose_ft/checkpoints/tum_cache_prepare_dummy.pt \
  --teacher-cache-dir "${TUM_CACHE}" \
  --prepare-global-window-cache \
  --cache-only \
  --image-resolution "${IMAGE_RESOLUTION}" \
  --teacher-window-length "${TEACHER_WINDOW_LENGTH}" \
  --subclip-lengths "${SUBCLIP_LENGTHS[@]}" \
  --subclip-stride 1 \
  --cache-teacher-tokens \
  --device "${DEVICE}" \
  --log-every 1

echo "Preparing ETH cam0 teacher-token cache at ${ETH_CACHE}"
python scripts/train_fixed_pose_student.py \
  "${ETH_CAM0_DIRS[@]}" \
  --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
  --output /tmp/vggt_fixed_pose_ft/checkpoints/eth_cache_prepare_dummy.pt \
  --teacher-cache-dir "${ETH_CACHE}" \
  --prepare-global-window-cache \
  --cache-only \
  --image-resolution "${IMAGE_RESOLUTION}" \
  --teacher-window-length "${TEACHER_WINDOW_LENGTH}" \
  --subclip-lengths "${SUBCLIP_LENGTHS[@]}" \
  --subclip-stride 1 \
  --cache-teacher-tokens \
  --device "${DEVICE}" \
  --log-every 1

echo "Preparing MIT JPEG teacher-token cache at ${MIT_CACHE}"
python scripts/train_fixed_pose_student.py \
  "${MIT_JPG_DIRS[@]}" \
  --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
  --output /tmp/vggt_fixed_pose_ft/checkpoints/mit_cache_prepare_dummy.pt \
  --teacher-cache-dir "${MIT_CACHE}" \
  --prepare-global-window-cache \
  --cache-only \
  --image-resolution "${IMAGE_RESOLUTION}" \
  --teacher-window-length "${TEACHER_WINDOW_LENGTH}" \
  --subclip-lengths "${SUBCLIP_LENGTHS[@]}" \
  --subclip-stride 1 \
  --cache-teacher-tokens \
  --device "${DEVICE}" \
  --log-every 1

train_args=(
  "${TUM_RGBD_DIRS[@]}"
  "${ETH_CAM0_DIRS[@]}"
  "${MIT_JPG_DIRS[@]}"
  --teacher-checkpoint "${TEACHER_CHECKPOINT}"
  --output "${OUTPUT_CHECKPOINT}"
  --profile-output "${PROFILE_OUTPUT}"
  --teacher-cache-dir "${TUM_CACHE}"
  --teacher-cache-dir "${ETH_CACHE}"
  --teacher-cache-dir "${MIT_CACHE}"
  --image-resolution "${IMAGE_RESOLUTION}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --curriculum-clip-lengths "${CURRICULUM_CLIP_LENGTHS[@]}"
  --token-weight "${TOKEN_WEIGHT}"
  --fixed-token-weight 1.0
  --target-token-weight 1.0
  --fixed-raw-pose-weight 1.0
  --freeze-backbone
  --device "${DEVICE}"
  --log-every "${LOG_EVERY}"
)

if [[ "${#CURRICULUM_STEPS_PER_STAGE[@]}" -gt 0 ]]; then
  train_args+=(--curriculum-steps-per-stage "${CURRICULUM_STEPS_PER_STAGE[@]}")
fi

if [[ -n "${STUDENT_INIT_CHECKPOINT}" ]]; then
  echo "Training public-only from student init checkpoint: ${STUDENT_INIT_CHECKPOINT}"
  train_args+=(--student-init-checkpoint "${STUDENT_INIT_CHECKPOINT}")
else
  echo "Training public-only from original VGGT-Omega checkpoint; no DJI-trained student init is used."
fi

if [[ "${WANDB}" != "0" ]]; then
  echo "Logging training to wandb project=${WANDB_PROJECT} run=${WANDB_RUN_NAME} mode=${WANDB_MODE}"
  train_args+=(
    --wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-run-name "${WANDB_RUN_NAME}"
    --wandb-mode "${WANDB_MODE}"
    --wandb-log-every "${WANDB_LOG_EVERY}"
    --wandb-log-samples "${WANDB_LOG_SAMPLES}"
    --wandb-tags fixed-pose public-only tum eth mit curriculum lr8e-6
  )
fi

python scripts/train_fixed_pose_student.py "${train_args[@]}"

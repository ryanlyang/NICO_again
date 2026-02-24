#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOB_SCRIPT="$SCRIPT_DIR/run_nico_guided_optuna_sgd_sweep.sh"

REPO_ROOT=${REPO_ROOT:-/home/ryreu/guided_cnn/NICO_again/NICO_again}
TXTLIST_DIR=${TXTLIST_DIR:-/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist}
MASK_ROOT=${MASK_ROOT:-/home/ryreu/guided_cnn/code/HaveNicoLearn/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}
IMAGE_ROOT=${IMAGE_ROOT:-/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ryreu/guided_cnn/NICO_runs/output}

mkdir -p "$SCRIPT_DIR/logsNICO"

TIMEOUT_SECONDS=${TIMEOUT_SECONDS-}
N_TRIALS=${N_TRIALS:-20}
FRESH_START=${FRESH_START:-1}

for TARGET in "autumn rock" "dim grass" "outdoor water"; do
  TARGET_TAG=$(echo "$TARGET" | tr ' ' '-')
  case "$TARGET_TAG" in
    autumn-rock)
      LOCK_BASE_LR=0.0006986960740598744
      LOCK_CLASSIFIER_LR=0.0012794989281937562
      LOCK_DROPOUT=0.0
      ;;
    dim-grass)
      LOCK_BASE_LR=0.00021570008495483682
      LOCK_CLASSIFIER_LR=0.0007490773637911799
      LOCK_DROPOUT=0.5
      ;;
    outdoor-water)
      LOCK_BASE_LR=0.000250957524758074
      LOCK_CLASSIFIER_LR=0.002118046217184987
      LOCK_DROPOUT=0.5
      ;;
    *)
      echo "Unknown target tag: $TARGET_TAG" >&2
      exit 2
      ;;
  esac

  REPO_ROOT="$REPO_ROOT" \
  TXTLIST_DIR="$TXTLIST_DIR" \
  MASK_ROOT="$MASK_ROOT" \
  IMAGE_ROOT="$IMAGE_ROOT" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  TARGET_DOMAINS="$TARGET" \
  TARGET_TAG="$TARGET_TAG" \
  TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
  N_TRIALS="$N_TRIALS" \
  FRESH_START="$FRESH_START" \
  USE_CLI_RANGES=1 \
  BASE_LR_LOW="$LOCK_BASE_LR" \
  BASE_LR_HIGH="$LOCK_BASE_LR" \
  CLASSIFIER_LR_LOW="$LOCK_CLASSIFIER_LR" \
  CLASSIFIER_LR_HIGH="$LOCK_CLASSIFIER_LR" \
  RESNET_DROPOUT="$LOCK_DROPOUT" \
  STUDY_NAME="nico_guided_sgd_locked_vanilla_${TARGET_TAG}" \
  OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///$OUTPUT_DIR/optuna_guided_sgd_locked_vanilla_v1_${TARGET_TAG}.db}" \
  sbatch "$JOB_SCRIPT"
done

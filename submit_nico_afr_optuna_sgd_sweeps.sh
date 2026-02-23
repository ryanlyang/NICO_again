#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOB_SCRIPT="$SCRIPT_DIR/run_nico_afr_optuna_sgd_sweep.sh"

REPO_ROOT=${REPO_ROOT:-/home/ryreu/guided_cnn/NICO_again/NICO_again}
TXTLIST_DIR=${TXTLIST_DIR:-/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist}
IMAGE_ROOT=${IMAGE_ROOT:-/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ryreu/guided_cnn/NICO_runs/output}

mkdir -p "$SCRIPT_DIR/logsNICO"

TIMEOUT_SECONDS=${TIMEOUT_SECONDS-}
N_TRIALS=${N_TRIALS:-50}
FRESH_START=${FRESH_START:-1}

for TARGET in "autumn rock" "dim grass" "outdoor water"; do
  TARGET_TAG=$(echo "$TARGET" | tr ' ' '-')

  # Seed stage-1 AFR settings from best vanilla runs per held-out pair.
  case "$TARGET_TAG" in
    autumn-rock)
      STAGE1_BASE_LR=0.0006986960740598744
      STAGE1_CLASSIFIER_LR=0.0012794989281937562
      STAGE1_DROPOUT=0.0
      ;;
    dim-grass)
      STAGE1_BASE_LR=0.00021570008495483682
      STAGE1_CLASSIFIER_LR=0.0007490773637911799
      STAGE1_DROPOUT=0.5
      ;;
    outdoor-water)
      STAGE1_BASE_LR=0.000250957524758074
      STAGE1_CLASSIFIER_LR=0.002118046217184987
      STAGE1_DROPOUT=0.5
      ;;
    *)
      STAGE1_BASE_LR=""
      STAGE1_CLASSIFIER_LR=""
      STAGE1_DROPOUT=0.0
      ;;
  esac

  sbatch --export=NONE,HOME="$HOME",USER="$USER",PATH="$PATH",REPO_ROOT="$REPO_ROOT",TXTLIST_DIR="$TXTLIST_DIR",IMAGE_ROOT="$IMAGE_ROOT",OUTPUT_DIR="$OUTPUT_DIR",TARGET_DOMAINS="$TARGET",TARGET_TAG="$TARGET_TAG",TIMEOUT_SECONDS="$TIMEOUT_SECONDS",N_TRIALS="$N_TRIALS",FRESH_START="$FRESH_START",STAGE1_BASE_LR="$STAGE1_BASE_LR",STAGE1_CLASSIFIER_LR="$STAGE1_CLASSIFIER_LR",STAGE1_DROPOUT="$STAGE1_DROPOUT",STAGE1_WEIGHT_DECAY=1e-5,STUDY_NAME="nico_afr_sgd_${TARGET_TAG}",OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///$OUTPUT_DIR/afr_sgd_optuna_v1_${TARGET_TAG}.db}" \
    "$JOB_SCRIPT"
done

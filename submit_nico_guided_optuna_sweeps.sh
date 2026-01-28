#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOB_SCRIPT="$SCRIPT_DIR/run_nico_guided_optuna_sweep.sh"

REPO_ROOT=${REPO_ROOT:-/home/ryreu/guided_cnn/NICO_again/NICO_again}
TXTLIST_DIR=${TXTLIST_DIR:-/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist}
MASK_ROOT=${MASK_ROOT:-/home/ryreu/guided_cnn/code/HaveNicoLearn/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ryreu/guided_cnn/NICO_runs/output}
IMAGE_ROOT=${IMAGE_ROOT:-/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG}

mkdir -p "$SCRIPT_DIR/logsNICO"

TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-86400}
N_TRIALS=${N_TRIALS:-1000000}

for TARGET in "autumn" "rock" "dim" "grass" "outdoor" "water"; do
  TARGET_TAG=$(echo "$TARGET" | tr ' ' '-')
  sbatch --export=ALL,REPO_ROOT="$REPO_ROOT",TXTLIST_DIR="$TXTLIST_DIR",MASK_ROOT="$MASK_ROOT",OUTPUT_DIR="$OUTPUT_DIR",IMAGE_ROOT="$IMAGE_ROOT",TARGET_DOMAINS="$TARGET",TARGET_TAG="$TARGET_TAG",TIMEOUT_SECONDS="$TIMEOUT_SECONDS",N_TRIALS="$N_TRIALS",OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///$OUTPUT_DIR/optuna_${TARGET_TAG}.db}" \
    "$JOB_SCRIPT"
done

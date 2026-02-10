#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOB_SCRIPT="$SCRIPT_DIR/run_nico_swad_sgd_optuna_sweep.sh"

REPO_ROOT=${REPO_ROOT:-/home/ryreu/guided_cnn/NICO_again/NICO_again}
TXTLIST_DIR=${TXTLIST_DIR:-/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist}
IMAGE_ROOT=${IMAGE_ROOT:-/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ryreu/guided_cnn/NICO_runs/output}

mkdir -p "$SCRIPT_DIR/logsNICO"

TIMEOUT_SECONDS=${TIMEOUT_SECONDS-}
N_TRIALS=${N_TRIALS:-50}

for TARGET in "autumn" "rock" "dim" "grass" "outdoor" "water"; do
  TARGET_TAG=$(echo "$TARGET" | tr ' ' '-')
  sbatch --export=ALL,REPO_ROOT="$REPO_ROOT",TXTLIST_DIR="$TXTLIST_DIR",IMAGE_ROOT="$IMAGE_ROOT",OUTPUT_DIR="$OUTPUT_DIR",TARGET_DOMAINS="$TARGET",TARGET_TAG="$TARGET_TAG",TIMEOUT_SECONDS="$TIMEOUT_SECONDS",N_TRIALS="$N_TRIALS",FRESH_START=1,STUDY_NAME="nico_swad_sgd_${TARGET_TAG}",OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///$OUTPUT_DIR/optuna_swad_sgd_v1_${TARGET_TAG}.db}" \
    "$JOB_SCRIPT"
done

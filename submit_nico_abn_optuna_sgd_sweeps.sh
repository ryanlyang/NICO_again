#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOB_SCRIPT="$SCRIPT_DIR/run_nico_abn_optuna_sgd_sweep.sh"

REPO_ROOT=${REPO_ROOT:-/home/ryreu/guided_cnn/NICO_again/NICO_again}
TXTLIST_DIR=${TXTLIST_DIR:-/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist}
IMAGE_ROOT=${IMAGE_ROOT:-/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ryreu/guided_cnn/NICO_runs/output}

mkdir -p "$SCRIPT_DIR/logsNICO"

TIMEOUT_SECONDS=${TIMEOUT_SECONDS-}
N_TRIALS=${N_TRIALS:-50}
RERUN_BEST=${RERUN_BEST:-1}
RERUN_NUM_SEEDS=${RERUN_NUM_SEEDS:-5}
RERUN_SEED_START=${RERUN_SEED_START:-59}
FRESH_START=${FRESH_START:-1}

for TARGET in "autumn rock" "dim grass" "outdoor water"; do
  TARGET_TAG=$(echo "$TARGET" | tr ' ' '-')
  REPO_ROOT="$REPO_ROOT" \
  TXTLIST_DIR="$TXTLIST_DIR" \
  IMAGE_ROOT="$IMAGE_ROOT" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  TARGET_DOMAINS="$TARGET" \
  TARGET_TAG="$TARGET_TAG" \
  TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
  N_TRIALS="$N_TRIALS" \
  RERUN_BEST="$RERUN_BEST" \
  RERUN_NUM_SEEDS="$RERUN_NUM_SEEDS" \
  RERUN_SEED_START="$RERUN_SEED_START" \
  FRESH_START="$FRESH_START" \
  STUDY_NAME="nico_abn_sgd_${TARGET_TAG}" \
  OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///$OUTPUT_DIR/optuna_nico_abn_sgd_v1_${TARGET_TAG}.db}" \
  sbatch "$JOB_SCRIPT"
done

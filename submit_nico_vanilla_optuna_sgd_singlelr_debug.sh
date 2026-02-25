#!/bin/bash
set -Eeuo pipefail

# Debug-partition submit helper for one job that runs all 3 NICO target splits.
# Usage:
#   bash submit_nico_vanilla_optuna_sgd_singlelr_debug.sh
# Optional overrides:
#   N_TRIALS=1 FRESH_START=1 bash submit_nico_vanilla_optuna_sgd_singlelr_debug.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOB_SCRIPT="$SCRIPT_DIR/run_nico_vanilla_optuna_sgd_singlelr_debug_all3.sh"

REPO_ROOT=${REPO_ROOT:-/home/ryreu/guided_cnn/NICO_again/NICO_again}
TXTLIST_DIR=${TXTLIST_DIR:-/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist}
IMAGE_ROOT=${IMAGE_ROOT:-/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ryreu/guided_cnn/NICO_runs/output}

N_TRIALS=${N_TRIALS:-1}
FRESH_START=${FRESH_START:-1}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS-}
LR_LOW=${LR_LOW:-1e-5}
LR_HIGH=${LR_HIGH:-5e-2}

mkdir -p "$SCRIPT_DIR/logsNICO"

echo "Submitting debug job:"
echo "  targets:  autumn-rock, dim-grass, outdoor-water (single job)"
echo "  trials:   $N_TRIALS"
echo "  lr range: [$LR_LOW, $LR_HIGH]"

REPO_ROOT="$REPO_ROOT" \
TXTLIST_DIR="$TXTLIST_DIR" \
IMAGE_ROOT="$IMAGE_ROOT" \
OUTPUT_DIR="$OUTPUT_DIR" \
N_TRIALS="$N_TRIALS" \
FRESH_START="$FRESH_START" \
TIMEOUT_SECONDS="$TIMEOUT_SECONDS" \
LR_LOW="$LR_LOW" \
LR_HIGH="$LR_HIGH" \
sbatch \
  "$JOB_SCRIPT"

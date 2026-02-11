#!/bin/bash -l
# Quick smoke test for NICO strategies on tiny txtlist subsets.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=logsNICO/nico_strategy_smoke_%j.out
#SBATCH --error=logsNICO/nico_strategy_smoke_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=logsNICO
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
ENV_NAME=${ENV_NAME:-gals_a100}
conda activate "$ENV_NAME"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_DISABLED=true
export SAVE_CHECKPOINTS=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=${REPO_ROOT:-/home/ryreu/guided_cnn/NICO_again/NICO_again}
TXTLIST_DIR=${TXTLIST_DIR:-/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist}
IMAGE_ROOT=${IMAGE_ROOT:-/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG}
MASK_ROOT=${MASK_ROOT:-/home/ryreu/guided_cnn/code/HaveNicoLearn/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}
ATTENTION_ROOT=${ATTENTION_ROOT:-/home/ryreu/guided_cnn/NICO_runs/attention_maps/nico_gals_vit_b32}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ryreu/guided_cnn/NICO_runs/output/smoke_debug}
DATASET=${DATASET:-NICO}
TARGET_DOMAIN=${TARGET_DOMAIN:-autumn}
SMOKE_LINES=${SMOKE_LINES:-120}
SMOKE_WORKDIR=${SMOKE_WORKDIR:-/tmp/nico_smoke_txtlist_${SLURM_JOB_ID:-local}}

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "Missing REPO_ROOT: $REPO_ROOT" >&2
  exit 2
fi
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ ! -d "$TXTLIST_DIR/$DATASET" ]]; then
  echo "Missing txtlist dataset folder: $TXTLIST_DIR/$DATASET" >&2
  exit 2
fi
if [[ ! -d "$IMAGE_ROOT" ]]; then
  echo "Missing IMAGE_ROOT: $IMAGE_ROOT" >&2
  exit 2
fi
if [[ ! -d "$MASK_ROOT" ]]; then
  echo "Missing MASK_ROOT: $MASK_ROOT" >&2
  exit 2
fi
if [[ ! -d "$ATTENTION_ROOT" ]]; then
  echo "Missing ATTENTION_ROOT: $ATTENTION_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$SMOKE_WORKDIR/$DATASET"

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "txtlist src: $TXTLIST_DIR"
echo "txtlist smoke: $SMOKE_WORKDIR"
echo "image root: $IMAGE_ROOT"
echo "mask root: $MASK_ROOT"
echo "attention root: $ATTENTION_ROOT"
echo "output root: $OUTPUT_DIR"
echo "target domain: $TARGET_DOMAIN"
echo "smoke lines/file: $SMOKE_LINES"
which python

# Build tiny txtlist copy with first N lines per split file.
for f in "$TXTLIST_DIR/$DATASET"/*.txt; do
  b=$(basename "$f")
  head -n "$SMOKE_LINES" "$f" > "$SMOKE_WORKDIR/$DATASET/$b"
done

run_step() {
  local name="$1"
  shift
  echo "\n===== RUNNING: $name ====="
  set +e
  "$@"
  local rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    echo "[PASS] $name"
  else
    echo "[FAIL] $name (exit=$rc)"
  fi
  return $rc
}

FAILS=0

# 1) Guided (Optuna SGD)
run_step "guided_optuna_sgd" \
  srun --unbuffered python -u run_guided_optuna_sgd.py \
    --txtdir "$SMOKE_WORKDIR" \
    --dataset "$DATASET" \
    --image_root "$IMAGE_ROOT" \
    --mask_root "$MASK_ROOT" \
    --extra_mask_roots "" \
    --output_dir "$OUTPUT_DIR" \
    --target "$TARGET_DOMAIN" \
    --n_trials 1 \
    --rerun_best 0 \
    --num_workers 4 \
    --study_name "smoke_guided_${TARGET_DOMAIN}" \
    --storage "sqlite:///$OUTPUT_DIR/smoke_guided_${TARGET_DOMAIN}.db" || FAILS=$((FAILS+1))

# 2) GALS + ViT maps
run_step "gals_vit_optuna_sgd" \
  srun --unbuffered python -u run_nico_gals_vit_optuna_sgd.py \
    --txtdir "$SMOKE_WORKDIR" \
    --dataset "$DATASET" \
    --image_root "$IMAGE_ROOT" \
    --attention_root "$ATTENTION_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --target "$TARGET_DOMAIN" \
    --n_trials 1 \
    --rerun_best 0 \
    --num_workers 4 \
    --study_name "smoke_gals_${TARGET_DOMAIN}" \
    --storage "sqlite:///$OUTPUT_DIR/smoke_gals_${TARGET_DOMAIN}.db" || FAILS=$((FAILS+1))

# 3) Vanilla + SWAD
run_step "swad_optuna_sgd" \
  srun --unbuffered python -u run_nico_swad_sgd_optuna.py \
    --txtdir "$SMOKE_WORKDIR" \
    --dataset "$DATASET" \
    --image_root "$IMAGE_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --target "$TARGET_DOMAIN" \
    --n_trials 1 \
    --rerun_best 0 \
    --num_workers 4 \
    --study_name "smoke_swad_${TARGET_DOMAIN}" \
    --storage "sqlite:///$OUTPUT_DIR/smoke_swad_${TARGET_DOMAIN}.db" || FAILS=$((FAILS+1))

# 4) Guided + SWAD
run_step "guided_swad_optuna_sgd" \
  srun --unbuffered python -u run_nico_guided_swad_sgd_optuna.py \
    --txtdir "$SMOKE_WORKDIR" \
    --dataset "$DATASET" \
    --image_root "$IMAGE_ROOT" \
    --mask_root "$MASK_ROOT" \
    --extra_mask_roots "" \
    --output_dir "$OUTPUT_DIR" \
    --target "$TARGET_DOMAIN" \
    --n_trials 1 \
    --rerun_best 0 \
    --num_workers 4 \
    --study_name "smoke_guided_swad_${TARGET_DOMAIN}" \
    --storage "sqlite:///$OUTPUT_DIR/smoke_guided_swad_${TARGET_DOMAIN}.db" || FAILS=$((FAILS+1))

# 5) UpWeight
run_step "upweight_optuna_sgd" \
  srun --unbuffered python -u run_nico_upweight_optuna_sgd.py \
    --txtdir "$SMOKE_WORKDIR" \
    --dataset "$DATASET" \
    --image_root "$IMAGE_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --target "$TARGET_DOMAIN" \
    --n_trials 1 \
    --rerun_best 0 \
    --num_workers 4 \
    --study_name "smoke_upweight_${TARGET_DOMAIN}" \
    --storage "sqlite:///$OUTPUT_DIR/smoke_upweight_${TARGET_DOMAIN}.db" || FAILS=$((FAILS+1))

echo "\n===== SMOKE SUMMARY ====="
if [[ $FAILS -eq 0 ]]; then
  echo "All strategy smoke tests passed."
  exit 0
fi

echo "$FAILS strategy test(s) failed."
exit 1

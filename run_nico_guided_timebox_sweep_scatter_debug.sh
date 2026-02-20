#!/bin/bash -l
# Time-boxed guided Optuna sweep (autumn+rock), stops at 22h, then writes scatter plot.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=logsNICO/nico_guided_timebox_scatter_%j.out
#SBATCH --error=logsNICO/nico_guided_timebox_scatter_%j.err
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
MASK_ROOT=${MASK_ROOT:-/home/ryreu/guided_cnn/code/HaveNicoLearn/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}
IMAGE_ROOT=${IMAGE_ROOT:-/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG}
DATASET=${DATASET:-NICO}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ryreu/guided_cnn/NICO_runs/output/guided_timebox_autumn_rock}

TIME_BUDGET_HOURS=${TIME_BUDGET_HOURS:-22}
MAX_TRIALS=${MAX_TRIALS:-10000}
TARGET_DOMAINS=${TARGET_DOMAINS:-"autumn rock"}
STUDY_NAME=${STUDY_NAME:-nico_guided_timebox_scatter_${SLURM_JOB_ID:-manual}}
OPTUNA_STORAGE=${OPTUNA_STORAGE:-sqlite:///$OUTPUT_DIR/timebox_optuna_${SLURM_JOB_ID:-manual}.db}

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "Missing REPO_ROOT: $REPO_ROOT" >&2
  exit 2
fi
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ ! -f "$REPO_ROOT/run_nico_guided_timebox_sweep_scatter.py" ]]; then
  echo "Missing run_nico_guided_timebox_sweep_scatter.py in $REPO_ROOT" >&2
  exit 2
fi
if [[ ! -d "$TXTLIST_DIR/$DATASET" ]]; then
  echo "Missing txtlist dataset folder: $TXTLIST_DIR/$DATASET" >&2
  exit 2
fi
if [[ ! -d "$MASK_ROOT" ]]; then
  echo "Missing MASK_ROOT: $MASK_ROOT" >&2
  exit 2
fi
if [[ ! -d "$IMAGE_ROOT" ]]; then
  echo "Missing IMAGE_ROOT: $IMAGE_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "txtlist: $TXTLIST_DIR"
echo "Masks: $MASK_ROOT"
echo "Images: $IMAGE_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Targets: $TARGET_DOMAINS"
echo "Time budget (hours): $TIME_BUDGET_HOURS"
echo "Max Optuna trials: $MAX_TRIALS"
echo "Study name: $STUDY_NAME"
echo "Storage: $OPTUNA_STORAGE"
which python

srun --unbuffered python -u run_nico_guided_timebox_sweep_scatter.py \
  --txtdir "$TXTLIST_DIR" \
  --dataset "$DATASET" \
  --image_root "$IMAGE_ROOT" \
  --mask_root "$MASK_ROOT" \
  --target $TARGET_DOMAINS \
  --output_dir "$OUTPUT_DIR" \
  --time_budget_hours "$TIME_BUDGET_HOURS" \
  --max_trials "$MAX_TRIALS" \
  --study_name "$STUDY_NAME" \
  --storage "$OPTUNA_STORAGE" \
  --beta 10 \
  --ig_steps 16

#!/bin/bash -l
# Single-run Guided SGD "oracle curve" job: prints val_acc / optim_value / test_acc every epoch.
#
# Usage (example):
#   TARGET_DOMAINS="autumn" PRE_LR=1e-5 POST_LR=3e-5 ATTENTION_EPOCH=15 KL_LAMBDA_START=10 KL_INCREMENT=1 \
#     sbatch run_nico_guided_sgd_epoch_test_sweep.sh
#
# You can override paths by exporting: REPO_ROOT, TXTLIST_DIR, IMAGE_ROOT, MASK_ROOT, OUTPUT_DIR.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --output=logsNICOoptim/nico_guided_sgd_epoch_test_%j.out
#SBATCH --error=logsNICOoptim/nico_guided_sgd_epoch_test_%j.err
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=${REPO_ROOT:-/home/ryreu/guided_cnn/NICO_again/NICO_again}
TXTLIST_DIR=${TXTLIST_DIR:-/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist}
IMAGE_ROOT=${IMAGE_ROOT:-/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG}
MASK_ROOT=${MASK_ROOT:-/home/ryreu/guided_cnn/code/HaveNicoLearn/LearningToLook/code/WeCLIPPlus/results/val/prediction_cmap}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ryreu/guided_cnn/NICO_runs/output}

DATASET=${DATASET:-NICO}
# Default to autumn so you can `sbatch` without extra env vars.
TARGET_DOMAINS=${TARGET_DOMAINS:-autumn}
TARGET_TAG=${TARGET_TAG:-$(echo "$TARGET_DOMAINS" | tr ' ' '-')}

SEED=${SEED:-59}
NUM_EPOCHS=${NUM_EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-32}
BETA=${BETA:-0.1}

# Guided hyperparams (required)
PRE_LR=${PRE_LR:?Set PRE_LR (e.g., 1e-5)}
POST_LR=${POST_LR:?Set POST_LR (e.g., 1e-6)}
ATTENTION_EPOCH=${ATTENTION_EPOCH:?Set ATTENTION_EPOCH (e.g., 15)}
KL_LAMBDA_START=${KL_LAMBDA_START:?Set KL_LAMBDA_START (e.g., 1)}
KL_INCREMENT=${KL_INCREMENT:?Set KL_INCREMENT (e.g., 0.2)}

RUN_OUT=${RUN_OUT:-$OUTPUT_DIR/sgd_epoch_test/target_${TARGET_TAG}/seed_${SEED}/job_${SLURM_JOB_ID:-manual}}
mkdir -p "$RUN_OUT"

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "Missing REPO_ROOT: $REPO_ROOT" >&2
  exit 1
fi
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

if [[ ! -d "$TXTLIST_DIR/$DATASET" ]]; then
  echo "Missing txtlist dataset folder: $TXTLIST_DIR/$DATASET" >&2
  exit 1
fi
if [[ ! -d "$IMAGE_ROOT" ]]; then
  echo "Missing IMAGE_ROOT: $IMAGE_ROOT" >&2
  exit 1
fi
if [[ ! -d "$MASK_ROOT" ]]; then
  echo "Missing MASK_ROOT: $MASK_ROOT" >&2
  exit 1
fi
if [[ ! -f "$REPO_ROOT/run_nico_guided_sgd_epoch_test.py" ]]; then
  echo "Missing run_nico_guided_sgd_epoch_test.py in $REPO_ROOT" >&2
  exit 1
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "txtlist: $TXTLIST_DIR"
echo "Images: $IMAGE_ROOT"
echo "Masks: $MASK_ROOT"
echo "Output(run): $RUN_OUT"
echo "Targets: $TARGET_DOMAINS"
echo "Seed: $SEED"
echo "Epochs: $NUM_EPOCHS Batch: $BATCH_SIZE Beta: $BETA"
echo "pre_lr=$PRE_LR post_lr=$POST_LR att_epoch=$ATTENTION_EPOCH kl_start=$KL_LAMBDA_START kl_inc=$KL_INCREMENT"
which python

srun --unbuffered python -u run_nico_guided_sgd_epoch_test.py \
  --txtdir "$TXTLIST_DIR" \
  --dataset "$DATASET" \
  --image_root "$IMAGE_ROOT" \
  --mask_root "$MASK_ROOT" \
  --output_dir "$RUN_OUT" \
  --target $TARGET_DOMAINS \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --num_epochs "$NUM_EPOCHS" \
  --pre_lr "$PRE_LR" \
  --post_lr "$POST_LR" \
  --attention_epoch "$ATTENTION_EPOCH" \
  --kl_lambda_start "$KL_LAMBDA_START" \
  --kl_increment "$KL_INCREMENT" \
  --beta "$BETA"

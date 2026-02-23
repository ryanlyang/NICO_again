#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logsNICO/nico_guided_sgd_optuna_%j.out
#SBATCH --error=logsNICO/nico_guided_sgd_optuna_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=logsNICO
mkdir -p "$LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh

ENV_NAME=${ENV_NAME:-gals_a100}
BOOTSTRAP_ENV=${BOOTSTRAP_ENV:-0}
RECREATE_ENV=${RECREATE_ENV:-0}

if [[ "$BOOTSTRAP_ENV" -eq 1 ]]; then
  if [[ "$RECREATE_ENV" -eq 1 ]]; then
    conda env remove -n "$ENV_NAME" -y || true
  fi
  if ! conda env list | grep -E "^${ENV_NAME}[[:space:]]" >/dev/null; then
    conda create -y -n "$ENV_NAME" python=3.8
    conda activate "$ENV_NAME"
    conda install -y pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch -c nvidia -c conda-forge
    pip install opencv-python==4.6.0.66 optuna
    conda deactivate
  fi
fi

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
OUTPUT_DIR=${OUTPUT_DIR:-/home/ryreu/guided_cnn/NICO_runs/output}
TARGET_DOMAINS=${TARGET_DOMAINS:?Set TARGET_DOMAINS (e.g., "autumn")}

DATASET=${DATASET:-NICO}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS-}
N_TRIALS=${N_TRIALS:-50}
TARGET_TAG=${TARGET_TAG:-$(echo "$TARGET_DOMAINS" | tr ' ' '-')}
STUDY_NAME=${STUDY_NAME:-nico_guided_sgd_${TARGET_TAG}}
OPTUNA_STORAGE=${OPTUNA_STORAGE:-sqlite:///$OUTPUT_DIR/optuna_guided_sgd_v1_${TARGET_TAG}.db}
FRESH_START=${FRESH_START:-1}

# Guided Optuna sweep ranges
BASE_LR_LOW=${BASE_LR_LOW:-1e-5}
BASE_LR_HIGH=${BASE_LR_HIGH:-1e-3}
CLASSIFIER_LR_LOW=${CLASSIFIER_LR_LOW:-1e-4}
CLASSIFIER_LR_HIGH=${CLASSIFIER_LR_HIGH:-1e-2}
LR2_MULT_LOW=${LR2_MULT_LOW:-0.001}
LR2_MULT_HIGH=${LR2_MULT_HIGH:-1.0}
ATT_EPOCH_MIN=${ATT_EPOCH_MIN:-1}
ATT_EPOCH_MAX=${ATT_EPOCH_MAX:-29}
KL_START_LOW=${KL_START_LOW:-0.01}
KL_START_HIGH=${KL_START_HIGH:-5.0}
# Locked: no kl_increment sweep
KL_INC_LOW=${KL_INC_LOW:-0.0}
KL_INC_HIGH=${KL_INC_HIGH:-0.0}

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
if [[ ! -d "$MASK_ROOT" ]]; then
  echo "Missing MASK_ROOT: $MASK_ROOT" >&2
  exit 1
fi
if [[ ! -d "$IMAGE_ROOT" ]]; then
  echo "Missing IMAGE_ROOT: $IMAGE_ROOT" >&2
  exit 1
fi
if [[ ! -f "$REPO_ROOT/run_guided_optuna_sgd.py" ]]; then
  echo "Missing run_guided_optuna_sgd.py in $REPO_ROOT" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "$OPTUNA_STORAGE" ]]; then
  EXTRA_ARGS+=(--storage "$OPTUNA_STORAGE" --study_name "$STUDY_NAME")
  if [[ "$OPTUNA_STORAGE" == sqlite:///* ]]; then
    DB_PATH="${OPTUNA_STORAGE#sqlite:///}"
    if [[ "$FRESH_START" -eq 1 ]]; then
      rm -f "$DB_PATH"
    elif [[ -f "$DB_PATH" ]]; then
      EXTRA_ARGS+=(--load_if_exists)
    fi
  else
    if [[ "$FRESH_START" -ne 1 ]]; then
      EXTRA_ARGS+=(--load_if_exists)
    fi
  fi
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "txtlist: $TXTLIST_DIR"
echo "Masks: $MASK_ROOT"
echo "Images: $IMAGE_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Targets: $TARGET_DOMAINS"
echo "Trials: $N_TRIALS"
echo "Sweep ranges:"
echo "  base_lr: [$BASE_LR_LOW, $BASE_LR_HIGH]"
echo "  classifier_lr: [$CLASSIFIER_LR_LOW, $CLASSIFIER_LR_HIGH]"
echo "  lr2_mult: [$LR2_MULT_LOW, $LR2_MULT_HIGH]"
echo "  attention_epoch: [$ATT_EPOCH_MIN, $ATT_EPOCH_MAX]"
echo "  kl_lambda_start: [$KL_START_LOW, $KL_START_HIGH]"
echo "  kl_increment: [$KL_INC_LOW, $KL_INC_HIGH] (locked)"
if [[ -n "${TIMEOUT_SECONDS}" ]]; then
  echo "Timeout: $TIMEOUT_SECONDS"
fi
which python

if [[ -n "${TIMEOUT_SECONDS}" ]]; then
  EXTRA_ARGS+=(--timeout "$TIMEOUT_SECONDS")
fi

srun --unbuffered python -u run_guided_optuna_sgd.py \
  --txtdir "$TXTLIST_DIR" \
  --dataset "$DATASET" \
  --image_root "$IMAGE_ROOT" \
  --mask_root "$MASK_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --target $TARGET_DOMAINS \
  --n_trials "$N_TRIALS" \
  --base_lr_low "$BASE_LR_LOW" \
  --base_lr_high "$BASE_LR_HIGH" \
  --classifier_lr_low "$CLASSIFIER_LR_LOW" \
  --classifier_lr_high "$CLASSIFIER_LR_HIGH" \
  --lr2_mult_low "$LR2_MULT_LOW" \
  --lr2_mult_high "$LR2_MULT_HIGH" \
  --att_epoch_min "$ATT_EPOCH_MIN" \
  --att_epoch_max "$ATT_EPOCH_MAX" \
  --kl_start_low "$KL_START_LOW" \
  --kl_start_high "$KL_START_HIGH" \
  --kl_inc_low "$KL_INC_LOW" \
  --kl_inc_high "$KL_INC_HIGH" \
  "${EXTRA_ARGS[@]}"

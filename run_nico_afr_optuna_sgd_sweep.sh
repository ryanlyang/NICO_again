#!/bin/bash -l
#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logsNICO/nico_afr_sgd_optuna_%j.out
#SBATCH --error=logsNICO/nico_afr_sgd_optuna_%j.err
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
    pip install optuna
    conda deactivate
  fi
fi

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
OUTPUT_DIR=${OUTPUT_DIR:-/home/ryreu/guided_cnn/NICO_runs/output}
TARGET_DOMAINS=${TARGET_DOMAINS:?Set TARGET_DOMAINS (e.g., "autumn rock")}

DATASET=${DATASET:-NICO}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS-}
N_TRIALS=${N_TRIALS:-50}
TARGET_TAG=${TARGET_TAG:-$(echo "$TARGET_DOMAINS" | tr ' ' '-')}
STUDY_NAME=${STUDY_NAME:-nico_afr_sgd_${TARGET_TAG}}
OPTUNA_STORAGE=${OPTUNA_STORAGE:-sqlite:///$OUTPUT_DIR/afr_sgd_optuna_v1_${TARGET_TAG}.db}
FRESH_START=${FRESH_START:-1}

# AFR setup
ERM_TRAIN_PROP=${ERM_TRAIN_PROP:-0.8}
STAGE1_EPOCHS=${STAGE1_EPOCHS:-30}
STAGE1_BATCH_SIZE=${STAGE1_BATCH_SIZE:-32}
STAGE1_LR=${STAGE1_LR:-3e-3}
STAGE1_MOMENTUM=${STAGE1_MOMENTUM:-0.9}
STAGE1_WEIGHT_DECAY=${STAGE1_WEIGHT_DECAY:-1e-4}

STAGE2_EPOCHS=${STAGE2_EPOCHS:-500}
STAGE2_LR=${STAGE2_LR:-1e-2}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}

# Waterbirds-style AFR tuning space
GAMMA_LOW=${GAMMA_LOW:-4.0}
GAMMA_HIGH=${GAMMA_HIGH:-20.0}
REG_COEFF_CHOICES=${REG_COEFF_CHOICES:-0.0,0.1,0.2,0.3,0.4}

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
if [[ ! -f "$REPO_ROOT/run_nico_afr_optuna_sgd.py" ]]; then
  echo "Missing run_nico_afr_optuna_sgd.py in $REPO_ROOT" >&2
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

if [[ -n "${TIMEOUT_SECONDS}" ]]; then
  EXTRA_ARGS+=(--timeout "$TIMEOUT_SECONDS")
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "txtlist: $TXTLIST_DIR"
echo "Images: $IMAGE_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Targets: $TARGET_DOMAINS"
echo "Trials: $N_TRIALS"
echo "AFR stage1: epochs=$STAGE1_EPOCHS batch=$STAGE1_BATCH_SIZE lr=$STAGE1_LR mom=$STAGE1_MOMENTUM wd=$STAGE1_WEIGHT_DECAY"
echo "AFR stage2: epochs=$STAGE2_EPOCHS lr=$STAGE2_LR eval_batch=$EVAL_BATCH_SIZE"
echo "AFR split: erm_train_prop=$ERM_TRAIN_PROP"
echo "AFR sweep: gamma=[$GAMMA_LOW,$GAMMA_HIGH], reg_coeff={${REG_COEFF_CHOICES}}"
if [[ -n "${TIMEOUT_SECONDS}" ]]; then
  echo "Timeout: $TIMEOUT_SECONDS"
fi
which python

srun --unbuffered python -u run_nico_afr_optuna_sgd.py \
  --txtdir "$TXTLIST_DIR" \
  --dataset "$DATASET" \
  --image_root "$IMAGE_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --target $TARGET_DOMAINS \
  --n_trials "$N_TRIALS" \
  --erm_train_prop "$ERM_TRAIN_PROP" \
  --stage1_epochs "$STAGE1_EPOCHS" \
  --stage1_batch_size "$STAGE1_BATCH_SIZE" \
  --stage1_lr "$STAGE1_LR" \
  --stage1_momentum "$STAGE1_MOMENTUM" \
  --stage1_weight_decay "$STAGE1_WEIGHT_DECAY" \
  --stage2_epochs "$STAGE2_EPOCHS" \
  --stage2_lr "$STAGE2_LR" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --gamma_low "$GAMMA_LOW" \
  --gamma_high "$GAMMA_HIGH" \
  --reg_coeff_choices "$REG_COEFF_CHOICES" \
  "${EXTRA_ARGS[@]}"

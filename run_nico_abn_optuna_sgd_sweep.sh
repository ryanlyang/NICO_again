#!/bin/bash -l
# NICO ABN Optuna SGD sweep

#SBATCH --account=reu-aisocial
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logsNICO/nico_abn_sgd_optuna_%j.out
#SBATCH --error=logsNICO/nico_abn_sgd_optuna_%j.err
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
N_TRIALS=${N_TRIALS:-20}
RERUN_BEST=${RERUN_BEST:-1}
RERUN_NUM_SEEDS=${RERUN_NUM_SEEDS:-5}
RERUN_SEED_START=${RERUN_SEED_START:-59}
NUM_WORKERS=${NUM_WORKERS:-4}

TARGET_TAG=${TARGET_TAG:-$(echo "$TARGET_DOMAINS" | tr ' ' '-')}
STUDY_NAME=${STUDY_NAME:-nico_abn_sgd_${TARGET_TAG}}
OPTUNA_STORAGE=${OPTUNA_STORAGE:-sqlite:///$OUTPUT_DIR/optuna_nico_abn_sgd_v1_${TARGET_TAG}.db}
FRESH_START=${FRESH_START:-1}

# Sweep ranges
BASE_LR_LOW=${BASE_LR_LOW:-1e-5}
BASE_LR_HIGH=${BASE_LR_HIGH:-5e-2}
CLASSIFIER_LR_LOW=${CLASSIFIER_LR_LOW:-1e-5}
CLASSIFIER_LR_HIGH=${CLASSIFIER_LR_HIGH:-5e-2}

# Fixed ABN/SGD settings
NUM_EPOCHS=${NUM_EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-32}
MOMENTUM=${MOMENTUM:-0.9}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
MAIN_CLS_WEIGHT=${MAIN_CLS_WEIGHT:-1.0}
ABN_CLS_WEIGHT=${ABN_CLS_WEIGHT:-1.0}
ABN_ADD_AFTER_ATTENTION=${ABN_ADD_AFTER_ATTENTION:-1}
PRETRAINED=${PRETRAINED:-1}
ABN_WEIGHTS_PATH=${ABN_WEIGHTS_PATH:-/home/ryreu/guided_cnn/NICO_again/NICO_again/GALS/weights/resnet50_abn_imagenet.pth.tar}

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
if [[ ! -f "$REPO_ROOT/run_nico_abn_optuna_sgd.py" ]]; then
  echo "Missing run_nico_abn_optuna_sgd.py in $REPO_ROOT" >&2
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
echo "Rerun best: $RERUN_BEST"
echo "Rerun seeds: $RERUN_NUM_SEEDS (start=$RERUN_SEED_START)"
echo "Sweep ranges:"
echo "  base_lr: [$BASE_LR_LOW, $BASE_LR_HIGH]"
echo "  classifier_lr: [$CLASSIFIER_LR_LOW, $CLASSIFIER_LR_HIGH]"
echo "Fixed settings:"
echo "  epochs=$NUM_EPOCHS batch_size=$BATCH_SIZE momentum=$MOMENTUM weight_decay=$WEIGHT_DECAY"
echo "  main_cls_weight=$MAIN_CLS_WEIGHT abn_cls_weight=$ABN_CLS_WEIGHT"
echo "  add_after_attention=$ABN_ADD_AFTER_ATTENTION pretrained=$PRETRAINED"
if [[ -f "$ABN_WEIGHTS_PATH" ]]; then
  echo "  abn_weights: $ABN_WEIGHTS_PATH"
else
  echo "  abn_weights: $ABN_WEIGHTS_PATH (not found; script will fallback to torchvision init)"
fi
if [[ -n "${TIMEOUT_SECONDS}" ]]; then
  echo "Timeout: $TIMEOUT_SECONDS"
fi
which python

srun --unbuffered python -u run_nico_abn_optuna_sgd.py \
  --txtdir "$TXTLIST_DIR" \
  --dataset "$DATASET" \
  --image_root "$IMAGE_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --target $TARGET_DOMAINS \
  --n_trials "$N_TRIALS" \
  --num_workers "$NUM_WORKERS" \
  --num_epochs "$NUM_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --momentum "$MOMENTUM" \
  --weight_decay "$WEIGHT_DECAY" \
  --main_cls_weight "$MAIN_CLS_WEIGHT" \
  --abn_cls_weight "$ABN_CLS_WEIGHT" \
  --abn_add_after_attention "$ABN_ADD_AFTER_ATTENTION" \
  --pretrained "$PRETRAINED" \
  --abn_weights_path "$ABN_WEIGHTS_PATH" \
  --base_lr_low "$BASE_LR_LOW" \
  --base_lr_high "$BASE_LR_HIGH" \
  --classifier_lr_low "$CLASSIFIER_LR_LOW" \
  --classifier_lr_high "$CLASSIFIER_LR_HIGH" \
  --rerun_best "$RERUN_BEST" \
  --rerun_num_seeds "$RERUN_NUM_SEEDS" \
  --rerun_seed_start "$RERUN_SEED_START" \
  "${EXTRA_ARGS[@]}"

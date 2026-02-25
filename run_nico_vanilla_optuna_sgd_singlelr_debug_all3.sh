#!/bin/bash -l
# One debug-partition job that runs all 3 NICO target splits sequentially
# for the vanilla single-LR Optuna path.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logsNICO/nico_vanilla_singlelr_debug_all3_%j.out
#SBATCH --error=logsNICO/nico_vanilla_singlelr_debug_all3_%j.err
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1

REPO_ROOT=${REPO_ROOT:-/home/ryreu/guided_cnn/NICO_again/NICO_again}
TXTLIST_DIR=${TXTLIST_DIR:-/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist}
IMAGE_ROOT=${IMAGE_ROOT:-/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG}
OUTPUT_DIR=${OUTPUT_DIR:-/home/ryreu/guided_cnn/NICO_runs/output}

DATASET=${DATASET:-NICO}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS-}
N_TRIALS=${N_TRIALS:-1}
FRESH_START=${FRESH_START:-1}
LR_LOW=${LR_LOW:-1e-5}
LR_HIGH=${LR_HIGH:-5e-2}

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
if [[ ! -f "$REPO_ROOT/run_vanilla_optuna_sgd.py" ]]; then
  echo "Missing run_vanilla_optuna_sgd.py in $REPO_ROOT" >&2
  exit 1
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "txtlist: $TXTLIST_DIR"
echo "Images: $IMAGE_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Trials per split: $N_TRIALS"
echo "Single LR range: [$LR_LOW, $LR_HIGH]"
if [[ -n "${TIMEOUT_SECONDS}" ]]; then
  echo "Timeout per split: $TIMEOUT_SECONDS"
fi
which python

for TARGET in "autumn rock" "dim grass" "outdoor water"; do
  TARGET_TAG=$(echo "$TARGET" | tr ' ' '-')
  STUDY_NAME=${STUDY_NAME_OVERRIDE:-nico_vanilla_singlelr_debug_${TARGET_TAG}}
  OPTUNA_STORAGE=${OPTUNA_STORAGE_OVERRIDE:-sqlite:///$OUTPUT_DIR/vanilla_singlelr_debug_optuna_v1_${TARGET_TAG}.db}

  EXTRA_ARGS=(--storage "$OPTUNA_STORAGE" --study_name "$STUDY_NAME")
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
  if [[ -n "${TIMEOUT_SECONDS}" ]]; then
    EXTRA_ARGS+=(--timeout "$TIMEOUT_SECONDS")
  fi

  echo ""
  echo "===== RUNNING TARGET: $TARGET ====="
  srun --unbuffered python -u run_vanilla_optuna_sgd.py \
    --txtdir "$TXTLIST_DIR" \
    --dataset "$DATASET" \
    --image_root "$IMAGE_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --run_name_prefix "vanilla_singlelr_debug" \
    --target $TARGET \
    --n_trials "$N_TRIALS" \
    --single_lr \
    --base_lr_low "$LR_LOW" \
    --base_lr_high "$LR_HIGH" \
    "${EXTRA_ARGS[@]}"
done

echo ""
echo "All three target splits completed."

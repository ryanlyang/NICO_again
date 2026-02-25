#!/bin/bash -l
# Debug-partition CLIP ViT zero-shot eval on NICO++ for all 3 held-out domain pairs.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logsNICO/nico_clip_vit_zeroshot_debug_all3_%j.out
#SBATCH --error=logsNICO/nico_clip_vit_zeroshot_debug_all3_%j.err
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
    pip install opencv-python==4.6.0.66
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
PHASE=${PHASE:-test}
PROMPT_SET=${PROMPT_SET:-gals}
CLASS_NAME_SOURCE=${CLASS_NAME_SOURCE:-canonical}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-4}
MODEL_NAME=${MODEL_NAME:-ViT-B/32}

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
if [[ ! -f "$REPO_ROOT/run_nico_clip_vit_zeroshot.py" ]]; then
  echo "Missing run_nico_clip_vit_zeroshot.py in $REPO_ROOT" >&2
  exit 1
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "txtlist: $TXTLIST_DIR"
echo "images: $IMAGE_ROOT"
echo "output: $OUTPUT_DIR"
echo "model: $MODEL_NAME"
echo "phase: $PHASE"
echo "prompt_set: $PROMPT_SET"
echo "class_name_source: $CLASS_NAME_SOURCE"
echo "batch_size: $BATCH_SIZE"
which python

for TARGET in "autumn rock" "dim grass" "outdoor water"; do
  TARGET_TAG=$(echo "$TARGET" | tr ' ' '-')
  echo ""
  echo "===== CLIP ZERO-SHOT | TARGETS: $TARGET ====="
  srun --unbuffered python -u run_nico_clip_vit_zeroshot.py \
    --txtdir "$TXTLIST_DIR" \
    --dataset "$DATASET" \
    --image_root "$IMAGE_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --target $TARGET \
    --phase "$PHASE" \
    --model_name "$MODEL_NAME" \
    --prompt_set "$PROMPT_SET" \
    --class_name_source "$CLASS_NAME_SOURCE" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS"
done

echo ""
echo "Done: CLIP ViT zero-shot evaluated for all three held-out domain pairs."


#!/bin/bash -l
# Generate GALS-style CLIP ViT attention maps for NICO++ in chunks.

#SBATCH --account=reu-aisocial
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-23:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --output=/home/ryreu/guided_cnn/logsNICO/nico_vit_attention_gen_%j.out
#SBATCH --error=/home/ryreu/guided_cnn/logsNICO/nico_vit_attention_gen_%j.err
#SBATCH --signal=TERM@120

set -Eeuo pipefail

LOG_DIR=/home/ryreu/guided_cnn/logsNICO
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
    pip install ftfy regex tqdm
    conda deactivate
  fi
fi

conda activate "$ENV_NAME"

export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_DISABLED=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

REPO_ROOT=${REPO_ROOT:-/home/ryreu/guided_cnn/NICO_again/NICO_again}
TXTLIST_DIR=${TXTLIST_DIR:-/home/ryreu/guided_cnn/NICO_again/NICO_again/txtlist}
IMAGE_ROOT=${IMAGE_ROOT:-/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG}
ATTENTION_ROOT=${ATTENTION_ROOT:-/home/ryreu/guided_cnn/NICO_runs/attention_maps/nico_gals_vit_b32}
DATASET=${DATASET:-NICO}
DOMAINS=${DOMAINS:-autumn,rock,dim,grass,outdoor,water}
export REPO_ROOT TXTLIST_DIR IMAGE_ROOT ATTENTION_ROOT DATASET DOMAINS

cd "$REPO_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

if [[ ! -d "$TXTLIST_DIR/$DATASET" ]]; then
  echo "[ERROR] Missing txtlist dataset dir: $TXTLIST_DIR/$DATASET" >&2
  exit 2
fi
if [[ ! -d "$IMAGE_ROOT" ]]; then
  echo "[ERROR] Missing image root: $IMAGE_ROOT" >&2
  exit 2
fi
if [[ ! -f "$REPO_ROOT/generate_nico_gals_vit_attention.py" ]]; then
  echo "[ERROR] Missing script: $REPO_ROOT/generate_nico_gals_vit_attention.py" >&2
  exit 2
fi

mkdir -p "$ATTENTION_ROOT"

BPE_PATH="$REPO_ROOT/GALS/CLIP/clip/bpe_simple_vocab_16e6.txt.gz"
if [[ ! -f "$BPE_PATH" ]]; then
  echo "[GEN] Missing CLIP BPE vocab. Downloading to: $BPE_PATH"
  mkdir -p "$(dirname "$BPE_PATH")"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "$BPE_PATH" \
      "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz" || \
    curl -L --fail -o "$BPE_PATH" \
      "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$BPE_PATH" \
      "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz" || \
    wget -O "$BPE_PATH" \
      "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"
  else
    echo "[ERROR] Need curl or wget to download missing BPE vocab: $BPE_PATH" >&2
    exit 2
  fi
fi

echo "[$(date)] Host: $(hostname)"
echo "Repo: $REPO_ROOT"
echo "txtlist: $TXTLIST_DIR"
echo "images: $IMAGE_ROOT"
echo "output attention: $ATTENTION_ROOT"
echo "dataset: $DATASET"
echo "domains: $DOMAINS"
which python

CHUNK_SIZE=${CHUNK_SIZE:-1000}
OVERWRITE=${OVERWRITE:-0}
TOTAL_N=${TOTAL_N-}

if [[ -z "${TOTAL_N}" ]]; then
  echo "[GEN] Counting unique image records from txtlists..."
  TOTAL_N=$(python - <<'PY'
import os
import sys

repo_root = os.environ["REPO_ROOT"]
txtdir = os.environ["TXTLIST_DIR"]
dataset = os.environ.get("DATASET", "NICO")
domains = [d.strip() for d in os.environ.get("DOMAINS", "").split(",") if d.strip()]

sys.path.insert(0, repo_root)
from domainbed.datasets import _dataset_info

seen = set()
for d in domains:
    for phase in ("train", "val", "test"):
        txt_file = os.path.join(txtdir, dataset, f"{d}_{phase}.txt")
        if not os.path.exists(txt_file):
            continue
        names, _ = _dataset_info(txt_file)
        for n in names:
            seen.add(n)
print(len(seen))
PY
)
fi

if [[ -z "${TOTAL_N}" || "${TOTAL_N}" -le 0 ]]; then
  echo "[ERROR] Invalid TOTAL_N=${TOTAL_N}" >&2
  exit 2
fi

echo "[GEN] chunk size: $CHUNK_SIZE"
echo "[GEN] total records: $TOTAL_N"

for ((start=0; start<TOTAL_N; start+=CHUNK_SIZE)); do
  end=$((start + CHUNK_SIZE))
  if [[ "$end" -gt "$TOTAL_N" ]]; then
    end=$TOTAL_N
  fi
  echo "[GEN] chunk: START_IDX=$start END_IDX=$end"
  srun --unbuffered python -u generate_nico_gals_vit_attention.py \
    --txtdir "$TXTLIST_DIR" \
    --dataset "$DATASET" \
    --image_root "$IMAGE_ROOT" \
    --output_root "$ATTENTION_ROOT" \
    --domains "$DOMAINS" \
    --start_idx "$start" \
    --end_idx "$end" \
    --overwrite "$OVERWRITE"
done

echo "[GEN] Done."

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-$ROOT/.venv}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Missing venv at $VENV_DIR — run: bash scripts/install_ubuntu_jupyter.sh"
  exit 1
fi

source "$VENV_DIR/bin/activate"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"

LIMIT="${1:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"

echo "SLM_MODEL_ID=${SLM_MODEL_ID:-Qwen/Qwen3.5-2B}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "Dataset tip-2 116 → results/benchmark/"
if [ "$LIMIT" != "0" ]; then
  python main.py --step benchmark-slm --limit "$LIMIT" --batch-size "$BATCH_SIZE"
else
  python main.py --step benchmark-slm --batch-size "$BATCH_SIZE"
fi

echo "Done. Results:"
ls -la results/benchmark/slm_*.json results/benchmark/slm_*.md 2>/dev/null || true

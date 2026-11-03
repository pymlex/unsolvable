#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
source "$VENV_DIR/bin/activate"
export PYTHONUNBUFFERED=1
mkdir -p results
python main.py --step train "$@" 2>&1 | tee -a results/dpo-train.log

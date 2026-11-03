#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
if [ ! -x "$VENV_DIR/bin/python" ]; then
  bash scripts/install_ubuntu_jupyter.sh
fi
source "$VENV_DIR/bin/activate"
export PYTHONUNBUFFERED=1
mkdir -p results
python main.py --step all "$@" 2>&1 | tee -a results/dpo-run.log

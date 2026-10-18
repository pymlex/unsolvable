#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-$ROOT/.venv}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  bash scripts/install_ubuntu_jupyter.sh
fi

source "$VENV_DIR/bin/activate"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

bash scripts/run_benchmark_slm.sh

if [ -n "${ZVENOAI_API_KEY:-}" ]; then
  python main.py --step benchmark-deepseek
else
  echo "ZVENOAI_API_KEY unset — skipping DeepSeek benchmark"
fi

if [ -n "${GITHUB_TOKEN:-}" ] || [ -n "${GH_TOKEN:-}" ]; then
  bash scripts/push_results_github.sh "Ubuntu Jupyter: tip-2 SLM benchmark results"
else
  echo "GITHUB_TOKEN unset — results stay local under results/benchmark/"
fi

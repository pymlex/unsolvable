#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example — set HF_TOKEN (and ZVENOAI_API_KEY for DeepSeek steps)"
fi

if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" -eq 0 ]; then
  apt-get update -qq
  apt-get install -y -qq git build-essential curl ca-certificates
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

bootstrap_pip() {
  if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    return 0
  fi
  echo "pip missing — bootstrapping with get-pip.py"
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip-unsolvable.py
  "$PYTHON_BIN" /tmp/get-pip-unsolvable.py --user
  export PATH="${HOME}/.local/bin:${PATH}"
}

create_venv() {
  if [ -x "$VENV_DIR/bin/python" ]; then
    echo "Reusing venv at $VENV_DIR"
    return 0
  fi

  echo "Creating venv at $VENV_DIR"
  if "$PYTHON_BIN" -m venv "$VENV_DIR" 2>/tmp/unsolvable-venv.err; then
    return 0
  fi

  echo "python -m venv failed (ensurepip often broken on Clore images). Falling back to virtualenv."
  cat /tmp/unsolvable-venv.err >&2 || true
  rm -rf "$VENV_DIR"
  bootstrap_pip
  export PATH="${HOME}/.local/bin:${PATH}"
  "$PYTHON_BIN" -m pip install -q --user virtualenv
  "$PYTHON_BIN" -m virtualenv -p "$PYTHON_BIN" "$VENV_DIR"
}

create_venv

source "$VENV_DIR/bin/activate"

python -m pip install -q -U pip setuptools wheel
python -m pip install -q "torch>=2.5.0" --index-url "$TORCH_CUDA_INDEX"
python -m pip install -q -r requirements.txt

if [ "$(id -u)" -eq 0 ] && [ ! -e /root/venv ]; then
  ln -s "$VENV_DIR" /root/venv
  echo "Linked /root/venv -> $VENV_DIR"
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

echo
echo "Install complete."
echo "  source $VENV_DIR/bin/activate"
echo "  # edit .env: HF_TOKEN=...  (ZVENOAI_API_KEY for DeepSeek)"
echo "  bash scripts/run_benchmark_slm.sh"
echo "Artefacts: results/benchmark/slm_Qwen_Qwen3.5-2B.json"

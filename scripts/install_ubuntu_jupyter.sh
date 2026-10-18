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

have_skip_apt_tools() {
  command -v python3 >/dev/null 2>&1 && command -v curl >/dev/null 2>&1
}

if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" -eq 0 ]; then
  if have_skip_apt_tools; then
    echo "apt: python3 and curl already present — skipping apt"
  else
    echo "apt: updating package lists (non-fatal if NVIDIA key warnings appear)..."
    set +e
    apt-get update
    update_rc=$?
    if [ "$update_rc" -eq 0 ]; then
      echo "apt: installing git build-essential curl ca-certificates..."
      apt-get install -y git build-essential curl ca-certificates
      install_rc=$?
    else
      install_rc=1
    fi
    set -e
    if [ "$update_rc" -ne 0 ] || [ "$install_rc" -ne 0 ]; then
      echo "apt failed, continuing with venv/pip"
    fi
  fi
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

pip_user_install() {
  if "$PYTHON_BIN" -m pip install --help 2>/dev/null | grep -q -- '--break-system-packages'; then
    "$PYTHON_BIN" -m pip install -q --user --break-system-packages "$@"
  else
    "$PYTHON_BIN" -m pip install -q --user "$@"
  fi
}

create_venv() {
  if [ -x "$VENV_DIR/bin/python" ] && "$VENV_DIR/bin/python" -c "import sys" 2>/dev/null; then
    echo "Reusing venv at $VENV_DIR"
    return 0
  fi

  rm -rf "$VENV_DIR"
  echo "Creating venv at $VENV_DIR"
  if "$PYTHON_BIN" -m venv "$VENV_DIR" 2>/tmp/unsolvable-venv.err; then
    return 0
  fi

  echo "python -m venv failed (ensurepip often broken on Clore images). Falling back to virtualenv."
  cat /tmp/unsolvable-venv.err >&2 || true
  rm -rf "$VENV_DIR"
  bootstrap_pip
  export PATH="${HOME}/.local/bin:${PATH}"
  pip_user_install virtualenv
  "$PYTHON_BIN" -m virtualenv -p "$PYTHON_BIN" "$VENV_DIR"
}

create_venv

source "$VENV_DIR/bin/activate"

echo "pip: upgrading pip setuptools wheel..."
python -m pip install -U pip setuptools wheel
echo "pip: installing torch from $TORCH_CUDA_INDEX ..."
python -m pip install "torch>=2.5.0" --index-url "$TORCH_CUDA_INDEX"
echo "pip: installing requirements.txt ..."
python -m pip install -r requirements.txt

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

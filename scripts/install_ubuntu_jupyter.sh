#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example — fill HF_TOKEN, ZVENOAI_API_KEY, GITHUB_TOKEN"
fi

if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq git build-essential
fi

python3 -m pip install -q -U pip
python3 -m pip install -q -r requirements.txt

python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

echo "Install complete. Next:"
echo "  source .env   # or export tokens"
echo "  python main.py --step benchmark-slm"
echo "  python main.py --step benchmark-deepseek"

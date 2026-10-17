#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MESSAGE="${1:-Publish unsolvable benchmark and pipeline results}"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ -n "${GITHUB_NAME:-}" ]; then
  git config user.name "$GITHUB_NAME"
fi
if [ -n "${GITHUB_EMAIL:-}" ]; then
  git config user.email "$GITHUB_EMAIL"
fi

PATHS=(
  results/benchmark/
  results/preferences/
  results/plots/
  results/groups/
  results/synthesis/
  results/metrics/
)

for path in "${PATHS[@]}"; do
  mkdir -p "$path"
done

git add "${PATHS[@]}"
if git diff --cached --quiet; then
  echo "No changes in results/ to commit"
  exit 0
fi

git commit -m "$MESSAGE"
git push origin HEAD
echo "Pushed results to GitHub"

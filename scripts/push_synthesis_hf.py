#!/usr/bin/env python3
"""Push a local synthetic tip-2 JSONL as an HF dataset config/split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_loader import load_env_file

load_env_file("project")

from datasets import Dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--local", type=Path, required=True)
    p.add_argument("--repo", default="pymlex/reshuege-ru-2026")
    p.add_argument("--config-name", default="train-deepseek-v4-flash")
    p.add_argument("--split", default="train")
    args = p.parse_args()

    rows = []
    with args.local.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    ds = Dataset.from_list(rows)
    ds.push_to_hub(args.repo, config_name=args.config_name, split=args.split)
    print(f"Pushed {len(rows)} rows to {args.repo} config={args.config_name}")


if __name__ == "__main__":
    main()

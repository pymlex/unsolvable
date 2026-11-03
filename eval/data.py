"""Dataset loading helpers for tip-2 items."""

from __future__ import annotations

import json
from pathlib import Path


def normalize_row(row: dict) -> dict:
    """Map HF ``passage`` to in-code ``text`` and keep both fields."""
    out = dict(row)
    text = out.get("text") or out.get("passage") or ""
    out["text"] = text
    if "passage" not in out:
        out["passage"] = text
    return out


def load_rows(dataset: str, split: str = "test", *, config: str | None = None) -> list[dict]:
    """Load tip-2 rows from a local JSONL path or a Hugging Face dataset id."""
    path = Path(dataset)
    if path.exists() and path.suffix == ".jsonl":
        rows: list[dict] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(normalize_row(json.loads(line)))
        return rows

    from datasets import load_dataset

    from constants import DATASET_ID, EVAL_CONFIG, TRAIN_CONFIG

    if config is None and dataset == DATASET_ID:
        config = TRAIN_CONFIG if split == "train" else EVAL_CONFIG
    ds = load_dataset(dataset, name=config, split=split)
    return [normalize_row(dict(r)) for r in ds]

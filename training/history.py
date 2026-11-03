"""Persist every train log immediately; plotting never depends on a clean exit."""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from transformers import TrainerCallback
from utils.artifacts import read_jsonl, write_json


def clean_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if hasattr(value, "item"):
        return clean_value(value.item())
    return value


class TrainingHistoryCallback(TrainerCallback):
    def __init__(self, directory: Path, *, resume=False):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "train_history.jsonl"
        self.records = read_jsonl(self.path) if resume and self.path.exists() else []
        if not resume:
            self.path.write_text("", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        row = {"step": int(state.global_step), "epoch": float(state.epoch or 0),
               "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               **{key: clean_value(value) for key, value in logs.items()}}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.records.append(row)
        write_json(self.directory / "train_history.json", self.records)
        fields = list(dict.fromkeys(key for record in self.records for key in record))
        target = self.directory / "train_history.csv"
        tmp = target.with_suffix(".csv.tmp")
        with tmp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.records)
        tmp.replace(target)
        return control

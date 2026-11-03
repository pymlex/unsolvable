"""The DPO data contract: synthetic 500 for training, real 116 for monitoring."""

from __future__ import annotations

import json
import re
from pathlib import Path

from constants import DATASET_ID, DATASET_REVISION, EVAL_CONFIG, EVAL_SPLIT, TRAIN_CONFIG, TRAIN_SPLIT
from eval.prompt import item_text
from utils.artifacts import fingerprint, read_jsonl, write_json, write_jsonl
from utils.paths import RESULTS_DIR

PREPARED_DIR = RESULTS_DIR / "data"
MANIFEST_PATH = PREPARED_DIR / "manifest.json"


def canonical_item(row: dict) -> dict:
    """Hash only the actual input and labels, not storage-specific aliases."""
    return {"id": str(row["id"]), "passage": item_text(row), "options": list(row["options"]),
            "answer": str(row["answer"]), "explanation": str(row.get("explanation", "")),
            "highlighted_words": list(row.get("highlighted_words", []))}


def validate_sources(train: list[dict], test: list[dict], *, expected_train=500, expected_test=116) -> None:
    for name, rows, expected in (("train", train, expected_train), ("test", test, expected_test)):
        if len(rows) != expected:
            raise ValueError(f"{name}: expected {expected} items, got {len(rows)}")
        if len({str(row["id"]) for row in rows}) != len(rows):
            raise ValueError(f"{name}: duplicate item ids")
        for row in rows:
            highlights = re.findall(r"\*\*([^*]+)\*\*", item_text(row))
            answer = str(row["answer"])
            if (name == "train" and len(highlights) != 5) or len(row["options"]) != 5 or not all(str(o).strip() for o in row["options"]):
                raise ValueError(f"{name} item {row['id']}: invalid highlights/options")
            if not re.fullmatch(r"[1-5]{1,5}", answer) or answer != "".join(sorted(set(answer))):
                raise ValueError(f"{name} item {row['id']}: invalid gold answer {answer!r}")
            # HF metadata sometimes contains the dictionary form (правый),
            # while the passage has an inflected form (правом). The actual
            # **spans** are authoritative for rewrite preservation.
            if name == "train" and len(row.get("highlighted_words", highlights)) != 5:
                raise ValueError(f"{name} item {row['id']}: expected five highlight metadata entries")
    if {str(r["id"]) for r in train} & {str(r["id"]) for r in test}:
        raise ValueError("Train/test id overlap")
    if {fingerprint(item_text(r).split()) for r in train} & {fingerprint(item_text(r).split()) for r in test}:
        raise ValueError("Train/test passage overlap")


def prepare_data(*, refresh=False) -> Path:
    if MANIFEST_PATH.exists() and not refresh:
        _, _, existing = load_prepared()
        if DATASET_REVISION in {"main", existing["revision"]}:
            print(f"Using pinned source snapshot {existing['revision']} (500 train / 116 test)", flush=True)
            return MANIFEST_PATH
    from datasets import load_dataset
    from huggingface_hub import HfApi

    # Resolve once, then fetch BOTH configs at that immutable revision.
    revision = HfApi().dataset_info(DATASET_ID, revision=DATASET_REVISION).sha
    train = list(load_dataset(DATASET_ID, name=TRAIN_CONFIG, split=TRAIN_SPLIT, revision=revision))
    test = list(load_dataset(DATASET_ID, name=EVAL_CONFIG, split=EVAL_SPLIT, revision=revision))
    validate_sources(train, test)
    train, test = [canonical_item(r) for r in train], [canonical_item(r) for r in test]
    write_jsonl(PREPARED_DIR / "train.jsonl", train)
    write_jsonl(PREPARED_DIR / "test.jsonl", test)
    manifest = {"dataset": DATASET_ID, "revision": revision,
                "train": {"config": TRAIN_CONFIG, "split": TRAIN_SPLIT, "n": len(train), "fingerprint": fingerprint(train)},
                "test": {"config": EVAL_CONFIG, "split": EVAL_SPLIT, "n": len(test), "fingerprint": fingerprint(test)}}
    write_json(MANIFEST_PATH, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return MANIFEST_PATH


def load_prepared() -> tuple[list[dict], list[dict], dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("Run python main.py --step prepare first")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    train, test = read_jsonl(PREPARED_DIR / "train.jsonl"), read_jsonl(PREPARED_DIR / "test.jsonl")
    validate_sources(train, test)
    if manifest["dataset"] != DATASET_ID:
        raise ValueError("Prepared data belongs to another dataset; run prepare again")
    for name, rows, config, split in (("train", train, TRAIN_CONFIG, TRAIN_SPLIT), ("test", test, EVAL_CONFIG, EVAL_SPLIT)):
        meta = manifest[name]
        if meta["config"] != config or meta["split"] != split or meta["fingerprint"] != fingerprint(rows):
            raise ValueError(f"{name}: prepared snapshot/config changed; run prepare again")
    return train, test, manifest

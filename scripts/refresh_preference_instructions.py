"""Relax instructions/diagnostics while preserving all 500 texts and scores."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config_loader import load_env_file
load_env_file()
import csv
import json
import shutil
from datetime import datetime, timezone
from eval.ege_data import load_prepared
from generation.prompts import build_item_paraphrase_prompt
from judge.ege_quality import QualityGate, CHECK_VERSION
from preferences.build_dpo_dataset import PAIRS_PATH, PAIRS_META_PATH, publish_preferences
from utils.artifacts import fingerprint, read_jsonl, write_json, write_jsonl


def refresh():
    train, _, source = load_prepared()
    by_id = {str(r["id"]): r for r in train}
    pairs = read_jsonl(PAIRS_PATH)
    meta = json.loads(PAIRS_META_PATH.read_text(encoding="utf-8"))
    assert len(pairs) == len(train) == 500 and meta["source"] == source
    unchanged = [(p["item_id"], p["chosen"], p["rejected"], p["score_chosen"], p["score_rejected"], p["gap"]) for p in pairs]
    backup = PAIRS_PATH.parent / "history" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-before-relaxed-instructions")
    backup.mkdir(parents=True, exist_ok=True)
    for path in (PAIRS_PATH, PAIRS_META_PATH, PAIRS_PATH.with_suffix(".csv")):
        if path.exists():
            shutil.copy2(path, backup / path.name)
    gate = QualityGate()
    for pair in pairs:
        row = by_id[pair["item_id"]]
        pair.setdefault("generation_prompt", pair["prompt"])
        pair["prompt"] = build_item_paraphrase_prompt(row)
        for side in ("chosen", "rejected"):
            pair[side + "_quality"] = gate.check(row, {"text": pair[side]})
    assert unchanged == [(p["item_id"], p["chosen"], p["rejected"], p["score_chosen"], p["score_rejected"], p["gap"]) for p in pairs]
    meta["instruction_refresh"] = {"previous_fingerprint": meta["pairs_fingerprint"], "quality_policy": CHECK_VERSION,
                                    "note": "All 500 chosen/rejected texts and solver scores unchanged; generation_prompt retains original sampling instructions."}
    meta["pairs_fingerprint"] = fingerprint(pairs)
    write_jsonl(PAIRS_PATH, pairs)
    write_json(PAIRS_META_PATH, meta)
    with PAIRS_PATH.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["item_id", "prompt", "generation_prompt", "chosen", "rejected", "score_chosen", "score_rejected", "gap"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(pairs)
    print("[instructions] Updated all 500 prompts; texts/scores preserved; legacy artifacts saved:", backup, flush=True)


if __name__ == "__main__":
    refresh()
    if "--no-publish" not in sys.argv:
        publish_preferences()

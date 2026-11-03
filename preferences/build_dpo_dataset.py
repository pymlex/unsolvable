"""Train-only DPO pairs: two small-model rewrites ranked by a frozen solver."""

from __future__ import annotations

import math
from pathlib import Path

from constants import (DPO_CANDIDATES, DPO_SEED, GENERATION_TEMPERATURE, LENGTH_MARGIN,
                       PAIRWISE_GAP_TAU, PARAPHRASE_MAX_NEW_TOKENS, SLM_MODEL_ID, SOLVER_MODEL_ID)
from eval.ege_data import load_prepared
from eval.model_io import load_slm
from generation.ege_paraphrase import generate_item_paraphrases
from generation.prompts import build_item_paraphrase_prompt
from judge.ege_quality import CHECK_VERSION, QualityGate
from scoring.frozen_solver import FrozenSolver
from utils.artifacts import fingerprint, read_jsonl, write_json, write_jsonl
from utils.paths import PREFERENCES_DIR, ensure_result_dirs

PAIRS_PATH = PREFERENCES_DIR / "dpo_preferences.jsonl"
PAIRS_META_PATH = PREFERENCES_DIR / "dpo_manifest.json"
CACHE_PATH = PREFERENCES_DIR / "candidates.jsonl"


def select_pair(row: dict, candidates: list[dict], *, tau=PAIRWISE_GAP_TAU) -> dict | None:
    if len(candidates) != 2 or not all(c["quality"]["valid"] for c in candidates):
        return None
    if candidates[0]["text"] == candidates[1]["text"]:
        return None
    if not all(math.isfinite(c["score"]) for c in candidates):
        return None
    chosen, rejected = sorted(candidates, key=lambda c: c["score"])
    gap = rejected["score"] - chosen["score"]
    if gap <= 0 or gap < tau:
        return None
    return {"item_id": str(row["id"]), "prompt": build_item_paraphrase_prompt(row),
            "chosen": chosen["text"], "rejected": rejected["text"],
            "score_chosen": chosen["score"], "score_rejected": rejected["score"], "gap": gap,
            "chosen_quality": chosen["quality"], "rejected_quality": rejected["quality"]}


def build_preference_dataset(*, limit=0, device=None) -> Path:
    from transformers import set_seed

    ensure_result_dirs()
    train, _, manifest = load_prepared()
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    source = train[:limit] if limit else train
    generator, tokenizer, _ = load_slm(SLM_MODEL_ID, device)
    if SOLVER_MODEL_ID == SLM_MODEL_ID:
        solver = FrozenSolver(generator, tokenizer, model_id=SOLVER_MODEL_ID)
    else:
        solver_model, solver_tokenizer, _ = load_slm(SOLVER_MODEL_ID, device)
        solver = FrozenSolver(solver_model, solver_tokenizer, model_id=SOLVER_MODEL_ID)
    gate = QualityGate()
    cache = {r["key"]: r for r in read_jsonl(CACHE_PATH)} if CACHE_PATH.exists() else {}
    settings = {"version": 2, "generator": SLM_MODEL_ID, "solver": SOLVER_MODEL_ID,
                "seed": DPO_SEED, "samples": DPO_CANDIDATES, "temperature": GENERATION_TEMPERATURE,
                "max_new_tokens": PARAPHRASE_MAX_NEW_TOKENS, "length_margin": LENGTH_MARGIN,
                "structural_check_version": CHECK_VERSION, "source_revision": manifest["revision"],
                "score": "gold_logprob_sum_with_eos_no_thinking"}
    pairs, records = [], []
    for index, row in enumerate(source):
        key = fingerprint({"settings": settings, "prompt": build_item_paraphrase_prompt(row)})
        cached = cache.get(key)
        if cached is None:
            set_seed(DPO_SEED + index)
            candidates = generate_item_paraphrases(generator, tokenizer, [row], num_samples=2)[0]
            for candidate in candidates:
                candidate["quality"] = gate.check(row, candidate)
                candidate["score"] = solver.score(row, candidate["text"]) if candidate["quality"]["valid"] else None
            cached = {"key": key, "item_id": str(row["id"]), "candidates": candidates}
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            import json
            with CACHE_PATH.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(cached, ensure_ascii=False, allow_nan=False) + "\n")
        pair = select_pair(row, cached["candidates"])
        records.append(cached)
        if pair:
            pairs.append(pair)
        print(f"[preferences] {index + 1}/{len(source)} | pairs={len(pairs)} | id={row['id']}", flush=True)
    write_jsonl(PAIRS_PATH, pairs)
    write_json(PAIRS_META_PATH, {"source": manifest, "settings": settings, "tau": PAIRWISE_GAP_TAU,
                               "n_source": len(source), "n_pairs": len(pairs), "partial": bool(limit),
                               "pairs_fingerprint": fingerprint(pairs)})
    gaps = [p["gap"] for p in pairs]
    write_json(PREFERENCES_DIR / "build_stats.json", {
        "train_rows": len(source), "pairs_built": len(pairs), "pairs_skipped": len(source) - len(pairs),
        "invalid_candidates": sum(not c["quality"]["valid"] for r in records for c in r["candidates"]),
        "gaps": gaps, "mean_gap": sum(gaps) / len(gaps) if gaps else None,
    })
    if not pairs:
        raise RuntimeError("No valid DPO pairs; inspect candidates.jsonl and build_stats.json before training")
    # Same portable artifacts as the detector project.
    import csv
    from datasets import Dataset
    with (PREFERENCES_DIR / "dpo_preferences.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["item_id", "prompt", "chosen", "rejected", "score_chosen", "score_rejected", "gap"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pairs)
    Dataset.from_list([{k: p[k] for k in ("prompt", "chosen", "rejected")} for p in pairs]).save_to_disk(str(PREFERENCES_DIR / "dpo_hf_dataset"))
    return PAIRS_PATH

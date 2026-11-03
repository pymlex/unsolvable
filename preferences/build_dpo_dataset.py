"""Train-only DPO pairs: two small-model rewrites ranked by a frozen solver."""

from __future__ import annotations

import math
from pathlib import Path

from constants import (DPO_CANDIDATES, DPO_SEED, GENERATION_BATCH_SIZE, GENERATION_TEMPERATURE, LENGTH_MARGIN,
                       PAIRWISE_GAP_TAU, SLM_MODEL_ID, SOLVER_MODEL_ID)
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


def candidate_key(settings: dict, row: dict) -> str:
    # Batch size is a throughput setting, not a change in the sampling policy.
    identity = {k: v for k, v in settings.items() if k not in {"generation_batch_size", "max_new_tokens"}}
    return fingerprint({"settings": identity, "prompt": build_item_paraphrase_prompt(row)})


def cached_candidate(cache: dict, settings: dict, row: dict) -> tuple[str, dict | None]:
    key = candidate_key(settings, row)
    if key in cache and all(not c.get("truncated") for c in cache[key]["candidates"]):
        return key, cache[key]
    cache.pop(key, None)
    # Preserve candidates from the first v4 runs when tuning GPU batch size.
    variants = [settings]
    if "max_new_tokens" in settings:
        variants.append({**settings, "max_new_tokens": 1024})
    for variant in variants:
        for previous_batch in (4, 8, 16, 32, 64, 128):
            legacy = {**variant, "generation_batch_size": previous_batch}
            possible = [legacy, {k: v for k, v in legacy.items() if k != "generation_batch_size"}]
            for identity in possible:
                old_key = fingerprint({"settings": identity, "prompt": build_item_paraphrase_prompt(row)})
                if old_key in cache and all(not c.get("truncated") for c in cache[old_key]["candidates"]):
                    record = {**cache[old_key], "key": key, "generation_batch_size": previous_batch}
                    cache[key] = record
                    return key, record
    return key, None


def select_pair(row: dict, candidates: list[dict], *, tau=PAIRWISE_GAP_TAU) -> dict:
    if len(candidates) != 2:
        raise ValueError(f"Item {row['id']} must have exactly two candidates")
    if not all(math.isfinite(c["score"]) for c in candidates):
        raise ValueError(f"Item {row['id']} has an invalid score")
    chosen, rejected = sorted(candidates, key=lambda c: c["score"])
    gap = rejected["score"] - chosen["score"]
    return {"item_id": str(row["id"]), "prompt": build_item_paraphrase_prompt(row),
            "chosen": chosen["text"], "rejected": rejected["text"],
            "score_chosen": chosen["score"], "score_rejected": rejected["score"], "gap": gap,
            "chosen_quality": chosen["quality"], "rejected_quality": rejected["quality"],
            "is_tie": gap == 0, "is_duplicate": chosen["text"] == rejected["text"],
            "below_probe_margin": gap < tau}


def build_preference_dataset(*, limit=0, device=None, publish=False) -> Path:
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
    settings = {"version": 4, "generator": SLM_MODEL_ID, "solver": SOLVER_MODEL_ID,
                "seed": DPO_SEED, "samples": DPO_CANDIDATES, "generation_batch_size": GENERATION_BATCH_SIZE, "temperature": GENERATION_TEMPERATURE,
                "max_new_tokens": None, "length_margin": LENGTH_MARGIN,
                "structural_check_version": CHECK_VERSION, "source_revision": manifest["revision"],
                "score": "gold_logprob_sum_with_eos_no_thinking_float32"}
    pairs, records = [], []
    if GENERATION_BATCH_SIZE < 1:
        raise ValueError("GENERATION_BATCH_SIZE must be positive")
    for start in range(0, len(source), GENERATION_BATCH_SIZE):
        chunk = source[start:start + GENERATION_BATCH_SIZE]
        keys = [cached_candidate(cache, settings, row)[0] for row in chunk]
        missing = [(row, key) for row, key in zip(chunk, keys, strict=True) if key not in cache]
        if missing:
            set_seed(DPO_SEED + start)
            groups = generate_item_paraphrases(generator, tokenizer, [row for row, _ in missing], num_samples=2)
            for (row, key), candidates in zip(missing, groups, strict=True):
                for candidate in candidates:
                    candidate["quality"] = gate.check(row, candidate)
                    candidate["score"] = solver.score(row, candidate["text"])
                cached = {"key": key, "item_id": str(row["id"]), "candidates": candidates,
                          "generation_batch_size": GENERATION_BATCH_SIZE}
                CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                import json
                with CACHE_PATH.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(cached, ensure_ascii=False, allow_nan=False) + "\n")
                cache[key] = cached
        for offset, (row, key) in enumerate(zip(chunk, keys, strict=True)):
            cached = cache[key]
            rescored = False
            for candidate in cached["candidates"]:
                if candidate.get("score") is None:
                    candidate["score"] = solver.score(row, candidate["text"])
                    rescored = True
            if rescored:
                import json
                with CACHE_PATH.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(cached, ensure_ascii=False, allow_nan=False) + "\n")
            pair = select_pair(row, cached["candidates"])
            records.append(cached)
            if pair:
                pairs.append(pair)
            print(f"[preferences] {start + offset + 1}/{len(source)} | pairs={len(pairs)} | id={row['id']}", flush=True)
    write_jsonl(PAIRS_PATH, pairs)
    write_json(PAIRS_META_PATH, {"source": manifest, "settings": settings, "tau": PAIRWISE_GAP_TAU,
                               "n_source": len(source), "n_pairs": len(pairs), "partial": bool(limit),
                               "pairs_fingerprint": fingerprint(pairs)})
    gaps = [p["gap"] for p in pairs]
    write_json(PREFERENCES_DIR / "build_stats.json", {
        "train_rows": len(source), "pairs_built": len(pairs), "pairs_skipped": len(source) - len(pairs),
        "invalid_candidates": sum(not c["quality"]["valid"] for r in records for c in r["candidates"]),
        "ties_retained": sum(p["is_tie"] for p in pairs),
        "duplicates_retained": sum(p["is_duplicate"] for p in pairs),
        "below_margin_retained": sum(p["below_probe_margin"] for p in pairs),
        "gaps": gaps, "mean_gap": sum(gaps) / len(gaps) if gaps else None,
    })
    if not pairs:
        raise RuntimeError("No valid DPO pairs; inspect candidates.jsonl and build_stats.json before training")
    if len(pairs) != len(source):
        raise RuntimeError("Every source item must have exactly one DPO pair")
    # Same portable artifacts as the detector project.
    import csv
    from datasets import Dataset
    with (PREFERENCES_DIR / "dpo_preferences.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["item_id", "prompt", "chosen", "rejected", "score_chosen", "score_rejected", "gap"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pairs)
    Dataset.from_list([{k: p[k] for k in ("prompt", "chosen", "rejected")} for p in pairs]).save_to_disk(str(PREFERENCES_DIR / "dpo_hf_dataset"))
    if publish and not limit:
        publish_preferences()
    return PAIRS_PATH


def publish_preferences() -> str:
    """Publish all 500 pairs plus every raw candidate attempt to the existing HF repo."""
    from datasets import Dataset
    from huggingface_hub import HfApi
    from constants import DATASET_ID
    import json

    pairs = read_jsonl(PAIRS_PATH)
    meta = json.loads(PAIRS_META_PATH.read_text(encoding="utf-8"))
    if len(pairs) != 500 or meta["partial"] or len({p["item_id"] for p in pairs}) != 500:
        raise ValueError("HF publication requires one pair for each of the 500 train items")
    from training.dpo_train import validate_pairs
    train, _, source = load_prepared()
    validate_pairs(pairs, meta, train, source)
    config = "dpo-" + SLM_MODEL_ID.split("/")[-1].lower()
    Dataset.from_list(pairs).push_to_hub(DATASET_ID, config_name=config, split="train")
    HfApi().upload_folder(repo_id=DATASET_ID, repo_type="dataset", folder_path=str(PREFERENCES_DIR),
                         path_in_repo=f"artifacts/{config}",
                         allow_patterns=["candidates.jsonl", "dpo_manifest.json", "build_stats.json", "dpo_preferences.csv"])
    url = f"https://huggingface.co/datasets/{DATASET_ID}/viewer/{config}/train"
    write_json(PREFERENCES_DIR / "hf_publication.json", {"repo": DATASET_ID, "config": config, "n_pairs": 500, "url": url})
    print(f"[HF] Published all 500 pairs: {url}", flush=True)
    return url

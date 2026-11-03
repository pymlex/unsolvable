"""Train-only DPO pairs: two small-model rewrites ranked by a frozen solver."""

from __future__ import annotations

import math
from pathlib import Path

from constants import (DPO_CANDIDATES, DPO_MIN_PAIR_GAP, DPO_SEED, GENERATION_BATCH_SIZE, GENERATION_TEMPERATURE, LENGTH_MARGIN,
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
    if len(candidates) < 2:
        raise ValueError(f"Item {row['id']} must have at least two candidates")
    if not all(math.isfinite(c["score"]) for c in candidates):
        raise ValueError(f"Item {row['id']} has an invalid score")
    ranked = sorted(candidates, key=lambda c: c["score"])
    chosen, rejected = ranked[0], ranked[-1]
    gap = rejected["score"] - chosen["score"]
    return {"item_id": str(row["id"]), "prompt": build_item_paraphrase_prompt(row),
            "chosen": chosen["text"], "rejected": rejected["text"],
            "score_chosen": chosen["score"], "score_rejected": rejected["score"], "gap": gap,
            "chosen_quality": chosen["quality"], "rejected_quality": rejected["quality"],
            "is_tie": gap == 0, "is_duplicate": chosen["text"] == rejected["text"],
            "below_probe_margin": gap < tau, "candidate_count": len(candidates)}


def refine_until_margin(source, pools, *, generate, score, gate, persist,
                        minimum_gap=DPO_MIN_PAIR_GAP, batch_size=GENERATION_BATCH_SIZE):
    """Keep every attempt; continue only pending items, with no round limit."""
    import torch
    from transformers import set_seed
    round_id = max((r.get("refinement_round", 0) for r in pools.values()), default=0)
    while True:
        pairs = [select_pair(row, pools[str(row["id"])]["candidates"]) for row in source]
        pending = [row for row, pair in zip(source, pairs, strict=True) if pair["gap"] <= minimum_gap]
        progress = {"n_source": len(source), "n_ready": len(source) - len(pending), "n_pending": len(pending),
                    "round": round_id, "minimum_gap_strictly_greater_than": minimum_gap,
                    "min_gap": min(p["gap"] for p in pairs),
                    "n_candidates": sum(len(r["candidates"]) for r in pools.values())}
        write_json(PREFERENCES_DIR / "refinement_progress.json", progress)
        write_jsonl(PREFERENCES_DIR / "refinement_best_pairs.jsonl", pairs)
        print(f"[refine] round={round_id} ready={progress['n_ready']}/{len(source)} "
              f"pending={len(pending)} min_gap={progress['min_gap']:.6f}", flush=True)
        if not pending:
            return pairs
        round_id += 1
        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            set_seed(DPO_SEED + round_id * 100003 + start)
            try:
                groups = generate(chunk)
            except torch.OutOfMemoryError:
                # An unusually long natural completion can exhaust a batch's VRAM.
                # Keep all completed pools and retry with fresh random samples;
                # never impose a token limit or repeat the same failing seed.
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                for row in chunk:
                    record = pools[str(row["id"])]
                    record["refinement_round"] = round_id
                    record.setdefault("failed_generation_attempts", []).append({"round": round_id, "reason": "cuda_oom"})
                    persist(record)
                print(f"[refine-retry] round={round_id} CUDA OOM; preserved pools, retrying fresh samples", flush=True)
                break
            for row, candidates in zip(chunk, groups, strict=True):
                record = pools[str(row["id"])]
                incomplete = [c for c in candidates if c.get("truncated")]
                if incomplete:
                    record.setdefault("failed_candidates", []).extend(
                        [{**c, "generation_round": round_id, "failure_reason": c.get("failure_reason", "native_context_exhausted")} for c in incomplete])
                candidates = [c for c in candidates if not c.get("truncated")]
                for candidate in candidates:
                    candidate["quality"] = gate.check(row, candidate)
                    candidate["score"] = score(row, candidate["text"])
                    candidate["generation_round"] = round_id
                record["candidates"].extend(candidates)
                record["refinement_round"] = round_id
                persist(record)
            print(f"[refine-batch] round={round_id} processed={min(start+batch_size,len(pending))}/{len(pending)}", flush=True)


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
    # Recover EOS-complete samples even if another row held the previous batch
    # open. Never treat an unfinished completion as a training candidate.
    journal = PREFERENCES_DIR / "generation_samples.jsonl"
    if journal.exists():
        source_by_id = {str(row["id"]): row for row in source}
        recovered = 0
        for sample in read_jsonl(journal):
            row = source_by_id.get(sample["item_id"])
            if row is None or sample.get("truncated"):
                continue
            if sample.get("source_fingerprint") not in (None, fingerprint(row)):
                continue
            key, record = cached_candidate(cache, settings, row)
            if record is None:
                continue
            event_id = f"{sample['run_id']}:{sample['item_id']}:{sample['sample_pass']}"
            if any(c.get("generation_event_id") == event_id for c in record["candidates"]):
                continue
            candidate = {**sample, "generation_event_id": event_id}
            candidate["quality"] = gate.check(row, candidate)
            candidate["score"] = solver.score(row, candidate["text"])
            record["candidates"].append(candidate)
            import json
            with CACHE_PATH.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            recovered += 1
        print(f"[recovery] recovered {recovered} EOS-complete samples from generation journal", flush=True)
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
    pools = {r["item_id"]: r for r in records}
    # Refinement is a continuation of the existing pools, not a reason to discard
    # the original temperature-0.8 candidates and their completed scores.
    import os
    refinement_sampling = {"temperature": float(os.getenv("REFINEMENT_TEMPERATURE", "1.1")),
                           "top_p": float(os.getenv("REFINEMENT_TOP_P", "0.95")),
                           "repetition_penalty": float(os.getenv("REFINEMENT_REPETITION_PENALTY", "1.05"))}
    def generate_refinement(rows):
        groups = generate_item_paraphrases(generator, tokenizer, rows, num_samples=2,
                                          progress_dir=PREFERENCES_DIR, **refinement_sampling)
        for group in groups:
            for candidate in group:
                candidate["sampling"] = refinement_sampling.copy()
        return groups
    def persist(record):
        import json
        import os
        with CACHE_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    pairs = refine_until_margin(
        source, pools,
        generate=generate_refinement,
        score=solver.score, gate=gate, persist=persist,
    )
    records = list(pools.values())
    write_jsonl(PAIRS_PATH, pairs)
    write_json(PAIRS_META_PATH, {"source": manifest, "settings": settings, "tau": PAIRWISE_GAP_TAU,
                               "n_source": len(source), "n_pairs": len(pairs), "partial": bool(limit),
                               "minimum_gap": DPO_MIN_PAIR_GAP, "selection": "cumulative_pool_extremes_strict",
                               "refinement_sampling": refinement_sampling,
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
    from preferences.gap_report import write_gap_report
    write_gap_report(pairs, PREFERENCES_DIR, model_id=SOLVER_MODEL_ID)
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
    commit = HfApi().upload_folder(repo_id=DATASET_ID, repo_type="dataset", folder_path=str(PREFERENCES_DIR),
                         path_in_repo=f"artifacts/{config}",
                         allow_patterns=["candidates.jsonl", "dpo_manifest.json", "build_stats.json", "dpo_preferences.csv", "gap_distribution.json", "gap_histogram.png", "refinement_progress.json", "generation_samples.jsonl"])
    url = f"https://huggingface.co/datasets/{DATASET_ID}/viewer/{config}/train"
    write_json(PREFERENCES_DIR / "hf_publication.json", {"repo": DATASET_ID, "config": config, "n_pairs": 500, "url": url, "commit": commit.oid})
    print(f"[HF] Published all 500 pairs: {url}", flush=True)
    return url

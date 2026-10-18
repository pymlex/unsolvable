#!/usr/bin/env python3
"""Zero-edit tip-2 benchmark for the local SLM (temperature 0)."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm

import _bootstrap  # noqa: F401

from constants import BATCH_SIZE, DATASET_ID, DATASET_SPLIT, MAX_NEW_TOKENS, SLM_MODEL_ID
from eval.data import load_rows
from eval.metrics import (
    answer_jaccard,
    binomial_wilson_ci,
    bootstrap_ci,
    exact_match,
    gold_normalize,
    normalize_answer,
    write_benchmark_table,
)
from eval.model_io import generate_tip2_answers_batch, load_slm
from utils.paths import BENCHMARK_DIR, ensure_result_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark local SLM on tip-2")
    p.add_argument("--dataset", default=DATASET_ID)
    p.add_argument("--split", default=DATASET_SPLIT)
    p.add_argument("--model", default=SLM_MODEL_ID)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def run_benchmark_slm(
    *,
    dataset: str = DATASET_ID,
    split: str = DATASET_SPLIT,
    model_id: str = SLM_MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
    batch_size: int = BATCH_SIZE,
    limit: int = 0,
    device: str | None = None,
    out: str | None = None,
    seed: int = 0,
) -> Path:
    """Run greedy SLM eval on all tip-2 items and write JSON + Markdown."""
    ensure_result_dirs()
    rows = load_rows(dataset, split)
    if limit and limit > 0:
        rows = rows[:limit]

    model, tokenizer, device_name = load_slm(model_id, device)
    bs = max(1, int(batch_size))

    records = []
    correct_flags: list[bool] = []
    latencies: list[float] = []
    jaccards: list[float] = []

    for start in tqdm(range(0, len(rows), bs), desc="benchmark-slm"):
        batch = rows[start : start + bs]
        started = time.perf_counter()
        raws = generate_tip2_answers_batch(
            model, tokenizer, device_name, batch, max_new_tokens=max_new_tokens
        )
        batch_elapsed = time.perf_counter() - started
        per_item = batch_elapsed / max(1, len(batch))
        for row, raw in zip(batch, raws):
            latencies.append(per_item)
            pred = normalize_answer(raw)
            gold = gold_normalize(row["answer"])
            ok = exact_match(raw, row["answer"])
            jac = answer_jaccard(pred, gold)
            correct_flags.append(ok)
            jaccards.append(jac)
            records.append(
                {
                    "id": row["id"],
                    "gold": gold,
                    "pred": pred,
                    "raw": raw,
                    "correct": ok,
                    "jaccard": jac,
                    "latency_sec": per_item,
                }
            )

    n = len(correct_flags)
    successes = int(sum(correct_flags))
    acc, wilson_lo, wilson_hi = binomial_wilson_ci(successes, n)
    _, boot_lo, boot_hi = bootstrap_ci(correct_flags, seed=seed)
    latency_mean = float(sum(latencies) / max(1, len(latencies)))
    mean_j = float(sum(jaccards) / max(1, len(jaccards)))

    summary = {
        "dataset": dataset,
        "split": split,
        "model": model_id,
        "n_items": n,
        "temperature": 0,
        "do_sample": False,
        "protocol": "zero-edit-full-set",
        "accuracy": acc,
        "n_correct": successes,
        "mean_jaccard": mean_j,
        "wilson_95_ci": [wilson_lo, wilson_hi],
        "bootstrap_95_ci": [boot_lo, boot_hi],
        "device": device_name,
        "batch_size": bs,
        "latency_sec_mean": latency_mean,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    slug = model_id.replace("/", "_").replace(" ", "_")
    out_path = Path(out) if out else BENCHMARK_DIR / f"slm_{slug}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "items": records}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_benchmark_table(
        out_path.with_suffix(".md"),
        model_id,
        n,
        acc,
        (wilson_lo, wilson_hi),
        (boot_lo, boot_hi),
        mean_jaccard=mean_j,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return out_path


def main() -> None:
    args = parse_args()
    run_benchmark_slm(
        dataset=args.dataset,
        split=args.split,
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        limit=args.limit,
        device=args.device,
        out=args.out,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

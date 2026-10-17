#!/usr/bin/env python3
"""Zero-edit tip-2 benchmark via Zveno DeepSeek (temperature 0)."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm

import _bootstrap  # noqa: F401

from constants import DATASET_ID, DATASET_SPLIT, MAX_NEW_TOKENS, ZVENO_MODEL_ID
from eval.data import load_rows
from eval.metrics import (
    binomial_wilson_ci,
    bootstrap_ci,
    exact_match,
    gold_normalize,
    normalize_answer,
    write_benchmark_table,
)
from eval.prompt import build_messages
from eval.zveno_client import chat_completion, completion_text
from utils.paths import BENCHMARK_DIR, ensure_result_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark DeepSeek via Zveno on tip-2")
    p.add_argument("--dataset", default=DATASET_ID)
    p.add_argument("--split", default=DATASET_SPLIT)
    p.add_argument("--model", default=ZVENO_MODEL_ID)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Reasoning models need headroom so the final digit answer is emitted",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sleep", type=float, default=0.15)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def run_benchmark_deepseek(
    *,
    dataset: str = DATASET_ID,
    split: str = DATASET_SPLIT,
    model_id: str = ZVENO_MODEL_ID,
    max_tokens: int = 1024,
    limit: int = 0,
    sleep: float = 0.15,
    out: str | None = None,
    seed: int = 0,
) -> Path:
    """Run API tip-2 eval and write JSON + Markdown under results/benchmark/."""
    ensure_result_dirs()
    rows = load_rows(dataset, split)
    if limit and limit > 0:
        rows = rows[:limit]

    records = []
    correct_flags: list[bool] = []
    latencies: list[float] = []
    prompt_tokens_total = 0
    completion_tokens_total = 0

    for row in tqdm(rows, desc="benchmark-deepseek"):
        body = chat_completion(
            build_messages(row),
            model=model_id,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        raw = completion_text(body)
        latency = float(body.get("_latency_sec") or 0.0)
        latencies.append(latency)
        usage = body.get("usage") or {}
        prompt_tokens_total += int(usage.get("prompt_tokens") or 0)
        completion_tokens_total += int(usage.get("completion_tokens") or 0)
        pred = normalize_answer(raw)
        gold = gold_normalize(row["answer"])
        ok = exact_match(raw, row["answer"])
        correct_flags.append(ok)
        records.append(
            {
                "id": row["id"],
                "gold": gold,
                "pred": pred,
                "raw": raw,
                "correct": ok,
                "latency_sec": latency,
                "usage": usage,
            }
        )
        if sleep > 0:
            time.sleep(sleep)

    n = len(correct_flags)
    successes = int(sum(correct_flags))
    acc, wilson_lo, wilson_hi = binomial_wilson_ci(successes, n)
    _, boot_lo, boot_hi = bootstrap_ci(correct_flags, seed=seed)
    latency_mean = float(sum(latencies) / max(1, len(latencies)))

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
        "wilson_95_ci": [wilson_lo, wilson_hi],
        "bootstrap_95_ci": [boot_lo, boot_hi],
        "device": "zveno-api",
        "latency_sec_mean": latency_mean,
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    slug = model_id.split("/")[-1].replace(" ", "_")
    out_path = Path(out) if out else BENCHMARK_DIR / f"deepseek_{slug}.json"
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
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return out_path


def main() -> None:
    args = parse_args()
    run_benchmark_deepseek(
        dataset=args.dataset,
        split=args.split,
        model_id=args.model,
        max_tokens=args.max_tokens,
        limit=args.limit,
        sleep=args.sleep,
        out=args.out,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

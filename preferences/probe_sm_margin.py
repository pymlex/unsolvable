#!/usr/bin/env python3
"""Histogram of \(s_m\) and pairwise gaps for δ/τ calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import _bootstrap  # noqa: F401

from constants import (
    ANALYZE_MARGIN_SAMPLES,
    DATASET_ID,
    DATASET_SPLIT,
    LENGTH_MARGIN,
    PARAPHRASES_PER_TEXT,
    SLM_MODEL_ID,
)
from eval.data import load_rows
from eval.model_io import load_slm
from eval.prompt import item_text
from generation.paraphrase import generate_paraphrases
from scoring.answer_logprob import answer_token_logprob_sum
from utils.paths import PLOTS_DIR, PREFERENCES_DIR, ensure_result_dirs


def _within_length(original: str, candidate: str, margin: float = LENGTH_MARGIN) -> bool:
    o = max(1, len(original.split()))
    c = len(candidate.split())
    return abs(c - o) / o <= margin


def run_probe(
    *,
    dataset: str = DATASET_ID,
    split: str = DATASET_SPLIT,
    model_id: str = SLM_MODEL_ID,
    n_samples: int = ANALYZE_MARGIN_SAMPLES,
    k: int = PARAPHRASES_PER_TEXT,
    device: str | None = None,
) -> Path:
    """Sample paraphrases, score \(s_m\), and write histograms for calibration."""
    ensure_result_dirs()
    rows = load_rows(dataset, split)[:n_samples]
    model, tokenizer, device_name = load_slm(model_id, device)
    texts = [item_text(row) for row in rows]

    paraphrase_groups = generate_paraphrases(
        model=model,
        tokenizer=tokenizer,
        original_texts=texts,
        device=device_name,
        num_samples=k,
    )

    score_rows: list[dict] = []
    all_scores: list[float] = []
    pairwise_gaps: list[float] = []

    for row, paraphrases in tqdm(
        list(zip(rows, paraphrase_groups, strict=True)),
        desc="score paraphrases",
    ):
        base = dict(row)
        scores: list[float] = []
        kept: list[str] = []
        for paraphrase in paraphrases:
            if not paraphrase.strip():
                continue
            if not _within_length(item_text(row), paraphrase):
                continue
            candidate = dict(base)
            candidate["text"] = paraphrase
            candidate["passage"] = paraphrase
            score = answer_token_logprob_sum(model, tokenizer, candidate)
            scores.append(score)
            kept.append(paraphrase)
            all_scores.append(score)
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                pairwise_gaps.append(abs(scores[i] - scores[j]))
        score_rows.append(
            {
                "id": row["id"],
                "n_kept": len(kept),
                "scores": scores,
                "paraphrases": kept,
            }
        )

    out_json = PREFERENCES_DIR / "sm_margin_probe.json"
    out_json.write_text(
        json.dumps(
            {
                "n_items": len(rows),
                "k": k,
                "model": model_id,
                "score_mean": float(np.mean(all_scores)) if all_scores else None,
                "gap_percentiles": {
                    str(p): float(np.quantile(pairwise_gaps, p))
                    for p in (0.25, 0.5, 0.75, 0.9, 0.95)
                }
                if pairwise_gaps
                else {},
                "items": score_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if all_scores:
        axes[0].hist(np.asarray(all_scores, dtype=np.float64), bins=30)
    axes[0].set_title(r"$s_m(y^\star\mid x)$ histogram")
    axes[0].set_xlabel(r"$s_m$")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.5)
    if pairwise_gaps:
        axes[1].hist(np.asarray(pairwise_gaps, dtype=np.float64), bins=30)
    axes[1].set_title("Pairwise |gap| histogram")
    axes[1].set_xlabel("|gap|")
    axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.5)
    fig.tight_layout()
    plot_path = PLOTS_DIR / "sm_margin_probe_hist.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_json}")
    print(f"Saved {plot_path}")
    return out_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DATASET_ID)
    p.add_argument("--split", default=DATASET_SPLIT)
    p.add_argument("--model", default=SLM_MODEL_ID)
    p.add_argument("--n-samples", type=int, default=ANALYZE_MARGIN_SAMPLES)
    p.add_argument("--k", type=int, default=PARAPHRASES_PER_TEXT)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    run_probe(
        dataset=args.dataset,
        split=args.split,
        model_id=args.model,
        n_samples=args.n_samples,
        k=args.k,
        device=args.device,
    )


if __name__ == "__main__":
    main()

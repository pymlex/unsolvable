#!/usr/bin/env python3
"""Build GRPO groups and optional pairwise DPO export from paraphrases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

import _bootstrap  # noqa: F401

from constants import (
    DATASET_ID,
    DATASET_SPLIT,
    GRPO_GROUP_SIZE,
    LENGTH_MARGIN,
    PAIRWISE_GAP_TAU,
    PARAPHRASES_PER_TEXT,
    SLM_MODEL_ID,
)
from eval.data import load_rows
from eval.model_io import load_slm
from eval.prompt import item_text
from generation.paraphrase import generate_paraphrases
from generation.prompts import build_paraphrase_prompt
from judge.zveno_judge import judge_paraphrase
from schemas.models import GrpoGroup, PreferencePair
from scoring.answer_logprob import answer_token_logprob_sum
from utils.paths import GROUPS_DIR, PREFERENCES_DIR, ensure_result_dirs


def _within_length(original: str, candidate: str, margin: float = LENGTH_MARGIN) -> bool:
    o = max(1, len(original.split()))
    c = len(candidate.split())
    return abs(c - o) / o <= margin


def build_groups(
    *,
    dataset: str = DATASET_ID,
    split: str = DATASET_SPLIT,
    model_id: str = SLM_MODEL_ID,
    k: int = PARAPHRASES_PER_TEXT,
    group_size: int = GRPO_GROUP_SIZE,
    limit: int = 0,
    skip_judge: bool = False,
    device: str | None = None,
    export_dpo: bool = True,
) -> Path:
    """For each item sample K paraphrases, filter, score, and emit GRPO groups.

    Reward for GRPO is ``-s_m`` so lower answer likelihood under the exam prompt
    is preferred for adversarial hardening of the text generator.
    """
    ensure_result_dirs()
    rows = load_rows(dataset, split)
    if limit and limit > 0:
        rows = rows[:limit]

    model, tokenizer, device_name = load_slm(model_id, device)
    texts = [item_text(row) for row in rows]
    paraphrase_groups = generate_paraphrases(
        model=model,
        tokenizer=tokenizer,
        original_texts=texts,
        device=device_name,
        num_samples=k,
    )

    groups: list[GrpoGroup] = []
    pairs: list[PreferencePair] = []
    stats = {
        "items": len(rows),
        "kept_groups": 0,
        "skipped_short": 0,
        "skipped_judge": 0,
        "dpo_pairs": 0,
    }

    for row, paraphrases in tqdm(
        list(zip(rows, paraphrase_groups, strict=True)),
        desc="build-groups",
    ):
        original = item_text(row)
        prompt = build_paraphrase_prompt(original)
        kept_texts: list[str] = []
        kept_scores: list[float] = []
        for paraphrase in paraphrases:
            if not paraphrase.strip():
                continue
            if not _within_length(original, paraphrase):
                continue
            if not skip_judge:
                verdict = judge_paraphrase(
                    original,
                    paraphrase,
                    str(row["answer"]),
                    str(row.get("explanation", "")),
                )
                if verdict.broken:
                    stats["skipped_judge"] += 1
                    continue
            candidate = dict(row)
            candidate["text"] = paraphrase
            candidate["passage"] = paraphrase
            score = answer_token_logprob_sum(model, tokenizer, candidate)
            kept_texts.append(paraphrase)
            kept_scores.append(score)

        if len(kept_texts) < min(2, group_size):
            stats["skipped_short"] += 1
            continue

        kept_texts = kept_texts[:group_size]
        kept_scores = kept_scores[:group_size]
        rewards = [-s for s in kept_scores]
        groups.append(
            GrpoGroup(
                item_id=str(row["id"]),
                prompt=prompt,
                completions=kept_texts,
                scores=kept_scores,
                rewards=rewards,
                gold_answer=str(row["answer"]),
                explanation=str(row.get("explanation", "")),
            )
        )
        stats["kept_groups"] += 1

        if export_dpo and len(kept_texts) >= 2:
            order = sorted(range(len(kept_scores)), key=lambda i: kept_scores[i])
            best_i, worst_i = order[0], order[-1]
            gap = abs(kept_scores[best_i] - kept_scores[worst_i])
            if gap >= PAIRWISE_GAP_TAU:
                pairs.append(
                    PreferencePair(
                        item_id=str(row["id"]),
                        prompt=prompt,
                        chosen=kept_texts[best_i],
                        rejected=kept_texts[worst_i],
                        score_chosen=kept_scores[best_i],
                        score_rejected=kept_scores[worst_i],
                        gap=gap,
                    )
                )
                stats["dpo_pairs"] += 1

    groups_path = GROUPS_DIR / "grpo_groups.jsonl"
    with groups_path.open("w", encoding="utf-8") as handle:
        for group in groups:
            handle.write(group.model_dump_json() + "\n")

    if export_dpo:
        dpo_path = PREFERENCES_DIR / "dpo_preferences.jsonl"
        with dpo_path.open("w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(pair.model_dump_json() + "\n")

    stats_path = GROUPS_DIR / "build_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Saved {groups_path}")
    return groups_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=DATASET_ID)
    p.add_argument("--split", default=DATASET_SPLIT)
    p.add_argument("--model", default=SLM_MODEL_ID)
    p.add_argument("--k", type=int, default=PARAPHRASES_PER_TEXT)
    p.add_argument("--group-size", type=int, default=GRPO_GROUP_SIZE)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--no-dpo", action="store_true")
    args = p.parse_args()
    build_groups(
        dataset=args.dataset,
        split=args.split,
        model_id=args.model,
        k=args.k,
        group_size=args.group_size,
        limit=args.limit,
        skip_judge=args.skip_judge,
        device=args.device,
        export_dpo=not args.no_dpo,
    )


if __name__ == "__main__":
    main()

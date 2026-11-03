#!/usr/bin/env python3
"""CLI entry for the unsolvable tip-2 adversarial pipeline."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from utils.paths import ensure_result_dirs


STEPS = [
    "prepare", "preferences", "train", "evaluate",
    "benchmark-slm",
    "benchmark-deepseek",
    "probe-margin",
    "build-groups",
    "grpo-train",
    "synthesis",
    "all",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unsolvable: train-only DPO paraphrasing and full Reshu EGE evaluation"
    )
    parser.add_argument("--step", choices=STEPS, default="prepare")
    parser.add_argument("--limit", type=int, default=0, help="Optional item cap for smoke runs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="SLM generate batch size (default BATCH_SIZE=8 from env)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate DPO pairs/settings without loading models")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--n-synth", type=int, default=None)
    parser.add_argument("--resume", nargs="?", const=True, default=None, help="Resume latest DPO checkpoint, or specify its path")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--checkpoint", default=None, help="Paraphraser checkpoint for standalone evaluation")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    ensure_result_dirs()
    args = parse_args()
    limit = args.limit

    if args.step in {"train", "evaluate", "all"} and limit:
        raise ValueError("--limit is only for preference smoke runs; DPO evaluation always uses all 116 items")
    if args.step == "all" and args.dry_run:
        raise ValueError("Use --step train --dry-run after preparing preferences")
    if args.step in {"prepare", "all"}:
        from eval.ege_data import prepare_data
        prepare_data()
    if args.step in {"preferences", "all"}:
        from preferences.build_dpo_dataset import build_preference_dataset
        build_preference_dataset(limit=limit, device=args.device)
    if args.step in {"train", "all"}:
        from training.dpo_train import train_dpo
        train_dpo(dry_run=args.dry_run, resume=args.resume, max_steps=args.max_steps, device=args.device)
    if args.step == "evaluate":
        from eval.evaluate_paraphraser import run_evaluation
        run_evaluation(checkpoint=args.checkpoint, device=args.device)

    if args.step == "benchmark-slm":
        from eval.benchmark_slm import run_benchmark_slm

        kwargs = {"limit": limit}
        if args.batch_size is not None:
            kwargs["batch_size"] = args.batch_size
        run_benchmark_slm(**kwargs)

    if args.step == "benchmark-deepseek":
        from eval.benchmark_deepseek import run_benchmark_deepseek

        run_benchmark_deepseek(limit=limit)

    if args.step == "probe-margin":
        from preferences.probe_sm_margin import run_probe

        n_samples = limit if limit > 0 else None
        kwargs = {}
        if n_samples is not None:
            kwargs["n_samples"] = n_samples
        run_probe(**kwargs)

    if args.step == "build-groups":
        from preferences.build_groups import build_groups

        build_groups(limit=limit, skip_judge=args.skip_judge)

    if args.step == "grpo-train":
        from training.grpo_train import run_grpo

        run_grpo(dry_run=args.dry_run or args.step == "all")

    if args.step == "synthesis":
        from synthesis.generate import generate_synthetic_split

        kwargs = {}
        if args.n_synth is not None:
            kwargs["n_items"] = args.n_synth
        elif limit > 0:
            kwargs["n_items"] = limit
        generate_synthetic_split(**kwargs)


if __name__ == "__main__":
    main()

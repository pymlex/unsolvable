#!/usr/bin/env python3
"""CLI entry for the unsolvable tip-2 adversarial pipeline."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from utils.paths import ensure_result_dirs


STEPS = [
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
        description="Unsolvable: tip-2 benchmark, paraphrase groups, GRPO scaffold"
    )
    parser.add_argument("--step", choices=STEPS, default="benchmark-slm")
    parser.add_argument("--limit", type=int, default=0, help="Optional item cap for smoke runs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="SLM generate batch size (default BATCH_SIZE=8 from env)",
    )
    parser.add_argument("--dry-run", action="store_true", help="GRPO dry-run only")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--n-synth", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    ensure_result_dirs()
    args = parse_args()
    limit = args.limit

    if args.step in {"benchmark-slm", "all"}:
        from eval.benchmark_slm import run_benchmark_slm

        kwargs = {"limit": limit}
        if args.batch_size is not None:
            kwargs["batch_size"] = args.batch_size
        run_benchmark_slm(**kwargs)

    if args.step in {"benchmark-deepseek", "all"}:
        from eval.benchmark_deepseek import run_benchmark_deepseek

        run_benchmark_deepseek(limit=limit)

    if args.step in {"probe-margin", "all"}:
        from preferences.probe_sm_margin import run_probe

        n_samples = limit if limit > 0 else None
        kwargs = {}
        if n_samples is not None:
            kwargs["n_samples"] = n_samples
        run_probe(**kwargs)

    if args.step in {"build-groups", "all"}:
        from preferences.build_groups import build_groups

        build_groups(limit=limit, skip_judge=args.skip_judge)

    if args.step in {"grpo-train", "all"}:
        from training.grpo_train import run_grpo

        run_grpo(dry_run=args.dry_run or args.step == "all")

    if args.step in {"synthesis", "all"}:
        from synthesis.generate import generate_synthetic_split

        kwargs = {}
        if args.n_synth is not None:
            kwargs["n_items"] = args.n_synth
        elif limit > 0:
            kwargs["n_items"] = limit
        generate_synthetic_split(**kwargs)


if __name__ == "__main__":
    main()

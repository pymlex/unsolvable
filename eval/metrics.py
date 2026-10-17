"""Exact-match scoring and confidence intervals for tip-2 answers."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from scipy import stats


ANSWER_DIGITS_RE = re.compile(r"[0-9]+")


def normalize_answer(text: str) -> str:
    """Extract the first contiguous digit run; empty if none."""
    if text is None:
        return ""
    m = ANSWER_DIGITS_RE.search(str(text))
    if not m:
        return ""
    digits = m.group(0)
    seen = set()
    ordered: list[str] = []
    for ch in digits:
        if ch not in seen and ch in "12345":
            seen.add(ch)
            ordered.append(ch)
    return "".join(sorted(ordered))


def gold_normalize(answer: str) -> str:
    digits = "".join(ch for ch in str(answer) if ch.isdigit())
    return "".join(sorted(set(digits)))


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == gold_normalize(gold) and bool(gold_normalize(gold))


def binomial_wilson_ci(
    successes: int,
    n: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    z = stats.norm.ppf(1 - alpha / 2)
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    lo = float(max(0.0, centre - half))
    hi = float(min(1.0, centre + half))
    return float(p), lo, hi


def bootstrap_ci(
    correct: Iterable[bool],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    arr = np.asarray(list(correct), dtype=np.float64)
    n = arr.size
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    p = float(arr.mean())
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return p, lo, hi


def write_benchmark_table(
    path,
    model: str,
    n: int,
    acc: float,
    wilson: tuple[float, float],
    bootstrap: tuple[float, float],
) -> None:
    """Write a one-row Markdown accuracy table."""
    path = path if hasattr(path, "write_text") else path
    from pathlib import Path

    out = Path(path)
    out.write_text(
        (
            f"| model | n | accuracy | Wilson 95% CI | bootstrap 95% CI |\n"
            f"|---|---:|---:|---|---|\n"
            f"| {model} | {n} | {acc:.4f} | "
            f"[{wilson[0]:.4f}, {wilson[1]:.4f}] | "
            f"[{bootstrap[0]:.4f}, {bootstrap[1]:.4f}] |\n"
        ),
        encoding="utf-8",
    )

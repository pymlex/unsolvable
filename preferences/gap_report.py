"""Export the full score-gap distribution without filtering any preference rows."""

from __future__ import annotations

from pathlib import Path

from utils.artifacts import write_json


def write_gap_report(pairs: list[dict], directory: Path, *, model_id: str) -> dict:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gaps = np.asarray([p["gap"] for p in pairs], dtype=np.float64)
    if not len(gaps):
        raise ValueError("Cannot plot an empty preference dataset")
    stats = {"n_pairs": len(pairs), "model": model_id, "units": "nats",
             "score": "sum log P(gold digit answer tokens + EOS | exam prompt)",
             "gap": "abs(score_1 - score_2)", "gaps": gaps.tolist(),
             "mean": float(gaps.mean()), "median": float(np.median(gaps)),
             "min": float(gaps.min()), "max": float(gaps.max()),
             "p25": float(np.quantile(gaps, .25)), "p75": float(np.quantile(gaps, .75)),
             "p90": float(np.quantile(gaps, .90)), "n_zero": int((gaps == 0).sum()),
             "n_duplicates": sum(p.get("is_duplicate", False) for p in pairs)}
    write_json(directory / "gap_distribution.json", stats)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.hist(gaps, bins=40, color="#3971a8", edgecolor="white", linewidth=.6)
    ax.axvline(stats["median"], color="#a44435", linewidth=1.5,
               label=f"Медиана: {stats['median']:.3f} нат")
    ax.set(title=f"Разница оценок в DPO-парах — все {len(pairs)} пар",
           xlabel="Δ = |score₁ − score₂|, нат", ylabel="Количество пар")
    ax.legend(frameon=False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(.1, .025, f"{model_id}, замороженный решатель, без thinking. "
             f"Train DeepSeek; минимум Δ = {stats['min']:.3f} нат; нулевых: {stats['n_zero']}.", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(directory / "gap_histogram.png", dpi=160)
    plt.close(fig)
    return stats

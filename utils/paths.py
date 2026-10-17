from __future__ import annotations

from pathlib import Path

from utils.config_loader import env_str


RESULTS_DIR = Path(env_str("RESULTS_DIR", "results"))
BENCHMARK_DIR = RESULTS_DIR / "benchmark"
PREFERENCES_DIR = RESULTS_DIR / "preferences"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_DIR = RESULTS_DIR / "metrics"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
MONITORING_DIR = RESULTS_DIR / "monitoring"
SYNTHESIS_DIR = RESULTS_DIR / "synthesis"
GROUPS_DIR = RESULTS_DIR / "groups"
HF_CACHE_DIR = Path(env_str("HF_CACHE_DIR", "hf_cache"))
DATA_DIR = Path(env_str("DATA_DIR", "data"))


def ensure_result_dirs() -> None:
    """Create standard artefact directories under ``RESULTS_DIR``."""
    for path in (
        RESULTS_DIR,
        BENCHMARK_DIR,
        PREFERENCES_DIR,
        PLOTS_DIR,
        METRICS_DIR,
        CHECKPOINTS_DIR,
        MONITORING_DIR,
        SYNTHESIS_DIR,
        GROUPS_DIR,
        HF_CACHE_DIR,
        DATA_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

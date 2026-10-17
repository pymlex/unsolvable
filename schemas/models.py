from __future__ import annotations

from pydantic import BaseModel, Field


class BenchmarkSummary(BaseModel):
    dataset: str
    split: str
    model: str
    n_items: int
    temperature: float = 0.0
    do_sample: bool = False
    protocol: str = "zero-edit-full-set"
    accuracy: float
    n_correct: int
    wilson_95_ci: list[float]
    bootstrap_95_ci: list[float]
    device: str | None = None
    latency_sec_mean: float | None = None
    cost_usd_estimate: float | None = None
    created_at_utc: str


class JudgeVerdict(BaseModel):
    broken: bool
    reason: str = ""
    raw: str = ""


class PreferencePair(BaseModel):
    item_id: str
    prompt: str
    chosen: str
    rejected: str
    score_chosen: float
    score_rejected: float
    gap: float


class GrpoGroup(BaseModel):
    item_id: str
    prompt: str
    completions: list[str]
    scores: list[float]
    rewards: list[float]
    gold_answer: str
    explanation: str = ""


class SyntheticItem(BaseModel):
    id: str
    text: str
    options: list[str]
    answer: str
    answer_list: list[int] = Field(default_factory=list)
    explanation: str = ""
    lemmas: list[str] = Field(default_factory=list)
    senses: list[str] = Field(default_factory=list)
    source_model: str = ""

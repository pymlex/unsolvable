"""Use an immutable exam solver, including adapter-disabled base-model reuse."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext

from constants import BATCH_SIZE
from utils.paths import MONITORING_DIR
from eval.model_io import generate_tip2_answers_batch
from scoring.answer_logprob import answer_token_logprob_sum


class FrozenSolver:
    def __init__(self, model, tokenizer, *, model_id: str, disable_adapter=False):
        if disable_adapter and not hasattr(model, "disable_adapter"):
            raise ValueError("Shared trainable solver requires PEFT disable_adapter()")
        self.model, self.tokenizer = model, tokenizer
        self.model_id, self.disable_adapter = model_id, disable_adapter

    @contextmanager
    def inference(self):
        was_training = self.model.training
        self.model.eval()
        try:
            with self.model.disable_adapter() if self.disable_adapter else nullcontext():
                yield
        finally:
            self.model.train(was_training)

    def score(self, row: dict, text: str) -> float:
        candidate = {**row, "text": text, "passage": text}
        with self.inference():
            return answer_token_logprob_sum(self.model, self.tokenizer, candidate)

    def solve(self, rows: list[dict]) -> list[str]:
        with self.inference():
            outputs = []
            for start in range(0, len(rows), BATCH_SIZE):
                outputs.extend(generate_tip2_answers_batch(
                    self.model, self.tokenizer, str(self.model.device), rows[start:start + BATCH_SIZE],
                    max_new_tokens=None,
                    progress_dir=MONITORING_DIR / "solver_generation",
                ))
                print(f"[solver] completed={min(start+BATCH_SIZE,len(rows))}/{len(rows)}", flush=True)
            return outputs

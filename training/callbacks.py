"""Full 116-item exam monitoring before, during and after DPO."""

from __future__ import annotations

from transformers import TrainerCallback


class EgeMonitorCallback(TrainerCallback):
    def __init__(self, evaluator, *, every_steps: int):
        if every_steps < 1:
            raise ValueError("DPO_MONITOR_EVERY_STEPS must be positive")
        self.evaluator, self.every_steps = evaluator, every_steps
        self.last_step = None

    def _run(self, model, state):
        step = int(state.global_step)
        if step != self.last_step:
            self.evaluator.run(model, step=step, epoch=float(state.epoch or 0.0))
            self.last_step = step

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._run(model, state)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every_steps == 0:
            self._run(model, state)
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._run(model, state)
        return control

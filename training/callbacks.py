"""Full 116-item exam monitoring before, during and after DPO."""

from __future__ import annotations

from transformers import TrainerCallback
import math


class EgeMonitorCallback(TrainerCallback):
    def __init__(self, evaluator, *, every_steps: int, every_epochs=None,
                 loss_evaluator=None, reporter=None):
        if every_steps < 1:
            raise ValueError("DPO_MONITOR_EVERY_STEPS must be positive")
        self.evaluator, self.every_steps = evaluator, every_steps
        self.last_step = None
        if every_epochs is not None and every_epochs <= 0:
            raise ValueError("Epoch monitoring interval must be positive")
        self.every_epochs = every_epochs
        self.next_epoch = None
        self.loss_evaluator, self.reporter = loss_evaluator, reporter
        self.pending = False

    def _run(self, model, state):
        step = int(state.global_step)
        if step != self.last_step:
            was_training = model.training if hasattr(model, "training") else None
            payload = self.evaluator.run(model, step=step, epoch=float(state.epoch or 0.0))
            train_loss = self.loss_evaluator() if self.loss_evaluator else None
            if train_loss is not None and hasattr(self.evaluator, "attach_train_loss"):
                self.evaluator.attach_train_loss(payload, train_loss)
            if self.reporter:
                self.reporter.record(step, float(state.epoch or 0.0), train_loss, self.evaluator.history)
            if was_training is not None:
                model.train(was_training)
            self.last_step = step
        if self.every_epochs:
            self.next_epoch = (math.floor(float(state.epoch or 0) / self.every_epochs + 1e-7) + 1) * self.every_epochs

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._run(model, state)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.every_epochs:
            if float(state.epoch or 0) >= self.next_epoch - 1e-7:
                self.pending = True  # Save this step's loss before lengthy evaluation.
        elif state.global_step % self.every_steps == 0:
            self._run(model, state)
        return control

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self.pending and "loss" in (logs or {}):
            self.pending = False
            self._run(model, state)
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._run(model, state)
        return control

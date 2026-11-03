"""Stop on a measured plateau of non-overlapping mean training-loss windows."""

import math
from pathlib import Path
from transformers import TrainerCallback
from utils.artifacts import read_jsonl, write_json


class LossConvergenceCallback(TrainerCallback):
    def __init__(self, directory: Path, *, window=20, patience=3, min_steps=80,
                 min_delta=.001, resume=False):
        self.directory = directory
        self.window, self.patience = window, patience
        self.min_steps, self.min_delta = min_steps, min_delta
        self.losses = {}
        self.windows = []
        self.best = math.inf
        self.stale = 0
        self.converged = False
        if resume and (directory / "train_history.jsonl").exists():
            for row in read_jsonl(directory / "train_history.jsonl"):
                if isinstance(row.get("loss"), (int, float)) and math.isfinite(row["loss"]):
                    self._observe(int(row["step"]), row["loss"])

    def _observe(self, step, loss):
        self.losses[step] = loss
        if step % self.window or any(w["step"] == step for w in self.windows):
            return
        values = [self.losses[s] for s in range(step-self.window+1, step+1) if s in self.losses]
        if len(values) != self.window:
            return
        mean = sum(values) / len(values)
        if mean < self.best - self.min_delta:
            self.best, self.stale = mean, 0
        else:
            self.stale += 1
        self.windows.append({"step": step, "mean_loss": mean, "stale_windows": self.stale})
        self.converged = step >= self.min_steps and self.stale >= self.patience

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if not isinstance(loss, (int, float)) or not math.isfinite(loss):
            return control
        self._observe(int(state.global_step), float(loss))
        write_json(self.directory / "convergence.json", {
            "converged": self.converged, "criterion": "training_loss_plateau",
            "window_steps": self.window, "patience_windows": self.patience,
            "min_steps": self.min_steps, "min_delta": self.min_delta,
            "step": int(state.global_step), "windows": self.windows,
            "note": "Loss convergence does not imply improved held-out attack success; inspect 116-item eval metrics.",
        })
        if self.converged:
            print(f"[convergence] loss plateau at step {state.global_step}; stopping training", flush=True)
            control.should_training_stop = True
        return control

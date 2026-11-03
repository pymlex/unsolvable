"""Immutable epoch reports, logarithmic loss plots and asynchronous HF publication."""

from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
from pathlib import Path
import shutil
from datetime import datetime, timezone

from constants import DATASET_ID
from utils.artifacts import read_jsonl, write_json


class EpochReporter:
    def __init__(self, directory: Path, plan: dict):
        self.directory, self.plan = directory, plan
        self.run_id = plan["live_run_id"]
        self.prefix = f"artifacts/dpo-qwen3.5-2b/training/{self.run_id}"
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending, self.published = {}, {}
        self.status, self.model_url = "training", None

    def record(self, step, epoch, loss, validation):
        target = self.directory / "live" / f"step-{step:06d}"
        target.mkdir(parents=True, exist_ok=True)
        logs = read_jsonl(self.directory / "train_history.jsonl")
        plot_epoch_report(logs, validation, target)
        event = {"run_id": self.run_id, "step": step, "epoch": epoch,
                 "created_at_utc": datetime.now(timezone.utc).isoformat(),
                 "dpo_train_loss": loss, "n_train_loss_pairs": self.plan["pairs"],
                 "eval": validation[-1], "path": f"{self.prefix}/snapshots/step-{step:06d}"}
        write_json(target / "snapshot.json", event)
        write_json(target / "run_config.json", self.plan)
        for name in ("train_history.jsonl", "validation_history.json", f"monitor_step_{step:06d}.json"):
            source = self.directory / name
            if source.exists():
                shutil.copy2(source, target / name)
        print(f"[epoch-report] epoch={epoch:.3f} step={step} DPO_loss_500={loss:.6f}", flush=True)
        self.executor.submit(self._publish, target, event)

    def _publish(self, target=None, event=None):
        from huggingface_hub import HfApi
        if event:
            self.pending[event["step"]] = (target, event)
        try:
            api = HfApi()
            for step, (folder, record) in list(self.pending.items()):
                api.upload_folder(repo_id=DATASET_ID, repo_type="dataset", folder_path=str(folder),
                                  path_in_repo=record["path"], commit_message=f"DPO epoch {record['epoch']:.3f}: loss and full 116-item eval")
                self.published[step] = record
                del self.pending[step]
            manifest = {"run_id": self.run_id, "status": self.status, "model_url": self.model_url,
                        "events": [self.published[k] for k in sorted(self.published)]}
            content = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
            api.upload_file(repo_id=DATASET_ID, repo_type="dataset", path_or_fileobj=io.BytesIO(content),
                            path_in_repo=f"{self.prefix}/live_manifest.json", commit_message="Update live DPO training reports")
            pointer = {"run_id": self.run_id, "manifest": f"{self.prefix}/live_manifest.json"}
            api.upload_file(repo_id=DATASET_ID, repo_type="dataset", path_or_fileobj=io.BytesIO(json.dumps(pointer).encode()),
                            path_in_repo="artifacts/dpo-qwen3.5-2b/training/latest.json", commit_message="Update current DPO run")
            write_json(self.directory / "live_publication.json", manifest)
        except Exception as exc:
            # Network publication must not interrupt optimizer work. The next
            # report retries every pending immutable snapshot.
            write_json(self.directory / "live_publication_error.json", {"error": str(exc), "pending_steps": list(self.pending)})
            print(f"[HF-live] publication pending: {exc}", flush=True)

    def finish(self, *, converged=False, model_url=None):
        self.executor.shutdown(wait=True)
        self.status = "converged" if converged else "budget_exhausted"
        self.model_url = model_url
        self._publish()


def plot_epoch_report(logs, validation, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latest = {r["step"]: r for r in logs if isinstance(r.get("loss"), (int, float))}
    raw = [latest[k] for k in sorted(latest) if math.isfinite(latest[k]["loss"]) and latest[k]["loss"] > 0]
    probes = [r for r in validation if r.get("dpo_train_loss", 0) > 0]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot([r["epoch"] for r in raw], [r["loss"] for r in raw], alpha=.5, linewidth=1,
            label="Обучение: loss каждого шага")
    if probes:
        ax.plot([r["epoch"] for r in probes], [r["dpo_train_loss"] for r in probes], "o-",
                color="#b54e43", label="Оценка DPO-loss на всех 500 train-парах")
    ax.set_yscale("log")
    ax.set(title="DPO loss — сохранённые измерения", xlabel="Эпоха", ylabel="Loss — логарифмическая шкала")
    ax.grid(alpha=.2, which="both"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(target / "loss.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    epochs = [r["epoch"] for r in validation]
    axes[0].plot(epochs, [r["baseline_accuracy"] * 100 for r in validation], "--", label="Исходные задания")
    axes[0].plot(epochs, [r["accuracy"] * 100 for r in validation], "o-", label="После перефразирования")
    axes[0].set(title="Точность замороженного решателя", ylabel="Точность, %")
    axes[1].plot(epochs, [r["valid_rate"] * 100 for r in validation], "o-", label="Структурно допустимые")
    axes[1].plot(epochs, [100 * r["conditional_asr"] if r["conditional_asr"] is not None else float("nan")
                          for r in validation], "o-", label="Успешные атаки (ASR)")
    axes[1].set(title="Формат перефразов и атаки", ylabel="Доля, %")
    for ax in axes:
        ax.set_xlabel("Эпоха"); ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.suptitle("Eval: все 116 настоящих Reshu EGE; недопустимый перефраз → исходный текст", fontsize=10)
    fig.tight_layout(); fig.savefig(target / "eval.png", dpi=150); plt.close(fig)

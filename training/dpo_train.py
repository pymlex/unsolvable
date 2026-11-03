"""LoRA DPO of the paraphraser, with a frozen no-thinking exam scorer."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from constants import (DPO_BETA, DPO_EPOCHS, DPO_GRADIENT_ACCUMULATION_STEPS,
                       DPO_LEARNING_RATE, DPO_LOGGING_STEPS, DPO_LORA_ALPHA, DPO_LORA_R,
                       DPO_MAX_LENGTH, DPO_MONITOR_EVERY_STEPS, DPO_PER_DEVICE_BATCH_SIZE,
                       DPO_MIN_PAIR_GAP, DPO_SEED, DPO_WARMUP_STEPS, DPO_MONITOR_EVERY_EPOCHS, SLM_MODEL_ID, SOLVER_MODEL_ID)
from eval.ege_data import load_prepared
from eval.model_io import apply_chat_template_no_think, load_slm
from generation.prompts import build_item_paraphrase_prompt
from preferences.build_dpo_dataset import PAIRS_META_PATH, PAIRS_PATH
from utils.artifacts import fingerprint, read_jsonl, write_json
from utils.paths import CHECKPOINTS_DIR, MONITORING_DIR

DPO_DIR = CHECKPOINTS_DIR / "dpo"
FINAL_DIR = DPO_DIR / "final"


def restore_dpo_reference(model, adapter_dir):
    """Keep the original DPO reference when warm-starting a trained adapter."""
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    reference = Path(adapter_dir) / "ref" / "adapter_model.safetensors"
    if reference.exists():
        set_peft_model_state_dict(model, load_file(str(reference)), adapter_name="ref")
        print(f"[DPO] Restored frozen reference adapter from {reference}", flush=True)


def validate_pairs(pairs: list[dict], meta: dict, train: list[dict], manifest: dict) -> None:
    if meta["source"] != manifest or meta["pairs_fingerprint"] != fingerprint(pairs):
        raise ValueError("DPO pairs do not match the prepared data/manifest; rebuild preferences")
    if meta["partial"] or meta["n_source"] != len(train):
        raise ValueError("Partial smoke-test pairs cannot be used for the full DPO run")
    if len(pairs) != len(train) or meta["n_pairs"] != len(pairs):
        raise ValueError("DPO pairs are empty or their count disagrees with the manifest")
    if meta["settings"]["generator"] != SLM_MODEL_ID or meta["settings"]["solver"] != SOLVER_MODEL_ID:
        raise ValueError("Generator/solver changed; rebuild preferences with the intended models")
    if meta.get("minimum_gap", -1) < DPO_MIN_PAIR_GAP:
        raise ValueError("Preference data must certify the requested minimum gap before training")
    by_id = {str(r["id"]): r for r in train}
    seen = set()
    for pair in pairs:
        item_id = pair["item_id"]
        if item_id not in by_id or item_id in seen:
            raise ValueError(f"Duplicate or non-train DPO item {item_id}")
        seen.add(item_id)
        row = by_id[item_id]
        if pair["prompt"] != build_item_paraphrase_prompt(row):
            raise ValueError(f"Prompt/explanation mismatch for {item_id}")
        for side in ("chosen", "rejected"):
            quality = pair[f"{side}_quality"]
            if "valid" not in quality:
                raise ValueError(f"Missing diagnostic flags for {item_id}")
        gap = pair["score_rejected"] - pair["score_chosen"]
        if not math.isfinite(gap) or gap <= meta["minimum_gap"]:
            raise ValueError(f"Invalid adversarial preference direction/gap for {item_id}")


def format_pairs(pairs, tokenizer, *, max_length=DPO_MAX_LENGTH):
    from datasets import Dataset

    formatted = []
    for pair in pairs:
        # Explicitly render no-thinking ONCE. TRL standard text rows leave this
        # rendered chat prompt intact, unlike auto-templating conversation rows.
        prompt = apply_chat_template_no_think(tokenizer, [{"role": "user", "content": pair["prompt"]}])
        for side in ("chosen", "rejected"):
            # TRL tokenizes the prompt and completion separately and appends EOS.
            n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False)) + len(tokenizer.encode(pair[side], add_special_tokens=False)) + 1
            if max_length is not None and n_tokens > max_length:
                raise ValueError(f"Pair {pair['item_id']} {side}: {n_tokens} tokens > native context {max_length}; "
                                 "refusing to truncate the complete exam text")
        formatted.append({"prompt": prompt, "chosen": pair["chosen"], "rejected": pair["rejected"]})
    return Dataset.from_list(formatted)


def make_trainer(model, tokenizer, pairs, *, output_dir: Path, callbacks=None, max_steps=-1,
                 batch_size=DPO_PER_DEVICE_BATCH_SIZE, accumulation=DPO_GRADIENT_ACCUMULATION_STEPS,
                 max_length=DPO_MAX_LENGTH, monitor_every=DPO_MONITOR_EVERY_STEPS):
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    from utils.model_context import context_window
    dataset = format_pairs(pairs, tokenizer, max_length=context_window(model, tokenizer) if max_length is None else max_length)
    args = DPOConfig(
        output_dir=str(output_dir), num_train_epochs=DPO_EPOCHS, max_steps=max_steps,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=accumulation,
        per_device_eval_batch_size=batch_size, prediction_loss_only=True,
        learning_rate=DPO_LEARNING_RATE, beta=DPO_BETA,
        lr_scheduler_type="cosine", warmup_steps=DPO_WARMUP_STEPS,
        logging_steps=DPO_LOGGING_STEPS, logging_first_step=True,
        save_strategy="steps", save_steps=monitor_every, save_total_limit=4,
        eval_strategy="no", report_to="none", seed=DPO_SEED, data_seed=DPO_SEED,
        bf16=model.device.type == "cuda", fp16=False, use_cpu=model.device.type == "cpu",
        optim="adamw_torch", gradient_checkpointing=model.device.type == "cuda",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=max_length, remove_unused_columns=False, dataloader_num_workers=0,
        precompute_ref_log_probs=False,
    )
    peft_config = None if hasattr(model, "peft_config") else LoraConfig(
        r=DPO_LORA_R, lora_alpha=DPO_LORA_ALPHA, lora_dropout=0.0,
        target_modules="all-linear", task_type="CAUSAL_LM", bias="none",
    )
    return DPOTrainer(model=model, ref_model=None, args=args, train_dataset=dataset,
                      processing_class=tokenizer, peft_config=peft_config, callbacks=callbacks or [])


def train_dpo(*, dry_run=False, resume: str | bool | None = None, max_steps=-1, device=None) -> Path:
    train, _, manifest = load_prepared()
    if not PAIRS_PATH.exists() or not PAIRS_META_PATH.exists():
        raise FileNotFoundError("Run python main.py --step preferences first")
    pairs = read_jsonl(PAIRS_PATH)
    meta = json.loads(PAIRS_META_PATH.read_text(encoding="utf-8"))
    validate_pairs(pairs, meta, train, manifest)
    plan = {"model": SLM_MODEL_ID, "solver": SOLVER_MODEL_ID, "pairs": len(pairs),
            "train": manifest["train"], "eval": manifest["test"], "revision": manifest["revision"],
            "pairs_fingerprint": fingerprint(pairs), "epochs": DPO_EPOCHS, "max_steps": max_steps,
            "monitor_every_steps": DPO_MONITOR_EVERY_STEPS, "quality_mode": "completion_only",
            "monitor_every_epochs": DPO_MONITOR_EVERY_EPOCHS,
            "max_length": DPO_MAX_LENGTH, "seed": DPO_SEED,
            "beta": DPO_BETA, "learning_rate": DPO_LEARNING_RATE, "warmup_steps": DPO_WARMUP_STEPS,
            "batch_size": DPO_PER_DEVICE_BATCH_SIZE, "accumulation": DPO_GRADIENT_ACCUMULATION_STEPS,
            "lora_r": DPO_LORA_R, "lora_alpha": DPO_LORA_ALPHA}
    plan["convergence"] = {"criterion": "training_loss_plateau", "window_steps": 20,
                           "patience_windows": 3, "min_steps": 80, "min_delta": .001}
    initial_adapter = os.getenv("DPO_INIT_ADAPTER", "").strip()
    if initial_adapter:
        import hashlib
        initial_path = Path(initial_adapter).resolve()
        with (initial_path / "adapter_model.safetensors").open("rb") as stream:
            adapter_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        plan["initial_adapter"] = {"path": str(initial_path), "sha256": adapter_sha,
                                   "optimizer": "fresh", "scheduler": "fresh"}
        ref_path = initial_path / "ref" / "adapter_model.safetensors"
        if ref_path.exists():
            with ref_path.open("rb") as stream:
                plan["initial_adapter"]["reference_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    plan["live_run_id"] = fingerprint(plan)[:16]
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if dry_run:
        return PAIRS_META_PATH
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This framework uses one GPU; multi-process monitoring is not supported")
    run_path = DPO_DIR / "run_config.json"
    if run_path.exists():
        if not resume:
            raise ValueError("DPO run already exists: use --resume or a different RESULTS_DIR")
        if json.loads(run_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("DPO run/data/settings changed; resume requires the original run configuration")
    elif resume:
        raise ValueError("Cannot resume without the original DPO run_config.json")
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import set_seed
    from eval.evaluate_paraphraser import EgeEvaluator
    from scoring.frozen_solver import FrozenSolver
    from training.callbacks import EgeMonitorCallback
    from training.history import TrainingHistoryCallback
    from training.convergence import LossConvergenceCallback
    from training.reporting import EpochReporter

    set_seed(DPO_SEED)
    model, tokenizer, _ = load_slm(SLM_MODEL_ID, device)
    if initial_adapter:
        model = PeftModel.from_pretrained(model, initial_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=DPO_LORA_R, lora_alpha=DPO_LORA_ALPHA,
                                            lora_dropout=0.0, target_modules="all-linear",
                                            task_type="CAUSAL_LM", bias="none"))
    if SOLVER_MODEL_ID == SLM_MODEL_ID:
        # LoRA freezes the base; disabling its adapter provides the same solver
        # at every checkpoint without keeping another full model on the GPU.
        solver = FrozenSolver(model, tokenizer, model_id=SOLVER_MODEL_ID, disable_adapter=True)
    else:
        solver_model, solver_tokenizer, _ = load_slm(SOLVER_MODEL_ID, device)
        solver = FrozenSolver(solver_model, solver_tokenizer, model_id=SOLVER_MODEL_ID)
    evaluator = EgeEvaluator(tokenizer, solver, output_dir=MONITORING_DIR / "dpo", reset_history=not resume)
    reporter = EpochReporter(MONITORING_DIR / "dpo", plan)
    monitor = EgeMonitorCallback(evaluator, every_steps=DPO_MONITOR_EVERY_STEPS,
                                every_epochs=DPO_MONITOR_EVERY_EPOCHS, reporter=reporter)
    history = TrainingHistoryCallback(MONITORING_DIR / "dpo", resume=bool(resume))
    convergence = LossConvergenceCallback(MONITORING_DIR / "dpo", resume=bool(resume))
    trainer = make_trainer(model, tokenizer, pairs, output_dir=DPO_DIR, callbacks=[history, convergence, monitor], max_steps=max_steps)
    if initial_adapter:
        restore_dpo_reference(trainer.model, initial_adapter)
    monitor.loss_evaluator = lambda: trainer.evaluate(eval_dataset=trainer.train_dataset,
                                                     metric_key_prefix="train_probe")["train_probe_loss"]
    write_json(run_path, plan)
    trainer.train(resume_from_checkpoint=resume or None)
    final_dir = DPO_DIR / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    write_json(MONITORING_DIR / "dpo" / "trainer_state_log.json", trainer.state.log_history)
    plot_history(MONITORING_DIR / "dpo")
    published_url = None
    if (MONITORING_DIR / "dpo" / "convergence.json").exists():
        from training.publish import publish_converged_adapter
        published_url = publish_converged_adapter(final_dir, MONITORING_DIR / "dpo", plan)
    reporter.finish(converged=convergence.converged, model_url=published_url)
    return final_dir


def plot_history(directory: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = json.loads((directory / "validation_history.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    steps = [h["step"] for h in history]
    axes[0].plot(steps, [h["accuracy"] for h in history], label="Rewritten (invalid -> original)")
    axes[0].plot(steps, [h["baseline_accuracy"] for h in history], label="Original")
    axes[0].set(title="Frozen solver accuracy: Reshu EGE 116", xlabel="DPO optimizer step", ylabel="Exact match (fraction)")
    axes[0].legend()
    axes[1].plot(steps, [h["valid_rate"] for h in history], label="Valid rewrites")
    axes[1].plot(steps, [h["conditional_asr"] for h in history], label="ASR on originally correct")
    axes[1].set(title="Rewrite quality and attack success", xlabel="DPO optimizer step", ylabel="Fraction")
    axes[1].legend()
    fig.suptitle("Source: test-reshuege/test, fixed HF revision recorded in eval_protocol.json", fontsize=9)
    fig.tight_layout()
    fig.savefig(directory / "training_summary.png", dpi=150)
    plt.close(fig)
    logs = read_jsonl(directory / "train_history.jsonl")
    losses = [r for r in logs if isinstance(r.get("loss"), (int, float))]
    margins = [r for r in logs if isinstance(r.get("rewards/margins"), (int, float))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([r["step"] for r in losses], [r["loss"] for r in losses])
    axes[0].set(title="DPO training loss", xlabel="Optimizer step", ylabel="Loss")
    axes[1].plot([r["step"] for r in margins], [r["rewards/margins"] for r in margins])
    axes[1].set(title="DPO reward margin", xlabel="Optimizer step", ylabel="Reward margin")
    fig.tight_layout()
    fig.savefig(directory / "loss_history.png", dpi=150)
    plt.close(fig)

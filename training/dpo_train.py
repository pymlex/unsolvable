"""LoRA DPO of the paraphraser, with a frozen no-thinking exam scorer."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from constants import (DPO_BETA, DPO_EPOCHS, DPO_GRADIENT_ACCUMULATION_STEPS,
                       DPO_LEARNING_RATE, DPO_LOGGING_STEPS, DPO_LORA_ALPHA, DPO_LORA_R,
                       DPO_MAX_LENGTH, DPO_MONITOR_EVERY_STEPS, DPO_PER_DEVICE_BATCH_SIZE,
                       DPO_SEED, SLM_MODEL_ID, SOLVER_MODEL_ID)
from eval.ege_data import load_prepared
from eval.model_io import apply_chat_template_no_think, load_slm
from generation.prompts import build_item_paraphrase_prompt
from judge.ege_quality import structural_error
from preferences.build_dpo_dataset import PAIRS_META_PATH, PAIRS_PATH
from utils.artifacts import fingerprint, read_jsonl, write_json
from utils.paths import CHECKPOINTS_DIR, MONITORING_DIR

DPO_DIR = CHECKPOINTS_DIR / "dpo"
FINAL_DIR = DPO_DIR / "final"


def validate_pairs(pairs: list[dict], meta: dict, train: list[dict], manifest: dict) -> None:
    if meta["source"] != manifest or meta["pairs_fingerprint"] != fingerprint(pairs):
        raise ValueError("DPO pairs do not match the prepared data/manifest; rebuild preferences")
    if meta["partial"] or meta["n_source"] != len(train):
        raise ValueError("Partial smoke-test pairs cannot be used for the full DPO run")
    if not pairs or meta["n_pairs"] != len(pairs):
        raise ValueError("DPO pairs are empty or their count disagrees with the manifest")
    if meta["settings"]["generator"] != SLM_MODEL_ID or meta["settings"]["solver"] != SOLVER_MODEL_ID:
        raise ValueError("Generator/solver changed; rebuild preferences with the intended models")
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
            if quality.get("valid") is not True:
                raise ValueError(f"Unvalidated {side} for {item_id}")
            if structural_error(row, {"text": pair[side]}):
                raise ValueError(f"Invalid {side} for {item_id}")
        gap = pair["score_rejected"] - pair["score_chosen"]
        if pair["chosen"] == pair["rejected"] or not math.isfinite(gap) or gap <= 0 or gap < meta["tau"]:
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
        learning_rate=DPO_LEARNING_RATE, beta=DPO_BETA,
        lr_scheduler_type="cosine", warmup_ratio=0.1,
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
            "monitor_every_steps": DPO_MONITOR_EVERY_STEPS, "quality_mode": "structural_only",
            "max_length": DPO_MAX_LENGTH, "seed": DPO_SEED,
            "beta": DPO_BETA, "learning_rate": DPO_LEARNING_RATE,
            "batch_size": DPO_PER_DEVICE_BATCH_SIZE, "accumulation": DPO_GRADIENT_ACCUMULATION_STEPS,
            "lora_r": DPO_LORA_R, "lora_alpha": DPO_LORA_ALPHA}
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
    from peft import LoraConfig, get_peft_model
    from transformers import set_seed
    from eval.evaluate_paraphraser import EgeEvaluator
    from scoring.frozen_solver import FrozenSolver
    from training.callbacks import EgeMonitorCallback

    set_seed(DPO_SEED)
    model, tokenizer, _ = load_slm(SLM_MODEL_ID, device)
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
    monitor = EgeMonitorCallback(evaluator, every_steps=DPO_MONITOR_EVERY_STEPS)
    trainer = make_trainer(model, tokenizer, pairs, output_dir=DPO_DIR, callbacks=[monitor], max_steps=max_steps)
    write_json(run_path, plan)
    trainer.train(resume_from_checkpoint=resume or None)
    final_dir = DPO_DIR / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    write_json(MONITORING_DIR / "dpo" / "train_history.json", trainer.state.log_history)
    plot_history(MONITORING_DIR / "dpo")
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

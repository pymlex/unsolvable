#!/usr/bin/env python3
"""TRL GRPO training scaffold on paraphrase groups with reward ``-s_m``.

Full training is intended for Clore GPU. Use ``--dry-run`` on CPU VMs to
validate dataset wiring without launching the trainer loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from constants import (
    GRPO_EPOCHS,
    GRPO_GRADIENT_ACCUMULATION_STEPS,
    GRPO_GROUP_SIZE,
    GRPO_LEARNING_RATE,
    GRPO_MAX_STEPS,
    GRPO_PER_DEVICE_BATCH_SIZE,
    SLM_MODEL_ID,
)
from utils.paths import CHECKPOINTS_DIR, GROUPS_DIR, ensure_result_dirs


def load_groups(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def groups_to_trl_dataset(groups: list[dict]):
    """Flatten GRPO groups into a Hugging Face Dataset for TRL GRPOTrainer."""
    from datasets import Dataset

    prompts: list[str] = []
    for group in groups:
        prompts.append(group["prompt"])
    return Dataset.from_dict({"prompt": prompts})


def build_reward_lookup(groups: list[dict]) -> dict[str, dict[str, float]]:
    """Map prompt -> completion -> reward for offline reward lookup."""
    lookup: dict[str, dict[str, float]] = {}
    for group in groups:
        prompt = group["prompt"]
        lookup[prompt] = {
            completion: float(reward)
            for completion, reward in zip(
                group["completions"],
                group["rewards"],
                strict=True,
            )
        }
    return lookup


def run_grpo(
    *,
    groups_path: Path | None = None,
    model_id: str = SLM_MODEL_ID,
    dry_run: bool = False,
    max_steps: int = GRPO_MAX_STEPS,
) -> Path:
    """Train or dry-run GRPO on precomputed paraphrase groups."""
    ensure_result_dirs()
    path = groups_path or (GROUPS_DIR / "grpo_groups.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing groups file: {path}. Run preferences.build_groups first."
        )
    groups = load_groups(path)
    if not groups:
        raise RuntimeError(f"No groups in {path}")

    dataset = groups_to_trl_dataset(groups)
    reward_lookup = build_reward_lookup(groups)
    meta = {
        "n_groups": len(groups),
        "model": model_id,
        "group_size": GRPO_GROUP_SIZE,
        "dry_run": dry_run,
        "reward": "-s_m(y_star | x)",
        "dataset_rows": len(dataset),
    }
    meta_path = CHECKPOINTS_DIR / "grpo_dry_run_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    if dry_run:
        print("Dry-run complete. Skipping TRL GRPOTrainer.")
        return meta_path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)

    def reward_fn(completions: list[str], prompts: list[str], **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions, strict=True):
            table = reward_lookup.get(prompt, {})
            rewards.append(float(table.get(completion, 0.0)))
        return rewards

    args = GRPOConfig(
        output_dir=str(CHECKPOINTS_DIR / "grpo"),
        num_train_epochs=GRPO_EPOCHS,
        max_steps=max_steps,
        per_device_train_batch_size=GRPO_PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRPO_GRADIENT_ACCUMULATION_STEPS,
        learning_rate=GRPO_LEARNING_RATE,
        logging_steps=1,
        report_to="none",
        bf16=device.type == "cuda",
        remove_unused_columns=False,
        num_generations=min(GRPO_GROUP_SIZE, 4),
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    final_dir = CHECKPOINTS_DIR / "grpo" / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved GRPO model to {final_dir}")
    return final_dir


def main() -> None:
    p = argparse.ArgumentParser(description="TRL GRPO on tip-2 paraphrase groups")
    p.add_argument("--groups", type=Path, default=None)
    p.add_argument("--model", default=SLM_MODEL_ID)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-steps", type=int, default=GRPO_MAX_STEPS)
    args = p.parse_args()
    run_grpo(
        groups_path=args.groups,
        model_id=args.model,
        dry_run=args.dry_run,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Zero-edit tip-2 baseline: temperature 0, exact-match accuracy with 95% CI.

Example:
  python -m eval.baseline --model Qwen/Qwen2.5-0.5B-Instruct --limit 0
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from tqdm.auto import tqdm

from eval.metrics import (
    binomial_wilson_ci,
    bootstrap_ci,
    exact_match,
    gold_normalize,
    normalize_answer,
)
from eval.prompt import build_messages


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EGE tip-2 zero-edit SLM baseline")
    p.add_argument(
        "--dataset",
        default="pymlex/ege-rus-tip2-reshu-2026",
        help="HF dataset id or local jsonl path",
    )
    p.add_argument("--split", default="train")
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model id",
    )
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, evaluate only the first N items",
    )
    p.add_argument(
        "--device",
        default=None,
        help="cuda / cpu / mps; default: auto",
    )
    p.add_argument(
        "--out",
        default="results/baseline.json",
        help="Path for JSON results",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_rows(dataset: str, split: str) -> list[dict]:
    path = Path(dataset)
    if path.exists() and path.suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    ds = load_dataset(dataset, split=split)
    return [dict(r) for r in ds]


def load_model(model_id: str, device: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_one(model, tokenizer, device: str, row: dict, max_new_tokens: int) -> str:
    import torch

    messages = build_messages(row)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = messages[0]["content"] + "\n\n" + messages[1]["content"]

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.dataset, args.split)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    model, tokenizer, device = load_model(args.model, args.device)

    records = []
    correct_flags: list[bool] = []
    for row in tqdm(rows, desc="eval"):
        raw = generate_one(model, tokenizer, device, row, args.max_new_tokens)
        pred = normalize_answer(raw)
        gold = gold_normalize(row["answer"])
        ok = exact_match(raw, row["answer"])
        correct_flags.append(ok)
        records.append(
            {
                "id": row["id"],
                "gold": gold,
                "pred": pred,
                "raw": raw,
                "correct": ok,
            }
        )

    n = len(correct_flags)
    successes = int(sum(correct_flags))
    acc, wilson_lo, wilson_hi = binomial_wilson_ci(successes, n)
    _, boot_lo, boot_hi = bootstrap_ci(correct_flags, seed=args.seed)

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "n_items": n,
        "temperature": 0,
        "do_sample": False,
        "protocol": "zero-edit-full-set",
        "accuracy": acc,
        "n_correct": successes,
        "wilson_95_ci": [wilson_lo, wilson_hi],
        "bootstrap_95_ci": [boot_lo, boot_hi],
        "device": device,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "items": records}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    table_path = out_path.with_suffix(".md")
    table_path.write_text(
        (
            f"| model | n | accuracy | Wilson 95% CI | bootstrap 95% CI |\n"
            f"|---|---:|---:|---|---|\n"
            f"| {args.model} | {n} | {acc:.4f} | "
            f"[{wilson_lo:.4f}, {wilson_hi:.4f}] | "
            f"[{boot_lo:.4f}, {boot_hi:.4f}] |\n"
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

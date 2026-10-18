#!/usr/bin/env python3
"""Regenerate specific synthetic tip-2 rows in place (same pipeline, keep ids)."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from constants import (
    DATASET_ID,
    DATASET_SPLIT,
    SYNTHESIS_FEWSHOT_K,
    SYNTHESIS_SPLIT_NAME,
    ZVENO_MODEL_ID,
)
from eval.data import load_rows
from eval.zveno_client import chat_completion, completion_text, has_zveno_key
from synthesis.dictionary import fetch_senses_udarenieru, is_weak_gloss
from synthesis.generate import (
    MAX_ATTEMPTS_PER_ITEM,
    _chat_with_retries,
    _fetch_all_senses,
    _parse_item,
    build_synthesis_messages,
)
from synthesis.lemmas import sample_lemmas
from utils.paths import SYNTHESIS_DIR


def _regen_one(
    *,
    item_id: int,
    gold: list[dict],
    model_id: str,
    fewshot_k: int,
    rng: random.Random,
) -> dict:
    last_err = ""
    for attempt in range(1, MAX_ATTEMPTS_PER_ITEM * 3 + 1):
        try:
            lemmas = sample_lemmas(5, rng=rng)
            sense_map = _fetch_all_senses(lemmas)
            sense_pool = [g for lemma in lemmas for g in sense_map[lemma]]
            if any(is_weak_gloss(g) for g in sense_pool):
                raise ValueError("weak gloss leaked into sense pool")
            examples = rng.sample(gold, k=min(fewshot_k, len(gold)))
            body = _chat_with_retries(
                build_synthesis_messages(lemmas, sense_map, examples),
                model_id,
            )
            message = (body.get("choices") or [{}])[0].get("message") or {}
            candidates = [
                message.get("content") or "",
                message.get("reasoning_content") or "",
                message.get("reasoning") or "",
                completion_text(body),
            ]
            parsed = None
            for raw in candidates:
                if not str(raw).strip():
                    continue
                parsed = _parse_item(str(raw), item_id=item_id, sense_pool=sense_pool)
                break
            if parsed is None:
                raise ValueError("empty synthesis completion")
            for opt in parsed.options:
                if is_weak_gloss(opt):
                    raise ValueError(f"weak option survived: {opt[:80]!r}")
            return parsed.model_dump()
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(min(1 + attempt * 0.5, 20))
    raise RuntimeError(f"failed to regenerate id={item_id}: {last_err}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ids", nargs="+", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default=ZVENO_MODEL_ID)
    p.add_argument("--fewshot-k", type=int, default=SYNTHESIS_FEWSHOT_K)
    args = p.parse_args()

    if not has_zveno_key():
        raise SystemExit("ZVENOAI_API_KEY required")

    # Drop cached senses scraped before weak-gloss filter.
    fetch_senses_udarenieru.cache_clear()

    out_dir = SYNTHESIS_DIR / SYNTHESIS_SPLIT_NAME
    jsonl_path = out_dir / "train.jsonl"
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {int(r["id"]): i for i, r in enumerate(rows)}
    missing = [i for i in args.ids if i not in by_id]
    if missing:
        raise SystemExit(f"ids not found in train.jsonl: {missing}")

    gold = load_rows(DATASET_ID, DATASET_SPLIT)
    rng = random.Random(args.seed)
    replaced = []
    for item_id in args.ids:
        print(f"regenerating {item_id} ...", flush=True)
        new_row = _regen_one(
            item_id=item_id,
            gold=gold,
            model_id=args.model,
            fewshot_k=args.fewshot_k,
            rng=rng,
        )
        idx = by_id[item_id]
        old = rows[idx]
        rows[idx] = new_row
        replaced.append(
            {
                "id": item_id,
                "old_opts": [str(o)[:70] for o in old.get("options", [])],
                "new_opts": [str(o)[:70] for o in new_row.get("options", [])],
                "new_answer": new_row.get("answer"),
            }
        )
        print(json.dumps(replaced[-1], ensure_ascii=False, indent=2), flush=True)

    if len(rows) != 500:
        raise SystemExit(f"refusing to write: expected 500 rows, got {len(rows)}")

    tmp = jsonl_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(jsonl_path)

    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["n_items"] = len(rows)
    meta["regenerated_ids"] = list(args.ids)
    meta["regenerated_at_utc"] = datetime.now(timezone.utc).isoformat()
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "n_rows": len(rows),
        "replaced": replaced,
    }
    (out_dir / "regen_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

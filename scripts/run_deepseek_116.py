#!/usr/bin/env python3
"""Full-set DeepSeek tip-2 benchmark with incremental save and answer validation."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm

import _bootstrap  # noqa: F401

from constants import DATASET_ID, DATASET_SPLIT, ZVENO_MODEL_ID
from eval.data import load_rows
from eval.metrics import (
    binomial_wilson_ci,
    bootstrap_ci,
    exact_match,
    gold_normalize,
    normalize_answer,
    write_benchmark_table,
)
from eval.prompt import build_messages
from eval.zveno_client import chat_completion, completion_text
from utils.paths import BENCHMARK_DIR, ensure_result_dirs


OUT_JSON = BENCHMARK_DIR / "deepseek_v4_flash_tip2_116.json"
OUT_MD = BENCHMARK_DIR / "deepseek_v4_flash_tip2_116.md"
OUT_ISSUES = BENCHMARK_DIR / "deepseek_v4_flash_tip2_116_issues.jsonl"


def answer_sets(pred: str, gold: str) -> tuple[set[str], set[str]]:
    """Digit sets for tip-2 answers after normalization."""
    p = set(normalize_answer(pred))
    g = set(gold_normalize(gold))
    return p, g


def answer_jaccard(pred: str, gold: str) -> float:
    """Jaccard on digit sets: |P∩G| / |P∪G|. Empty-empty -> 0."""
    p, g = answer_sets(pred, gold)
    if not p and not g:
        return 0.0
    union = p | g
    if not union:
        return 0.0
    return len(p & g) / len(union)



def validate_pred(raw: str, pred: str, gold: str) -> dict:
    """Diagnose whether the model output is a usable tip-2 answer."""
    issues: list[str] = []
    if not str(raw).strip():
        issues.append("empty_raw")
    if not pred:
        issues.append("no_digits_1_to_5")
    if re.search(r"[^0-9\s]", str(raw).strip()) and not re.fullmatch(r"[0-9]+", str(raw).strip()):
        # raw is not pure digits
        if not re.fullmatch(r"[1-5]+", pred):
            issues.append("non_digit_noise_in_raw")
        else:
            issues.append("raw_not_pure_digits_but_parsed")
    if pred and re.search(r"[^1-5]", pred):
        issues.append("digits_outside_1_5")
    if pred and "".join(sorted(set(pred))) != pred:
        issues.append("pred_not_sorted_unique")
    # gold format check
    g = gold_normalize(gold)
    if not g:
        issues.append("empty_gold")
    if any(ch not in "12345" for ch in g):
        issues.append("gold_outside_1_5")
    valid = "no_digits_1_to_5" not in issues and "empty_raw" not in issues
    return {"valid_pred": valid, "issues": issues, "pred": pred, "gold": g}


def save_payload(records: list[dict], correct_flags: list[bool], meta: dict) -> None:
    n = len(correct_flags)
    successes = int(sum(correct_flags))
    acc, wilson_lo, wilson_hi = binomial_wilson_ci(successes, n) if n else (0.0, 0.0, 0.0)
    _, boot_lo, boot_hi = bootstrap_ci(correct_flags, seed=0) if n else (0.0, 0.0, 0.0)
    jaccards = [float(r.get("jaccard", 0.0)) for r in records]
    mean_j = float(sum(jaccards) / n) if n else 0.0
    n_invalid = sum(1 for r in records if not r.get("valid_pred"))
    summary = {
        **meta,
        "n_items_done": n,
        "n_correct": successes,
        "exact_match_accuracy": acc,
        "accuracy": acc,
        "mean_jaccard": mean_j,
        "wilson_95_ci_exact": [wilson_lo, wilson_hi],
        "wilson_95_ci": [wilson_lo, wilson_hi],
        "bootstrap_95_ci": [boot_lo, boot_hi],
        "n_invalid_pred": n_invalid,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload = {"summary": summary, "items": records}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if n:
        write_benchmark_table(OUT_MD, meta["model"], n, acc, (wilson_lo, wilson_hi), (boot_lo, boot_hi))


def main() -> None:
    ensure_result_dirs()
    rows = load_rows(DATASET_ID, DATASET_SPLIT)
    meta = {
        "dataset": DATASET_ID,
        "split": DATASET_SPLIT,
        "model": ZVENO_MODEL_ID,
        "temperature": 0,
        "sleep": 0,
        "max_tokens": None,
        "protocol": "zero-edit-full-set-116",
        "device": "zveno-api",
    }

    # resume if partial exists
    done_ids: set[str] = set()
    records: list[dict] = []
    correct_flags: list[bool] = []
    if OUT_JSON.exists():
        prev = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        for it in prev.get("items") or []:
            if "jaccard" not in it:
                it["jaccard"] = answer_jaccard(it.get("pred", ""), it.get("gold", ""))
            records.append(it)
            correct_flags.append(bool(it.get("correct")))
            done_ids.add(str(it.get("id")))
        mean_j = sum(float(r["jaccard"]) for r in records) / max(1, len(records))
        n_ok = int(sum(correct_flags))
        print(
            f"resume from {len(done_ids)} items exact={n_ok}/{len(done_ids)} "
            f"mean_jaccard={mean_j:.4f}",
            flush=True,
        )

    issues_f = OUT_ISSUES.open("a", encoding="utf-8")
    try:
        for row in tqdm(rows, desc="deepseek-116"):
            rid = str(row["id"])
            if rid in done_ids:
                continue
            t0 = time.perf_counter()
            try:
                body = chat_completion(
                    build_messages(row),
                    model=ZVENO_MODEL_ID,
                    temperature=0.0,
                    max_tokens=None,
                    timeout_sec=180.0,
                )
                raw = completion_text(body)
                err = None
            except Exception as exc:  # noqa: BLE001
                body = {}
                raw = ""
                err = f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - t0
            pred = normalize_answer(raw)
            gold = gold_normalize(row["answer"])
            ok = exact_match(raw, row["answer"]) if not err else False
            diagnosis = validate_pred(raw, pred, row["answer"])
            # if invalid, one retry with stricter reminder
            retried = False
            if not diagnosis["valid_pred"] and not err:
                retried = True
                messages = build_messages(row)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Предыдущий ответ невалиден для бланка ЕГЭ типа 2. "
                            "Верни только цифры номеров верных вариантов по возрастанию, "
                            "без пробелов и текста. Пример формата: 35"
                        ),
                    }
                )
                try:
                    body2 = chat_completion(
                        messages,
                        model=ZVENO_MODEL_ID,
                        temperature=0.0,
                        max_tokens=None,
                        timeout_sec=180.0,
                    )
                    raw2 = completion_text(body2)
                    pred2 = normalize_answer(raw2)
                    diag2 = validate_pred(raw2, pred2, row["answer"])
                    if diag2["valid_pred"] or pred2:
                        raw, pred, body = raw2, pred2, body2
                        diagnosis = diag2
                        ok = exact_match(raw, row["answer"])
                        latency = time.perf_counter() - t0
                except Exception as exc:  # noqa: BLE001
                    err = f"retry_{type(exc).__name__}: {exc}"

            usage = body.get("usage") or {}
            jac = answer_jaccard(pred, gold)
            pset, gset = answer_sets(pred, gold)
            rec = {
                "id": rid,
                "gold": gold,
                "pred": pred,
                "raw": raw,
                "correct": ok,
                "jaccard": jac,
                "intersection": "".join(sorted(pset & gset)),
                "extra": "".join(sorted(pset - gset)),
                "missing": "".join(sorted(gset - pset)),
                "valid_pred": diagnosis["valid_pred"],
                "issues": diagnosis["issues"],
                "retried": retried,
                "error": err,
                "latency_sec": latency,
                "usage": usage,
            }
            records.append(rec)
            correct_flags.append(ok)
            done_ids.add(rid)
            n_done = len(correct_flags)
            n_ok = int(sum(correct_flags))
            running_exact = n_ok / n_done if n_done else 0.0
            mean_j = sum(float(r["jaccard"]) for r in records) / n_done
            status = "OK" if ok else "WRONG"
            if err:
                status = "ERROR"
            elif not diagnosis["valid_pred"]:
                status = "INVALID"
            print(
                f"[{n_done}/116] id={rid} {status} gold={gold} pred={pred!r} "
                f"jaccard={jac:.3f} inter={rec['intersection']!r} extra={rec['extra']!r} missing={rec['missing']!r} "
                f"mean_jaccard={mean_j:.4f} exact={n_ok}/{n_done}={running_exact:.4f} "
                f"lat={latency:.1f}s",
                flush=True,
            )
            if not diagnosis["valid_pred"] or err:
                issues_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                issues_f.flush()
                print(
                    f"  diagnose issues={diagnosis['issues']} err={err} raw={raw[:160]!r}",
                    flush=True,
                )
            save_payload(records, correct_flags, meta)
    finally:
        issues_f.close()

    print(json.dumps(json.loads(OUT_JSON.read_text())["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

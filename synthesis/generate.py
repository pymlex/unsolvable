#!/usr/bin/env python3
"""Few-shot synthetic tip-2 generation with DeepSeek via Zveno.

Protocol:
1. Sample 5 non-frequent lemmas.
2. Fetch ALL udarenieru.ru senses for each lemma.
3. Few-shot from gold tip-2 rows.
4. Ask DeepSeek to write an EGE tip-2 item with literal dictionary glosses,
   exactly 2-3 correct options, and gold-compatible fields.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm

import _bootstrap  # noqa: F401

from constants import (
    DATASET_ID,
    DATASET_SPLIT,
    SYNTHESIS_FEWSHOT_K,
    SYNTHESIS_N_ITEMS,
    SYNTHESIS_SPLIT_NAME,
    ZVENO_MODEL_ID,
)
from eval.data import load_rows
from eval.prompt import FIXED_INSTRUCTION, format_options, item_text
from eval.zveno_client import chat_completion, completion_text, has_zveno_key
from schemas.models import SyntheticItem
from synthesis.dictionary import UDARENIE_RU_DOC, fetch_senses
from synthesis.lemmas import sample_lemmas
from utils.paths import SYNTHESIS_DIR, ensure_result_dirs


STATUS_PATH = Path(
    os.environ.get(
        "SYNTH_STATUS_PATH",
        "/cursor/stores/self/internal/synth-1000-status.md",
    )
)
PROGRESS_EVERY = int(os.environ.get("SYNTH_PROGRESS_EVERY", "25"))
MAX_ATTEMPTS_PER_ITEM = int(os.environ.get("SYNTH_MAX_ATTEMPTS", "5"))
API_RETRIES = int(os.environ.get("SYNTH_API_RETRIES", "5"))
SYNTH_WORKERS = int(os.environ.get("SYNTH_WORKERS", "6"))
SYNTH_ID_BASE = int(os.environ.get("SYNTH_ID_BASE", "800000"))
MIN_SENSES_PER_LEMMA = int(os.environ.get("SYNTH_MIN_SENSES", "2"))


def _normalize_gloss(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .;:")
    return text


def _gloss_matches_pool(option: str, pool: list[str]) -> bool:
    opt = _normalize_gloss(option)
    opt = re.sub(r"^[а-яёa-z\-]+\.\s*", "", opt)
    if len(opt) < 8:
        return False
    for sense in pool:
        s = _normalize_gloss(sense)
        if not s:
            continue
        if opt == s or opt in s or s in opt:
            return True
        if len(opt) >= 20 and opt[:40] in s:
            return True
    return False


def _fewshot_block(examples: list[dict]) -> str:
    chunks: list[str] = []
    for ex in examples:
        passage = item_text(ex)
        chunks.append(
            "Пример реального задания:\n"
            f"Инструкция: {FIXED_INSTRUCTION}\n"
            f"Текст:\n{passage}\n"
            f"Варианты:\n{format_options(list(ex['options']))}\n"
            f"Ответ: {ex['answer']}\n"
            f"Пояснение: {ex.get('explanation', '')}"
        )
    return "\n\n".join(chunks)


def _sense_block(lemmas: list[str], senses: dict[str, list[str]]) -> str:
    parts: list[str] = []
    for lemma in lemmas:
        glosses = senses.get(lemma) or []
        numbered = "\n".join(f"  {i}. {g}" for i, g in enumerate(glosses, start=1))
        parts.append(f"Слово «{lemma}» — все словарные значения:\n{numbered}")
    return "\n\n".join(parts)


def build_synthesis_messages(
    lemmas: list[str],
    senses: dict[str, list[str]],
    examples: list[dict],
) -> list[dict[str, str]]:
    user = (
        f"{_fewshot_block(examples)}\n\n"
        "Сгенерируй ОДНО новое задание ЕГЭ по русскому языку типа 2 "
        "в том же стиле и той же раскладке, что в примерах.\n\n"
        f"Фиксированная инструкция задания: {FIXED_INSTRUCTION}\n\n"
        "Требования:\n"
        "1. Возьми ровно эти пять слов как целевые: "
        f"{', '.join(lemmas)}.\n"
        "2. Напиши связный текст, в котором каждое из пяти слов встречается "
        "ровно один раз и выделено разметкой **слово** как в золотых примерах.\n"
        "3. Для КАЖДОГО выделенного слова используй РОВНО ОДНО значение из "
        "полного списка ниже, которое подходит по контексту текста. "
        "Остальные значения того же слова для этого контекста неверны — "
        "не смешивай их.\n"
        "4. Ровно пять вариантов ответа с подписями смысла 1.–5. "
        "Текст каждого варианта должен быть БУКВАЛЬНОЙ словарной формулировкой "
        "из списков значений ниже, без перефразирования и без примеров.\n"
        "5. Ровно 2 или 3 варианта из пяти верны для текста. "
        "Ответ — склейка номеров верных вариантов, например 35 или 124. "
        "Длина ответа 2 или 3 цифры.\n"
        "6. Длина и стиль текста как у few-shot примеров: примерно столько же "
        "предложений и слов, не короче заметно и не длиннее заметно. "
        "Не пиши эссе и не раздувай абзацы.\n"
        "7. Верни только один JSON-объект с полями "
        "passage, highlighted_words, options, answer, explanation. "
        "Поле passage — текст с **выделениями**. "
        "highlighted_words — список из пяти выделенных словоформ. "
        "options — список из пяти строк-толкований. "
        "answer — строка из цифр. "
        "Без markdown-ограждений и без текста вокруг JSON.\n\n"
        f"{_sense_block(lemmas, senses)}"
    )
    return [
        {
            "role": "system",
            "content": (
                "Ты автор печатно корректных заданий ЕГЭ по русскому языку типа 2. "
                "Пиши по-русски. Не копируй примеры дословно. "
                "Толкования в вариантах бери дословно из словаря."
            ),
        },
        {"role": "user", "content": user},
    ]


def _balanced_json_slices(text: str) -> list[str]:
    slices: list[str] = []
    start = None
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = idx
                depth = 1
                in_string = False
                escape = False
            continue
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                slices.append(text[start : idx + 1])
                start = None
    return slices


def _extract_json_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    for chunk in _balanced_json_slices(text):
        payload = json.loads(chunk)
        if not isinstance(payload, dict):
            continue
        if "options" in payload and "answer" in payload and (
            "passage" in payload or "text" in payload
        ):
            return payload
    raise ValueError("no tip-2 JSON object in model output")


def _validate_tip2(
    passage: str,
    options: list[str],
    answer: str,
    sense_pool: list[str],
) -> list[str]:
    highlights = re.findall(r"\*\*([^*]+)\*\*", passage)
    if len(highlights) != 5:
        raise ValueError(f"expected 5 **highlights**, got {len(highlights)}")
    if len(options) != 5:
        raise ValueError(f"expected 5 options, got {len(options)}")
    digits = [ch for ch in answer if ch.isdigit()]
    if len(digits) not in {2, 3}:
        raise ValueError(f"answer must have 2 or 3 digits, got {answer!r}")
    if any(ch not in "12345" for ch in digits):
        raise ValueError(f"answer digits out of range: {answer}")
    if len(set(digits)) != len(digits):
        raise ValueError(f"duplicate answer digits: {answer}")
    for opt in options:
        if not _gloss_matches_pool(str(opt), sense_pool):
            raise ValueError(f"option not literal dictionary gloss: {opt[:80]!r}")
    return [h.strip() for h in highlights]


def _parse_item(
    raw: str,
    item_id: int,
    sense_pool: list[str],
) -> SyntheticItem:
    payload = _extract_json_object(raw)
    passage = str(payload.get("passage") or payload.get("text") or "")
    options = [str(x) for x in list(payload["options"])]
    answer = "".join(ch for ch in str(payload["answer"]) if ch.isdigit())
    highlights = _validate_tip2(passage, options, answer, sense_pool)
    highlighted = payload.get("highlighted_words")
    if isinstance(highlighted, list) and len(highlighted) == 5:
        highlighted_words = [str(x) for x in highlighted]
    else:
        highlighted_words = highlights
    return SyntheticItem(
        id=item_id,
        passage=passage,
        highlighted_words=highlighted_words,
        options=options,
        answer=answer,
        answer_list=[int(ch) for ch in answer],
        explanation=str(payload.get("explanation", "")),
    )


def _load_existing(jsonl_path: Path) -> list[dict]:
    if not jsonl_path.exists():
        return []
    rows: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _append_item(jsonl_path: Path, item: SyntheticItem) -> None:
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(item.model_dump_json() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _chat_with_retries(messages: list[dict[str, str]], model_id: str) -> dict:
    last_err: Exception | None = None
    for attempt in range(1, API_RETRIES + 1):
        try:
            return chat_completion(
                messages,
                model=model_id,
                temperature=0.3,
                extra={"thinking": {"type": "disabled"}},
            )
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            time.sleep(min(2 ** attempt, 60))
        except Exception as exc:
            last_err = exc
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Zveno chat failed after {API_RETRIES} retries: {last_err}")


def _write_status(
    *,
    n_done: int,
    n_target: int,
    n_invalid: int,
    started_at: float,
    last_error: str,
    phase: str,
    jsonl_path: Path,
) -> None:
    elapsed_min = max((time.time() - started_at) / 60.0, 1e-6)
    rate = n_done / elapsed_min if n_done else 0.0
    remaining = max(n_target - n_done, 0)
    eta_min = (remaining / rate) if rate > 0 else None
    eta_str = f"{eta_min:.1f} min" if eta_min is not None else "n/a"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    body = (
        f"# Synth 1000 status\n\n"
        f"- updated: {now}\n"
        f"- phase: {phase}\n"
        f"- n_done: **{n_done} / {n_target}**\n"
        f"- invalid_or_skip: {n_invalid}\n"
        f"- rate: {rate:.2f} items/min\n"
        f"- ETA: {eta_str}\n"
        f"- last_error: {last_error or 'none'}\n"
        f"- output: `{jsonl_path}`\n"
        f"- split: `{SYNTHESIS_SPLIT_NAME}`\n"
        f"- model: `{ZVENO_MODEL_ID}`\n"
        f"- protocol: 5 lemmas → ALL udarenieru senses → gold few-shot → "
        f"EGE tip-2 JSON (passage/options/answer 2–3)\n"
    )
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(body, encoding="utf-8")
    alt = Path("/cursor/stores/bc-190c58b9-1167-4499-a9a7-ccf4cda65298/internal/synth-1000-status.md")
    if alt != STATUS_PATH:
        alt.parent.mkdir(parents=True, exist_ok=True)
        alt.write_text(body, encoding="utf-8")


def _fetch_all_senses(lemmas: list[str]) -> dict[str, list[str]]:
    sense_map: dict[str, list[str]] = {}
    for lemma in lemmas:
        senses = fetch_senses(lemma, max_senses=32)
        if len(senses) < MIN_SENSES_PER_LEMMA:
            raise ValueError(f"too few udarenieru senses for {lemma!r}: {len(senses)}")
        sense_map[lemma] = senses
    return sense_map


def generate_synthetic_split(
    *,
    n_items: int = SYNTHESIS_N_ITEMS,
    fewshot_k: int = SYNTHESIS_FEWSHOT_K,
    dataset: str = DATASET_ID,
    split: str = DATASET_SPLIT,
    model_id: str = ZVENO_MODEL_ID,
    seed: int = 0,
    stub_if_no_api: bool = True,
) -> Path:
    """Generate a local synthetic tip-2 split with resume-safe appends."""
    ensure_result_dirs()
    gold = load_rows(dataset, split)
    out_dir = SYNTHESIS_DIR / SYNTHESIS_SPLIT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "train.jsonl"
    progress_path = out_dir / "progress.json"

    existing = _load_existing(jsonl_path)
    n_done = len(existing)
    n_invalid = 0
    last_error = ""
    started_at = time.time()
    if progress_path.exists():
        prev = json.loads(progress_path.read_text(encoding="utf-8"))
        n_invalid = int(prev.get("n_invalid", 0))
        if prev.get("started_at_unix"):
            started_at = float(prev["started_at_unix"])

    _write_status(
        n_done=n_done,
        n_target=n_items,
        n_invalid=n_invalid,
        started_at=started_at,
        last_error=last_error,
        phase="resume" if n_done else "start",
        jsonl_path=jsonl_path,
    )

    if not has_zveno_key() and stub_if_no_api:
        raise RuntimeError("ZVENOAI_API_KEY required for udarenieru→DeepSeek synthesis")

    lock = threading.Lock()
    state = {
        "n_done": n_done,
        "n_invalid": n_invalid,
        "last_error": last_error,
        "last_status_write": time.time(),
    }
    pbar = tqdm(total=n_items, initial=n_done, desc="synthesis")

    def _persist_locked() -> None:
        progress = {
            "n_done": state["n_done"],
            "n_target": n_items,
            "n_invalid": state["n_invalid"],
            "last_error": state["last_error"],
            "started_at_unix": started_at,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "workers": SYNTH_WORKERS,
            "protocol": "udarenieru-all-senses-gold-fewshot-ege-tip2",
        }
        progress_path.write_text(
            json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if (
            state["n_done"] % PROGRESS_EVERY == 0
            or state["n_done"] >= n_items
            or (time.time() - state["last_status_write"]) >= 120
        ):
            _write_status(
                n_done=state["n_done"],
                n_target=n_items,
                n_invalid=state["n_invalid"],
                started_at=started_at,
                last_error=state["last_error"],
                phase="running",
                jsonl_path=jsonl_path,
            )
            state["last_status_write"] = time.time()

    def _worker_loop(worker_id: int) -> None:
        local_rng = random.Random(seed + 10007 * (worker_id + 1))
        while True:
            with lock:
                if state["n_done"] >= n_items:
                    return
            attempt_err = ""
            item = None
            for attempt in range(1, MAX_ATTEMPTS_PER_ITEM + 1):
                with lock:
                    if state["n_done"] >= n_items:
                        return
                try:
                    lemmas = sample_lemmas(5, rng=local_rng)
                    sense_map = _fetch_all_senses(lemmas)
                    sense_pool = [g for lemma in lemmas for g in sense_map[lemma]]
                    examples = local_rng.sample(gold, k=min(fewshot_k, len(gold)))
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
                        with lock:
                            idx = SYNTH_ID_BASE + state["n_done"]
                        parsed = _parse_item(str(raw), item_id=idx, sense_pool=sense_pool)
                        break
                    if parsed is None:
                        raise ValueError("empty synthesis completion")
                    item = parsed
                    break
                except Exception as exc:
                    attempt_err = f"{type(exc).__name__}: {exc}"
                    with lock:
                        state["n_invalid"] += 1
                        state["last_error"] = attempt_err
                    time.sleep(min(attempt, 5))
            if item is None:
                with lock:
                    state["last_error"] = attempt_err or "failed all attempts"
                    _write_status(
                        n_done=state["n_done"],
                        n_target=n_items,
                        n_invalid=state["n_invalid"],
                        started_at=started_at,
                        last_error=state["last_error"],
                        phase="retrying",
                        jsonl_path=jsonl_path,
                    )
                continue
            with lock:
                if state["n_done"] >= n_items:
                    return
                item.id = SYNTH_ID_BASE + state["n_done"]
                _append_item(jsonl_path, item)
                state["n_done"] += 1
                pbar.update(1)
                _persist_locked()

    workers = max(1, SYNTH_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker_loop, i) for i in range(workers)]
        for fut in as_completed(futures):
            fut.result()
    pbar.close()
    n_done = state["n_done"]
    n_invalid = state["n_invalid"]
    last_error = state["last_error"]

    rows = _load_existing(jsonl_path)
    meta = {
        "split": SYNTHESIS_SPLIT_NAME,
        "n_items": len(rows),
        "n_invalid_or_skip": n_invalid,
        "model": model_id,
        "dictionary": "udarenieru.ru HTML scrape (Толкование / Evgenyeva)",
        "udarenieru_note": UDARENIE_RU_DOC,
        "schema_fields": [
            "id",
            "passage",
            "highlighted_words",
            "options",
            "answer",
            "answer_list",
            "explanation",
        ],
        "answer_correct_count": "2-3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "temperature": 0.3,
        "fewshot_k": fewshot_k,
        "hf_push": (
            "python scripts/push_synthesis_hf.py "
            f"--local {jsonl_path} --repo pymlex/reshuege-ru-2026 "
            f"--config-name {SYNTHESIS_SPLIT_NAME}"
        ),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_status(
        n_done=len(rows),
        n_target=n_items,
        n_invalid=n_invalid,
        started_at=started_at,
        last_error=last_error,
        phase="done" if len(rows) >= n_items else "partial",
        jsonl_path=jsonl_path,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return jsonl_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-items", type=int, default=SYNTHESIS_N_ITEMS)
    p.add_argument("--fewshot-k", type=int, default=SYNTHESIS_FEWSHOT_K)
    p.add_argument("--dataset", default=DATASET_ID)
    p.add_argument("--split", default=DATASET_SPLIT)
    p.add_argument("--model", default=ZVENO_MODEL_ID)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    generate_synthetic_split(
        n_items=args.n_items,
        fewshot_k=args.fewshot_k,
        dataset=args.dataset,
        split=args.split,
        model_id=args.model,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

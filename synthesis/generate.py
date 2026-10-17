#!/usr/bin/env python3
"""Few-shot synthetic tip-2 generation with DeepSeek via Zveno."""

from __future__ import annotations

import argparse
import json
import random
import re
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
from eval.prompt import FIXED_INSTRUCTION, item_text
from eval.zveno_client import chat_completion, completion_text, has_zveno_key
from schemas.models import SyntheticItem
from synthesis.dictionary import UDARENIE_RU_DOC, fetch_senses
from synthesis.lemmas import sample_lemmas
from utils.paths import SYNTHESIS_DIR, ensure_result_dirs


def _fewshot_block(examples: list[dict]) -> str:
    chunks: list[str] = []
    for ex in examples:
        chunks.append(
            "Пример:\n"
            f"Текст: {item_text(ex)}\n"
            f"Варианты: {json.dumps(ex['options'], ensure_ascii=False)}\n"
            f"Ответ: {ex['answer']}\n"
            f"Пояснение: {ex.get('explanation', '')}"
        )
    return "\n\n".join(chunks)


def build_synthesis_messages(
    lemmas: list[str],
    senses: dict[str, list[str]],
    examples: list[dict],
) -> list[dict[str, str]]:
    sense_lines = []
    for lemma in lemmas:
        picked = senses.get(lemma) or ["значение не найдено"]
        sense_lines.append(f"- {lemma}: {picked[0]}")
    user = (
        f"Инструкция задания: {FIXED_INSTRUCTION}\n\n"
        f"{_fewshot_block(examples)}\n\n"
        "Сгенерируй одно новое задание того же типа. "
        "В тексте выдели ровно пять целевых слов разметкой **слово**. "
        "Пять вариантов ответа — краткие толкования без лишних примеров. "
        "Ответ — склейка верных номеров, например 35. "
        "Верни только один JSON-объект с полями text, options, answer, explanation. "
        "Без markdown и без текста вокруг JSON.\n\n"
        f"Леммы и выбранные значения:\n" + "\n".join(sense_lines)
    )
    return [
        {
            "role": "system",
            "content": (
                "Ты автор печатно корректных заданий на значение слова в контексте. "
                "Пиши по-русски. Не копируй примеры дословно."
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
        if isinstance(payload, dict) and "text" in payload and "options" in payload and "answer" in payload:
            return payload
    raise ValueError("no tip-2 JSON object in model output")


def _parse_item(raw: str, item_id: str, lemmas: list[str], senses: list[str], model: str) -> SyntheticItem:
    payload = _extract_json_object(raw)
    options = list(payload["options"])
    answer = str(payload["answer"])
    answer_list = [int(ch) for ch in answer if ch.isdigit()]
    return SyntheticItem(
        id=item_id,
        text=str(payload["text"]),
        options=options,
        answer=answer,
        answer_list=answer_list,
        explanation=str(payload.get("explanation", "")),
        lemmas=lemmas,
        senses=senses,
        source_model=model,
    )


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
    """Generate a local synthetic tip-2 split and document the HF push path."""
    ensure_result_dirs()
    rng = random.Random(seed)
    gold = load_rows(dataset, split)
    out_dir = SYNTHESIS_DIR / SYNTHESIS_SPLIT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    items: list[SyntheticItem] = []
    if not has_zveno_key() and stub_if_no_api:
        stub = SyntheticItem(
            id=f"synth-stub-0",
            text="В старом городе стоял высокий **замок**, а у двери скрипел **ключ**.",
            options=[
                "Здание-крепость",
                "Приспособление для запирания",
                "Источник воды",
                "Музыкальный знак",
                "Родник в лесу",
            ],
            answer="12",
            answer_list=[1, 2],
            explanation="Stub item for offline wiring checks.",
            lemmas=["замок", "ключ"],
            senses=["крепость", "для запирания"],
            source_model="stub",
        )
        items.append(stub)
    else:
        for idx in tqdm(range(n_items), desc="synthesis"):
            lemmas = sample_lemmas(5, rng=rng)
            sense_map: dict[str, list[str]] = {}
            chosen_senses: list[str] = []
            for lemma in lemmas:
                senses = fetch_senses(lemma, max_senses=5)
                sense_map[lemma] = senses
                chosen_senses.append(senses[0])
            examples = rng.sample(gold, k=min(fewshot_k, len(gold)))
            body = chat_completion(
                build_synthesis_messages(lemmas, sense_map, examples),
                model=model_id,
                temperature=0.3,
                max_tokens=4096,
            )
            message = (body.get("choices") or [{}])[0].get("message") or {}
            candidates = [
                message.get("content") or "",
                message.get("reasoning_content") or "",
                message.get("reasoning") or "",
                completion_text(body),
            ]
            item = None
            for raw in candidates:
                if not str(raw).strip():
                    continue
                item = _parse_item(
                    str(raw),
                    item_id=f"synth-{idx:05d}",
                    lemmas=lemmas,
                    senses=chosen_senses,
                    model=model_id,
                )
                break
            if item is None:
                raise RuntimeError(f"empty synthesis completion at idx={idx}")
            items.append(item)

    jsonl_path = out_dir / "train.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(item.model_dump_json() + "\n")

    meta = {
        "split": SYNTHESIS_SPLIT_NAME,
        "n_items": len(items),
        "model": model_id,
        "dictionary": "ru.wiktionary.org MediaWiki API",
        "udarenieru_note": UDARENIE_RU_DOC,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hf_push": (
            "python scripts/push_synthesis_hf.py "
            f"--local {jsonl_path} --repo pymlex/reshuege-ru-2026 "
            f"--config-name {SYNTHESIS_SPLIT_NAME}"
        ),
        "scale_note": (
            "Detector project: 8891 train abstracts -> 6396 DPO pairs. "
            "With 116 gold tip-2 items, first GRPO needs about 2k-4k synthetic "
            "source items (G=4-8). Pairwise DPO at K=2 needs about 6k-9k "
            "to approach the detector pair scale."
        ),
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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

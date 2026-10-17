#!/usr/bin/env python3
"""Parse a Решу ЕГЭ tip-2 DOCX scrape into JSONL for the HF dataset schema."""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from pathlib import Path

from docx import Document


HEADER_RE = re.compile(r"(?:^\s*\d+\.?\s*)?Тип\s*2\s*№\s*(\d+)", re.IGNORECASE)
OPTION_RE = re.compile(r"^(\d)\)\s*(.+)$", re.DOTALL)
ANSWER_RE = re.compile(r"^Ответ:\s*([0-9]+)\.?\s*$", re.IGNORECASE)


def clean_text(s: str) -> str:
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2009", " ")
    s = s.replace("\u200b", "").replace("\ufeff", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_items(paras: list[str]) -> list[tuple[str, list[str]]]:
    items: list[tuple[str, list[str]]] = []
    current_id: str | None = None
    buf: list[str] = []
    for p in paras:
        m = HEADER_RE.search(p)
        if m and (p.strip()[:1].isdigit() or "Тип 2" in p):
            if current_id is not None:
                items.append((current_id, buf))
            current_id = m.group(1)
            buf = []
            continue
        if current_id is not None:
            buf.append(p)
    if current_id is not None:
        items.append((current_id, buf))
    return items


def parse_item(item_id: str, lines: list[str]) -> dict:
    cleaned = [clean_text(x) for x in lines if clean_text(x)]
    instruction_parts: list[str] = []
    passage_parts: list[str] = []
    options: list[str | None] = [None] * 5
    explanation_parts: list[str] = []
    answer: str | None = None
    mode = "pre"

    for line in cleaned:
        if ANSWER_RE.match(line):
            if answer is None:
                answer = ANSWER_RE.match(line).group(1)
            continue
        if line.startswith("Пояснение"):
            mode = "explanation"
            rest = line
            if rest.lower().startswith("пояснение."):
                rest = rest[len("Пояснение.") :].strip()
            elif rest.lower().startswith("пояснение"):
                rest = rest[len("Пояснение") :].lstrip(".: ").strip()
            if rest:
                explanation_parts.append(rest)
            continue
        if mode == "explanation":
            explanation_parts.append(line)
            continue

        om = OPTION_RE.match(line)
        if om:
            idx = int(om.group(1)) - 1
            if 0 <= idx < 5:
                options[idx] = clean_text(om.group(2))
                mode = "options"
            continue

        if line.startswith("Прочитайте текст"):
            instruction_parts.append(line)
            mode = "passage"
            continue
        if line.startswith("В тексте выделено"):
            instruction_parts.append(line)
            mode = "after_passage"
            continue

        if mode in ("passage", "pre") and not line.startswith("Источник:"):
            if mode == "pre" and not instruction_parts:
                instruction_parts.append(line)
                mode = "passage"
            else:
                passage_parts.append(line)
            continue
        if mode == "after_passage":
            instruction_parts.append(line)

    if any(o is None for o in options):
        raise ValueError(f"item {item_id}: incomplete options")
    if answer is None or not answer.isdigit():
        raise ValueError(f"item {item_id}: missing or bad answer")
    answer_list = [int(ch) for ch in answer]
    if any(a < 1 or a > 5 for a in answer_list) or len(answer_list) != len(set(answer_list)):
        raise ValueError(f"item {item_id}: invalid answer digits {answer}")
    passage = "\n\n".join(passage_parts).strip()
    if not passage:
        raise ValueError(f"item {item_id}: empty passage")
    instruction = "\n".join(instruction_parts).strip()
    if not instruction:
        instruction = (
            "В тексте выделено пять слов. Укажите варианты ответов, в которых "
            "лексическое значение выделенного слова соответствует его значению "
            "в данном тексте. Запишите номера ответов."
        )
    return {
        "id": int(item_id),
        "task_type": 2,
        "instruction": instruction,
        "passage": passage,
        "options": list(options),
        "answer": answer,
        "answer_list": answer_list,
        "explanation": "\n".join(explanation_parts).strip(),
        "source": "reshu-ege-research",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = Document(args.docx)
    rows = []
    for item_id, lines in split_items([p.text for p in doc.paragraphs]):
        rows.append(parse_item(item_id, lines))
    jsonl = out_dir / "train.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = {
        "n_items": len(rows),
        "source": "reshu-ege-research",
        "task_type": 2,
        "answer_convention": "Concatenated correct option numbers, e.g. '35'.",
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"n": len(rows), "out": str(jsonl)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

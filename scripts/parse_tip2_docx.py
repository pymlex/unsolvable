#!/usr/bin/env python3
"""Parse a Решу ЕГЭ tip-2 DOCX scrape into JSONL, preserving bold lexical targets."""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


HEADER_RE = re.compile(r"(?:^\s*\d+\.?\s*)?Тип\s*2\s*№\s*(\d+)", re.IGNORECASE)
OPTION_RE = re.compile(r"^(\d)\)\s*(.+)$", re.DOTALL)
ANSWER_RE = re.compile(r"^Ответ:\s*([0-9]+)\.?\s*$", re.IGNORECASE)
INSTR_BOILERPLATE_RE = re.compile(
    r"^Прочитайте текст и выполните задан",
    re.IGNORECASE,
)
INSTR_STEM_RE = re.compile(
    r"^В тексте выделено пять слов",
    re.IGNORECASE,
)
EXAM_PART_RE = re.compile(
    r"задан(?:ие|ия)\s*[0-9０-９][0-9０-９\s\-–—−]*",
    re.IGNORECASE,
)

FIXED_INSTRUCTION = (
    "В тексте выделено пять слов. Укажите варианты ответов, в которых "
    "лексическое значение выделенного слова соответствует его значению "
    "в данном тексте. Запишите номера ответов."
)


def clean_text(s: str) -> str:
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2009", " ")
    s = s.replace("\u200b", "").replace("\ufeff", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def element_is_bold(r_elem) -> bool:
    r_pr = r_elem.find(qn("w:rPr"))
    if r_pr is None:
        return False
    b = r_pr.find(qn("w:b"))
    if b is None:
        return False
    val = b.get(qn("w:val"))
    return val is None or val not in ("0", "false", "False")


def run_element_text(r_elem) -> str:
    parts: list[str] = []
    for node in r_elem.iter():
        tag = node.tag
        if tag == qn("w:t") and node.text:
            parts.append(node.text)
        elif tag == qn("w:tab"):
            parts.append("\t")
        elif tag in (qn("w:br"), qn("w:cr")):
            parts.append("\n")
    return "".join(parts)


def paragraph_runs(paragraph) -> list[tuple[str, bool]]:
    """Collect runs including those nested under w:hyperlink."""
    out: list[tuple[str, bool]] = []
    for r_elem in paragraph._element.iter(qn("w:r")):
        text = run_element_text(r_elem)
        if not text:
            continue
        out.append((text, element_is_bold(r_elem)))
    return out


def merge_runs(runs: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
    if not runs:
        return []
    merged: list[tuple[str, bool]] = [runs[0]]
    for text, bold in runs[1:]:
        prev_text, prev_bold = merged[-1]
        if bold == prev_bold:
            merged[-1] = (prev_text + text, prev_bold)
        else:
            merged.append((text, bold))
    return merged


ELLIPSIS_TOKENS = {
    "...",
    "…",
    "[...]",
    "[…]",
    "<...>",
    "<…>",
    "[.]",
}


def is_lexical_highlight(text: str) -> bool:
    t = clean_text(text)
    if not t:
        return False
    if t in ELLIPSIS_TOKENS:
        return False
    if re.fullmatch(r"\d+", t):
        return False
    if re.fullmatch(r"[\[\]<>.…\s]+", t):
        return False
    if " " in t:
        return False
    if not re.search(r"[А-Яа-яЁёA-Za-z]", t):
        return False
    return True


def mark_passage(runs: list[tuple[str, bool]]) -> tuple[str, list[str]]:
    marked_parts: list[str] = []
    words: list[str] = []
    for text, bold in merge_runs(runs):
        if bold and is_lexical_highlight(text):
            inner = clean_text(text)
            words.append(inner)
            marked_parts.append(f"**{inner}**")
        else:
            marked_parts.append(text)
    passage = clean_text("".join(marked_parts))
    passage = re.sub(r"\*\*\s*\*\*", "", passage)
    return passage, words


def is_boilerplate_instruction(text: str) -> bool:
    t = clean_text(text)
    if not t:
        return False
    if INSTR_BOILERPLATE_RE.match(t):
        return True
    if INSTR_STEM_RE.match(t):
        return True
    return False


def strip_exam_part_mentions(text: str) -> str:
    return EXAM_PART_RE.sub("", text)


def split_items(
    paras: list[tuple[list[tuple[str, bool]], str]],
) -> list[tuple[str, list[tuple[list[tuple[str, bool]], str]]]]:
    items: list[tuple[str, list[tuple[list[tuple[str, bool]], str]]]] = []
    current_id: str | None = None
    buf: list[tuple[list[tuple[str, bool]], str]] = []
    for runs, plain in paras:
        m = HEADER_RE.search(plain)
        if m and (plain.strip()[:1].isdigit() or "Тип 2" in plain):
            if current_id is not None:
                items.append((current_id, buf))
            current_id = m.group(1)
            buf = []
            continue
        if current_id is not None:
            buf.append((runs, plain))
    if current_id is not None:
        items.append((current_id, buf))
    return items


def parse_item(
    item_id: str,
    lines: list[tuple[list[tuple[str, bool]], str]],
) -> dict:
    passage_blocks: list[str] = []
    highlighted: list[str] = []
    options: list[str | None] = [None] * 5
    explanation_parts: list[str] = []
    answer: str | None = None
    mode = "passage"

    for runs, plain in lines:
        text = clean_text(plain)
        if not text:
            continue
        if ANSWER_RE.match(text):
            if answer is None:
                answer = ANSWER_RE.match(text).group(1)
            continue
        if text.startswith("Пояснение"):
            mode = "explanation"
            rest = text
            if rest.lower().startswith("пояснение."):
                rest = rest[len("Пояснение.") :].strip()
            elif rest.lower().startswith("пояснение"):
                rest = rest[len("Пояснение") :].lstrip(".: ").strip()
            if rest:
                explanation_parts.append(rest)
            continue
        if mode == "explanation":
            if not text.startswith("Источник:"):
                explanation_parts.append(text)
            else:
                rest = text[len("Источник:") :].strip()
                if rest:
                    explanation_parts.append("Источник: " + rest)
            continue

        om = OPTION_RE.match(text)
        if om:
            idx = int(om.group(1)) - 1
            if 0 <= idx < 5:
                options[idx] = clean_text(om.group(2))
                mode = "options"
            continue

        if is_boilerplate_instruction(text):
            continue

        if text.startswith("Источник:"):
            continue

        if mode in ("passage", "pre"):
            marked, words = mark_passage(runs)
            marked = clean_text(strip_exam_part_mentions(marked))
            if marked:
                passage_blocks.append(marked)
                highlighted.extend(words)
            continue

    if any(o is None for o in options):
        raise ValueError(f"item {item_id}: incomplete options")
    if answer is None or not answer.isdigit():
        raise ValueError(f"item {item_id}: missing or bad answer")
    answer_list = [int(ch) for ch in answer]
    if any(a < 1 or a > 5 for a in answer_list) or len(answer_list) != len(set(answer_list)):
        raise ValueError(f"item {item_id}: invalid answer digits {answer}")
    passage = "\n\n".join(passage_blocks).strip()
    if not passage:
        raise ValueError(f"item {item_id}: empty passage")
    return {
        "id": int(item_id),
        "passage": passage,
        "highlighted_words": highlighted,
        "options": list(options),
        "answer": answer,
        "answer_list": answer_list,
        "explanation": "\n".join(explanation_parts).strip(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = Document(args.docx)
    paras: list[tuple[list[tuple[str, bool]], str]] = []
    for paragraph in doc.paragraphs:
        runs = paragraph_runs(paragraph)
        plain = "".join(t for t, _ in runs)
        paras.append((runs, plain))
    rows = []
    missing_bold: list[int] = []
    for item_id, lines in split_items(paras):
        row = parse_item(item_id, lines)
        if len(row["highlighted_words"]) < 1:
            missing_bold.append(row["id"])
        rows.append(row)
    jsonl = out_dir / "train.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_with = sum(1 for r in rows if r["highlighted_words"])
    hist = {}
    for r in rows:
        k = len(r["highlighted_words"])
        hist[k] = hist.get(k, 0) + 1
    info = {
        "n_items": len(rows),
        "dataset_id": "pymlex/reshuege-ru-2026",
        "fields": [
            "id",
            "passage",
            "highlighted_words",
            "options",
            "answer",
            "answer_list",
            "explanation",
        ],
        "card_note": (
            "All items are EGE Russian tip 2 from a Решу ЕГЭ 2026 research scrape. "
            "Fixed instruction and provenance live on the Hugging Face dataset card, "
            "not as row fields."
        ),
        "instruction_fixed": FIXED_INSTRUCTION,
        "highlight_markup": "**word** around each bold lexical target in passage",
        "answer_convention": "Concatenated correct option numbers, e.g. '35'.",
        "n_with_highlighted_words": n_with,
        "n_missing_highlighted_words": len(missing_bold),
        "missing_highlighted_ids": missing_bold,
        "highlighted_words_histogram": {str(k): hist[k] for k in sorted(hist)},
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "n": len(rows),
                "out": str(jsonl),
                "n_missing_bold": len(missing_bold),
                "missing_ids": missing_bold,
                "hist": hist,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

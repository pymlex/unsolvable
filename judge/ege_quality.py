"""Local structural checks only; no external semantic judge."""
from __future__ import annotations
import re
import unicodedata
from functools import lru_cache
from constants import LENGTH_MARGIN
from eval.prompt import item_text
CHECK_VERSION = "structural-lemma-v3"


@lru_cache(maxsize=1)
def _morph():
    from pymorphy3 import MorphAnalyzer
    return MorphAnalyzer()


@lru_cache(maxsize=8192)
def _analyses(word: str) -> frozenset:
    return frozenset((p.normal_form, p.tag.POS) for p in _morph().parse(word))


def same_lexeme(original: str, candidate: str) -> bool:
    if original.casefold() == candidate.casefold():
        return True
    left = re.findall(r"\w+(?:-\w+)*|[^\w\s]", original.casefold())
    right = re.findall(r"\w+(?:-\w+)*|[^\w\s]", candidate.casefold())
    return len(left) == len(right) and all(
        a == b or bool(_analyses(a) & _analyses(b))
        for a, b in zip(left, right, strict=True)
    )


def structural_error(row: dict, candidate: dict) -> str | None:
    text = candidate["text"]
    original = item_text(row)
    if candidate.get("truncated"):
        return "truncated"
    if not text.strip():
        return "empty"
    before = re.findall(r"\*\*([^*]+)\*\*", original)
    after = re.findall(r"\*\*([^*]+)\*\*", text)
    if len(before) != len(after) or not all(same_lexeme(a, b) for a, b in zip(before, after, strict=True)):
        return "highlight_changed"
    if text.count("**") != original.count("**"):
        return "broken_markup"
    if any(unicodedata.category(c).startswith("C") and c not in "\n\r\t" for c in text):
        return "invisible_character"
    if re.search(r"(?im)^\s*(?:ответ|ключ|пояснение|варианты|перефразированный текст)\s*:|</?think>|```", text):
        return "commentary_or_reasoning"
    if abs(len(text.split()) - len(original.split())) / max(1, len(original.split())) > LENGTH_MARGIN:
        return "length"
    return None


class QualityGate:
    def check(self, row: dict, candidate: dict) -> dict:
        error = structural_error(row, candidate)
        return {"valid": error is None, "reason": error or "structural_only", "judge_checked": False}

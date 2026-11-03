"""Local structural checks only; no external semantic judge."""
from __future__ import annotations
import re
import unicodedata
from constants import LENGTH_MARGIN
from eval.prompt import item_text
CHECK_VERSION = "structural-v2"


def structural_error(row: dict, candidate: dict) -> str | None:
    text = candidate["text"]
    original = item_text(row)
    if candidate.get("truncated"):
        return "truncated"
    if not text.strip():
        return "empty"
    if re.findall(r"\*\*([^*]+)\*\*", text) != re.findall(r"\*\*([^*]+)\*\*", original):
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

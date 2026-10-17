"""Russian paraphrase prompts for tip-2 text hardening."""

from __future__ import annotations


PARAPHRASE_PROMPT_TEMPLATE = (
    "Перефразируй следующий текст, сохранив смысл. "
    "Выделенные слова отмечай только разметкой **...**. "
    "Формы слов можно менять. Не добавляй комментарии и пояснения. "
    "Не меняй смысл. Текст: {original_text}"
)


def build_paraphrase_prompt(original_text: str) -> str:
    """Build a minimal Russian paraphrase instruction for one tip-2 text."""
    return PARAPHRASE_PROMPT_TEMPLATE.format(original_text=original_text)

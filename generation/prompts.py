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


def build_item_paraphrase_prompt(row: dict) -> str:
    """Give the author the explanation, but request only the rewritten passage."""
    from eval.prompt import format_options, item_text

    return (
        "Немного перефразируй предложения экзаменационного текста по русскому языку. "
        "Сохрани все факты, связи между предложениями и значения выделенных слов. "
        "Сохрани леммы выделенных слов, их порядок и разметку **слово**. "
        "Падежные формы можно менять, если этого требует грамматика нового предложения. "
        "Не заменяй выделенные слова синонимами и не подгоняй текст под неверные толкования. "
        "Не меняй правильность ни одного варианта ответа. Пиши грамотно и естественно. "
        "Словарные толкования, ключ и пояснение ниже нужны только для сохранения смысла. "
        "Верни ТОЛЬКО полный перефразированный текст: без вступления, вариантов, "
        "ключа, пояснения и собственных рассуждений.\n\n"
        f"Исходный текст:\n{item_text(row)}\n\n"
        f"Словарные толкования из задания (не переписывать):\n{format_options(row['options'])}\n\n"
        f"Ключ (не выводить): {row['answer']}\n"
        f"Пояснение (не выводить):\n{row.get('explanation', '')}\n\n"
        "Перефразированный текст:"
    )

"""Prompt templates for tip-2 exam solving."""

from __future__ import annotations


FIXED_INSTRUCTION = (
    "В тексте выделено пять слов. Укажите варианты ответов, в которых "
    "лексическое значение выделенного слова соответствует его значению "
    "в данном тексте. Запишите номера ответов."
)

SYSTEM_RU = (
    "Ты решаешь задание ЕГЭ по русскому языку типа 2. "
    "В тексте выделены слова. Нужно указать номера вариантов, "
    "в которых лексическое значение слова соответствует его значению в тексте. "
    "Ответ запиши только цифрами без пробелов и знаков препинания, "
    "как в бланке ЕГЭ: например 35 или 24. Ничего кроме цифр не пиши."
)


def item_text(row: dict) -> str:
    """Return the exam text field; HF rows store it as ``passage``."""
    if "text" in row and row["text"]:
        return str(row["text"])
    return str(row.get("passage", ""))


def format_options(options: list[str]) -> str:
    """Format option glosses with ``1.`` labels."""
    return "\n".join(f"{i}. {opt}" for i, opt in enumerate(options, start=1))


def build_user_prompt(row: dict) -> str:
    """Build the user exam prompt for one tip-2 item."""
    text = item_text(row)
    opt_lines = format_options(list(row["options"]))
    return (
        f"{FIXED_INSTRUCTION}\n\n"
        f"Текст:\n{text}\n\n"
        f"Варианты:\n{opt_lines}\n\n"
        "Ответ (только цифры):"
    )


def build_messages(row: dict) -> list[dict[str, str]]:
    """Build chat messages for SLM or API solvers."""
    return [
        {"role": "system", "content": SYSTEM_RU},
        {"role": "user", "content": build_user_prompt(row)},
    ]


def build_exam_prompt_string(row: dict) -> str:
    """Flatten chat messages into a single string for logprob scoring."""
    messages = build_messages(row)
    return messages[0]["content"] + "\n\n" + messages[1]["content"]

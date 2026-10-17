"""Prompt templates for EGE tip-2 zero-edit baseline."""

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


def build_user_prompt(row: dict) -> str:
    options = row["options"]
    opt_lines = "\n".join(f"{i}) {opt}" for i, opt in enumerate(options, start=1))
    return (
        f"{FIXED_INSTRUCTION}\n\n"
        f"Текст:\n{row['passage']}\n\n"
        f"Варианты:\n{opt_lines}\n\n"
        "Ответ (только цифры):"
    )


def build_messages(row: dict) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_RU},
        {"role": "user", "content": build_user_prompt(row)},
    ]

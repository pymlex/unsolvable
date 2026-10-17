"""LLM-as-judge via Zveno DeepSeek: did the paraphrase break the tip-2 task?"""

from __future__ import annotations

import json
import re

from constants import ZVENO_MODEL_ID
from eval.zveno_client import chat_completion, completion_text
from schemas.models import JudgeVerdict


JUDGE_SYSTEM = (
    "Ты проверяешь, сохраняет ли парафраз русского экзаменационного текста "
    "тот же правильный ответ по смыслу. Отвечай строго JSON вида "
    '{"broken": true|false, "reason": "..."}.'
)


def build_judge_messages(
    original_text: str,
    paraphrase: str,
    answer: str,
    explanation: str,
) -> list[dict[str, str]]:
    """Build judge chat messages for one paraphrase candidate."""
    user = (
        "Исходный текст:\n"
        f"{original_text}\n\n"
        "Парафраз:\n"
        f"{paraphrase}\n\n"
        f"Правильный ответ: {answer}\n"
        f"Пояснение: {explanation}\n\n"
        "Вопрос: парафраз сломал задание так, что прежний ответ уже неверен "
        "или текст стал грамматически непригоден для печати? "
        "broken=true если задание сломано."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _parse_verdict(raw: str) -> JudgeVerdict:
    text = raw.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    payload = json.loads(text)
    return JudgeVerdict(
        broken=bool(payload.get("broken")),
        reason=str(payload.get("reason", "")),
        raw=raw,
    )


def judge_paraphrase(
    original_text: str,
    paraphrase: str,
    answer: str,
    explanation: str,
    *,
    model: str | None = None,
) -> JudgeVerdict:
    """Ask DeepSeek whether a paraphrase breaks the tip-2 item."""
    body = chat_completion(
        build_judge_messages(original_text, paraphrase, answer, explanation),
        model=model or ZVENO_MODEL_ID,
        temperature=0.0,
        max_tokens=256,
    )
    raw = completion_text(body)
    return _parse_verdict(raw)

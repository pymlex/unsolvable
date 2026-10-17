"""Thin OpenAI-compatible client for Zveno chat completions."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from constants import ZVENO_BASE_URL, ZVENO_MODEL_ID
from utils.config_loader import require_env


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout_sec: float = 180.0,
) -> dict[str, Any]:
    """Call Zveno ``/chat/completions`` and return the parsed JSON body."""
    api_key = require_env("ZVENOAI_API_KEY")
    model_id = model or ZVENO_MODEL_ID
    url = ZVENO_BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        body = json.loads(response.read().decode("utf-8"))
    latency = time.perf_counter() - started
    body["_latency_sec"] = latency
    return body


def completion_text(body: dict[str, Any]) -> str:
    """Extract assistant text from a chat completion response."""
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        text = "".join(parts).strip()
    elif content is None:
        text = ""
    else:
        text = str(content).strip()
    if text:
        return text
    for key in ("reasoning_content", "reasoning"):
        alt = message.get(key)
        if isinstance(alt, str) and alt.strip():
            return alt.strip()
    return ""


def usage_cost_usd(body: dict[str, Any], prompt_per_1k: float = 0.0, completion_per_1k: float = 0.0) -> float | None:
    """Estimate USD cost from usage tokens when prices are known."""
    usage = body.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None or completion_tokens is None:
        return None
    if prompt_per_1k <= 0 and completion_per_1k <= 0:
        return None
    return (prompt_tokens / 1000.0) * prompt_per_1k + (completion_tokens / 1000.0) * completion_per_1k


def has_zveno_key() -> bool:
    return bool(os.environ.get("ZVENOAI_API_KEY"))

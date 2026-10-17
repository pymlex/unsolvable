"""Compute answer score \(s_m(y^\\star\\mid x)\) under the exam prompt."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from eval.metrics import gold_normalize
from eval.prompt import build_messages


def render_exam_prompt(tokenizer: PreTrainedTokenizerBase, row: dict) -> str:
    """Render tip-2 chat messages with the model chat template."""
    messages = build_messages(row)
    if not (hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template):
        return messages[0]["content"] + "\n\n" + messages[1]["content"]
    import inspect

    kwargs = {"tokenize": False, "add_generation_prompt": True}
    params = inspect.signature(tokenizer.apply_chat_template).parameters
    if "enable_thinking" in params:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def answer_token_logprob_sum(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    row: dict,
    answer: str | None = None,
) -> float:
    """Return the sum of log-probabilities of gold answer tokens given the exam prompt.

    This is \(s_m(y^\star\mid x)\): conditional log-likelihood of the digit answer
    string under the same tip-2 prompt used at eval time.
    """
    gold = gold_normalize(answer if answer is not None else row["answer"])
    if not gold:
        return float("-inf")

    prompt = render_exam_prompt(tokenizer, row)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    answer_ids = tokenizer(gold, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    prompt_ids = {k: v.to(device) for k, v in prompt_ids.items()}
    answer_ids = {k: v.to(device) for k, v in answer_ids.items()}

    input_ids = torch.cat([prompt_ids["input_ids"], answer_ids["input_ids"]], dim=1)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    prompt_len = prompt_ids["input_ids"].shape[-1]
    answer_len = answer_ids["input_ids"].shape[-1]
    if answer_len == 0:
        return float("-inf")

    shift_logits = logits[:, prompt_len - 1 : prompt_len + answer_len - 1, :]
    target = answer_ids["input_ids"]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logprobs = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    return float(token_logprobs.sum().item())


def score_rows(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    rows: list[dict],
) -> list[float]:
    """Score gold answers for a list of tip-2 rows."""
    return [answer_token_logprob_sum(model, tokenizer, row) for row in rows]

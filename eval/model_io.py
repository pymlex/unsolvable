"""Local SLM load and greedy generation helpers."""

from __future__ import annotations

from eval.prompt import build_messages


def load_slm(model_id: str, device: str | None = None):
    """Load a causal LM and tokenizer for tip-2 solving or paraphrase."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_chat(
    model,
    tokenizer,
    device: str,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float | None = None,
) -> str:
    """Generate a completion from chat messages."""
    import torch

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        import inspect

        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        params = inspect.signature(tokenizer.apply_chat_template).parameters
        if "enable_thinking" in params:
            kwargs["enable_thinking"] = False
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    else:
        prompt = messages[0]["content"] + "\n\n" + messages[1]["content"]

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device if hasattr(model, "device") else device) for k, v in inputs.items()}
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature if temperature is not None else 0.8
    else:
        gen_kwargs["temperature"] = None
        gen_kwargs["top_p"] = None

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    gen = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def generate_tip2_answer(
    model,
    tokenizer,
    device: str,
    row: dict,
    max_new_tokens: int = 32,
) -> str:
    """Greedy tip-2 solve for one item."""
    return generate_chat(
        model,
        tokenizer,
        device,
        build_messages(row),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )


def _row_prompt(tokenizer, row: dict) -> str:
    messages = build_messages(row)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        import inspect

        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        params = inspect.signature(tokenizer.apply_chat_template).parameters
        if "enable_thinking" in params:
            kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(messages, **kwargs)
    return messages[0]["content"] + "\n\n" + messages[1]["content"]


def generate_tip2_answers_batch(
    model,
    tokenizer,
    device: str,
    rows: list[dict],
    max_new_tokens: int = 32,
) -> list[str]:
    """Greedy tip-2 solve for a batch of items (left-padded)."""
    import torch

    if not rows:
        return []
    if len(rows) == 1:
        return [generate_tip2_answer(model, tokenizer, device, rows[0], max_new_tokens)]

    prompts = [_row_prompt(tokenizer, row) for row in rows]
    pad_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    target = model.device if hasattr(model, "device") else device
    inputs = {k: v.to(target) for k, v in inputs.items()}
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "temperature": None,
        "top_p": None,
    }
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    tokenizer.padding_side = pad_side
    prompt_len = inputs["input_ids"].shape[-1]
    texts = []
    for seq in out:
        gen = seq[prompt_len:]
        texts.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    return texts

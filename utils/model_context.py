"""Use the model's native context window; never silently truncate an input."""


def context_window(model, tokenizer) -> int:
    config = model.config
    text_config = getattr(config, "text_config", config)
    limits = [getattr(text_config, "max_position_embeddings", None),
              getattr(text_config, "n_positions", None),
              getattr(tokenizer, "model_max_length", None)]
    limits = [int(n) for n in limits if isinstance(n, (int, float)) and 0 < n < 100_000_000]
    if not limits:
        raise ValueError("Model context window is unknown; refusing an implicit generation/truncation limit")
    return min(limits)


def generation_bounds(model, tokenizer, prompt_length: int, max_new_tokens=None) -> dict:
    limit = context_window(model, tokenizer)
    if prompt_length >= limit:
        raise ValueError(f"Complete prompt has {prompt_length} tokens, exceeding the native context window {limit}")
    bounds = ({"max_new_tokens": max_new_tokens} if max_new_tokens is not None
              else {"max_length": limit, "max_new_tokens": None})
    # Trainer may replace Qwen's document EOS by its chat EOS. Both are natural
    # terminators; keep both so generation cannot continue after a finished turn.
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    ids = set(eos if isinstance(eos, list) else [eos])
    ids.add(getattr(tokenizer, "eos_token_id", None))
    special = getattr(tokenizer, "all_special_tokens", [])
    for token in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        if token in special:
            ids.add(tokenizer.convert_tokens_to_ids(token))
    ids.discard(None)
    if ids:
        bounds["eos_token_id"] = sorted(ids)
    return bounds

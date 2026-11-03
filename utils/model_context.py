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
    if max_new_tokens is not None:
        return {"max_new_tokens": max_new_tokens}  # Explicit bounded unit-test/legacy call.
    return {"max_length": limit, "max_new_tokens": None}

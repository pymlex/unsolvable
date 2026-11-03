"""Two independent base-policy samples per item, single greedy sample at eval."""

from __future__ import annotations

from constants import GENERATION_BATCH_SIZE, GENERATION_TEMPERATURE
from eval.model_io import apply_chat_template_no_think
from generation.prompts import build_item_paraphrase_prompt
from utils.model_context import generation_bounds


def generate_item_paraphrases(model, tokenizer, rows: list[dict], *, num_samples=2,
                             do_sample=True, max_new_tokens=None,
                             batch_size=GENERATION_BATCH_SIZE, temperature=GENERATION_TEMPERATURE,
                             top_p=0.9, repetition_penalty=1.0, progress_dir=None) -> list[list[dict]]:
    import torch

    if num_samples < 1 or batch_size < 1:
        raise ValueError("num_samples and batch_size must be positive")
    groups = [[] for _ in rows]
    was_training = model.training
    pad_side = tokenizer.padding_side
    model.eval()
    tokenizer.padding_side = "left"
    try:
        # Each pass calls generate independently (no shared beam / duplicated output).
        for sample_pass in range(num_samples):
            for start in range(0, len(rows), batch_size):
                chunk = rows[start:start + batch_size]
                prompts = [apply_chat_template_no_think(tokenizer, [{"role": "user", "content": build_item_paraphrase_prompt(r)}]) for r in chunk]
                inputs = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False).to(model.device)
                kwargs = {**generation_bounds(model, tokenizer, inputs["input_ids"].shape[1], max_new_tokens), "do_sample": do_sample,
                          "pad_token_id": tokenizer.pad_token_id, "use_cache": True,
                          "repetition_penalty": repetition_penalty}
                if do_sample:
                    kwargs.update(temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty)
                else:
                    kwargs.update(temperature=None, top_p=None, top_k=None)
                if progress_dir is not None:
                    from generation.progress import GenerationProgress, repetition_stopping_criteria
                    eos = kwargs["eos_token_id"]
                    kwargs["streamer"] = GenerationProgress(
                        tokenizer, chunk, eos if isinstance(eos, list) else [eos], progress_dir,
                        sample_pass=sample_pass + 1,
                        sampling={"do_sample": do_sample, "temperature": temperature if do_sample else None,
                                  "top_p": top_p if do_sample else None,
                                  "repetition_penalty": repetition_penalty, "seed": torch.initial_seed()})
                    kwargs["stopping_criteria"] = repetition_stopping_criteria(kwargs["streamer"])
                with torch.no_grad():
                    output = model.generate(**inputs, **kwargs)
                offset = inputs["input_ids"].shape[1]
                eos = kwargs.get("eos_token_id") or model.generation_config.eos_token_id or tokenizer.eos_token_id
                eos_ids = set(eos if isinstance(eos, list) else [eos])
                for i, sequence in enumerate(output):
                    tokens = sequence[offset:].tolist()
                    candidate = {
                        "text": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                        "truncated": not any(t in eos_ids for t in tokens),
                    }
                    if progress_dir is not None:
                        candidate["generation_event_id"] = f"{kwargs['streamer'].run_id}:{chunk[i]['id']}:{sample_pass+1}"
                        item_id = str(chunk[i]["id"])
                        if item_id in kwargs["streamer"].abandoned_ids:
                            candidate["truncated"] = True
                            candidate["failure_reason"] = kwargs["streamer"].invalid_sections.get(item_id, "repeated_paragraph_loop")
                    groups[start + i].append(candidate)
    finally:
        tokenizer.padding_side = pad_side
        model.train(was_training)
    return groups

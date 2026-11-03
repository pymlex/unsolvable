"""Observe generation without stopping, shortening or modifying any completion."""

import json
import os
import time
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path

from utils.artifacts import write_json, fingerprint


class GenerationProgress:
    def __init__(self, tokenizer, rows, eos_ids, directory: Path, *, sample_pass, sampling):
        self.tokenizer, self.rows, self.eos_ids = tokenizer, rows, set(eos_ids)
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.last_report = 0
        self.run_id = uuid.uuid4().hex
        self.sample_pass, self.sampling = sample_pass, sampling
        self.tokens = [[] for _ in rows]
        self.finished = set()
        self.loop_ids = set()
        self.invalid_sections = {}
        self.first = True

    def put(self, value):
        if self.first:
            self.first = False  # generate() first sends the complete input prompt.
            return
        values = value.tolist()
        for i, token in enumerate(values):
            item_id = str(self.rows[i]["id"])
            if i in self.finished or item_id in self.abandoned_ids:
                continue
            self.tokens[i].append(token)
            for size in (*range(1, 97), 128, 192, 256, 384, 512):
                tail = self.tokens[i]
                repeats = max(3, (96 + size - 1) // size)
                if len(tail) >= size * repeats and tail[-size*repeats:] == tail[-size:] * repeats:
                    self.loop_ids.add(str(self.rows[i]["id"]))
                    break
            if len(self.tokens[i]) % 16 == 0 or token in self.eos_ids:
                decoded = self.tokenizer.decode(self.tokens[i], skip_special_tokens=True)
                if any(section in decoded for section in (
                    "Ключ (не выводить):", "Пояснение (не выводить):",
                    "Словарные толкования из задания (не переписывать):",
                )):
                    self.invalid_sections[item_id] = "prompt_metadata_in_output"
                if re.search(r"(?im)^\s*(?:ключ|пояснение|словарные толкования(?: из задания)?|перефразированный текст|варианты ответа)\s*(?:\([^\n)]*\))?\s*:", decoded):
                    self.invalid_sections[item_id] = "prompt_metadata_in_output"
                # Once this existing structural condition is violated, appending
                # tokens cannot make the output admissible. Reject the attempt;
                # never shorten it into a purportedly complete training passage.
                from constants import LENGTH_MARGIN
                original = self.rows[i].get("passage") or self.rows[i].get("text") or ""
                original_words = len(original.split())
                if original_words and len(decoded.split()) > original_words * (1 + LENGTH_MARGIN):
                    self.invalid_sections[item_id] = "length_cannot_pass_structural_check"
            if item_id in self.abandoned_ids:
                continue
            if token in self.eos_ids:
                self.finished.add(i)
                record = {"run_id": self.run_id, "item_id": str(self.rows[i]["id"]),
                          "source_fingerprint": fingerprint(self.rows[i]),
                          "sample_pass": self.sample_pass, "sampling": self.sampling,
                          "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                          "n_tokens": len(self.tokens[i]), "truncated": False,
                          "text": self.tokenizer.decode(self.tokens[i], skip_special_tokens=True).strip()}
                with (self.directory / "generation_samples.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
        if time.monotonic() - self.last_report >= 10:
            self.report()

    def report(self):
        elapsed = time.monotonic() - self.started
        lengths = [len(tokens) for tokens in self.tokens]
        progress = {"run_id": self.run_id, "sample_pass": self.sample_pass,
                    "batch_size": len(self.rows), "n_finished": len(self.finished),
                    "n_active": len(self.rows) - len(self.finished) - len(self.abandoned_ids), "elapsed_sec": round(elapsed, 1),
                    "max_generated_tokens": max(lengths, default=0),
                    "total_generated_tokens": sum(lengths),
                    "tokens_per_sec": round(sum(lengths) / max(elapsed, .001), 1),
                    "active_item_ids": [str(row["id"]) for i, row in enumerate(self.rows)
                                        if i not in self.finished and str(row["id"]) not in self.abandoned_ids],
                    "repetition_loop_item_ids": sorted(self.loop_ids),
                    "invalid_section_item_ids": self.invalid_sections,
                    "active_tail_preview": {str(row["id"]): self.tokenizer.decode(self.tokens[i][-96:], skip_special_tokens=True)
                                            for i, row in enumerate(self.rows)
                                            if i not in self.finished and str(row["id"]) not in self.abandoned_ids},
                    "sampling": self.sampling}
        write_json(self.directory / "generation_progress.json", progress)
        print(f"[generate] pass={self.sample_pass} EOS={len(self.finished)}/{len(self.rows)} "
              f"max_tokens_so_far={progress['max_generated_tokens']} elapsed={elapsed:.0f}s "
              f"tokens/s={progress['tokens_per_sec']}", flush=True)
        self.last_report = time.monotonic()

    def end(self):
        self.report()

    @property
    def abandoned_ids(self):
        return self.loop_ids | self.invalid_sections.keys()


def repetition_stopping_criteria(progress):
    from transformers import StoppingCriteria, StoppingCriteriaList
    import torch
    class StopRepeatedParagraph(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return torch.tensor([str(row["id"]) in progress.abandoned_ids for row in progress.rows],
                                dtype=torch.bool, device=input_ids.device)
    return StoppingCriteriaList([StopRepeatedParagraph()])

"""Evaluate rewritten real EGE items with an immutable no-thinking solver."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from constants import SOLVER_MODEL_ID
from eval.ege_data import load_prepared
from eval.metrics import answer_jaccard, binomial_wilson_ci, exact_match, normalize_answer
from eval.model_io import load_slm
from eval.prompt import item_text
from generation.ege_paraphrase import generate_item_paraphrases
from judge.ege_quality import QualityGate
from scoring.frozen_solver import FrozenSolver
from utils.artifacts import fingerprint, write_json
from utils.paths import MONITORING_DIR


def evaluate_items(rows, generate, solve, gate, *, baseline_raw=None) -> dict:
    """Invalid rewrites fall back to originals; they never count as successful attacks."""
    if not rows:
        raise ValueError("Cannot evaluate an empty source")
    baseline_raw = solve(rows) if baseline_raw is None else baseline_raw
    groups = generate(rows)
    if len(groups) != len(rows) or len(baseline_raw) != len(rows) or any(len(g) != 1 for g in groups):
        raise ValueError("Expected exactly one eval rewrite and one baseline answer per item")
    candidates = [g[0] for g in groups]
    verdicts = [gate.check(row, candidate) for row, candidate in zip(rows, candidates, strict=True)]
    accepted = [i for i, v in enumerate(verdicts) if v["valid"]]
    edited_rows = [{**rows[i], "text": candidates[i]["text"], "passage": candidates[i]["text"]} for i in accepted]
    predictions = solve(edited_rows) if edited_rows else []
    if len(predictions) != len(accepted):
        raise ValueError("Solver returned an unexpected number of answers")
    raws = list(baseline_raw)
    for i, raw in zip(accepted, predictions, strict=True):
        raws[i] = raw
    records = []
    for i, row in enumerate(rows):
        base_ok = exact_match(baseline_raw[i], row["answer"])
        ok = exact_match(raws[i], row["answer"])
        records.append({"id": str(row["id"]), "gold": str(row["answer"]),
                        "original_text": item_text(row), "paraphrase": candidates[i]["text"],
                        "truncated": candidates[i].get("truncated", False), "quality": verdicts[i],
                        "changed": candidates[i]["text"] != item_text(row),
                        "baseline_raw": baseline_raw[i], "baseline_correct": base_ok,
                        "raw": raws[i], "pred": normalize_answer(raws[i]), "correct": ok,
                        "jaccard": answer_jaccard(raws[i], row["answer"]),
                        "solver_format_valid": bool(re.fullmatch(r"[1-5]{1,5}", raws[i].strip())),
                        "attack_success": verdicts[i]["valid"] and base_ok and not ok})
    n = len(rows)
    baseline_correct = sum(r["baseline_correct"] for r in records)
    successes = sum(r["attack_success"] for r in records)
    accuracy, lo, hi = binomial_wilson_ci(sum(r["correct"] for r in records), n)
    valid_only = [r for r in records if r["quality"]["valid"]]
    summary = {"n_items": n, "n_valid": len(accepted), "valid_rate": len(accepted) / n,
               "quality_mode": "structural_only",
               "baseline_accuracy": baseline_correct / n, "accuracy": accuracy,
               "accuracy_drop": baseline_correct / n - accuracy, "wilson_95_ci": [lo, hi],
               "mean_jaccard": sum(r["jaccard"] for r in records) / n,
               "valid_only_accuracy": sum(r["correct"] for r in valid_only) / len(valid_only) if valid_only else None,
               "n_baseline_correct": baseline_correct, "n_attack_success": successes,
               "conditional_asr": successes / baseline_correct if baseline_correct else None,
               "n_format_invalid": sum(not r["solver_format_valid"] for r in valid_only),
               "n_changed": sum(r["changed"] for r in records),
               "mean_gold_logprob": None}
    return {"summary": summary, "items": records}


class EgeEvaluator:
    def __init__(self, tokenizer, solver: FrozenSolver, *, output_dir: Path,
                 reset_history=False):
        self.tokenizer, self.solver, self.output_dir = tokenizer, solver, output_dir
        _, self.rows, self.manifest = load_prepared()
        self.gate = QualityGate()
        self.baseline_raw = None
        self.history_path = output_dir / "validation_history.json"
        self.history = [] if reset_history or not self.history_path.exists() else json.loads(self.history_path.read_text(encoding="utf-8"))
        self.identity = {"dataset": self.manifest, "solver": solver.model_id,
                         "quality_mode": "structural_only"}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_path = output_dir / "eval_protocol.json"
        if self.run_path.exists() and json.loads(self.run_path.read_text(encoding="utf-8")) != self.identity:
            raise ValueError("Evaluation protocol changed; use a new RESULTS_DIR")
        write_json(self.run_path, self.identity)

    def run(self, model, *, step: int, epoch=0.0) -> dict:
        if self.baseline_raw is None:
            self.baseline_raw = self.solver.solve(self.rows)
        payload = evaluate_items(
            self.rows,
            lambda rows: generate_item_paraphrases(model, self.tokenizer, rows, num_samples=1, do_sample=False,
                                                  progress_dir=self.output_dir / "generation"),
            self.solver.solve, self.gate, baseline_raw=self.baseline_raw,
        )
        # A differentiable-like signal analogous to mean detector logit, measured
        # on valid rewritten tasks only. Frozen solver never receives explanation.
        scores = [self.solver.score(row, record["paraphrase"])
                  for row, record in zip(self.rows, payload["items"], strict=True) if record["quality"]["valid"]]
        summary = payload["summary"]
        summary["mean_gold_logprob"] = sum(scores) / len(scores) if scores else None
        summary.update(step=int(step), epoch=float(epoch), solver=self.solver.model_id,
                       dataset=self.manifest["dataset"], config=self.manifest["test"]["config"], split="test",
                       source_revision=self.manifest["revision"], source_fingerprint=self.manifest["test"]["fingerprint"])
        write_json(self.output_dir / f"monitor_step_{step:06d}.json", payload)
        self.history = [h for h in self.history if h["step"] != step] + [summary]
        self.history.sort(key=lambda h: h["step"])
        write_json(self.history_path, self.history)
        fields = [k for k in summary if k != "wilson_95_ci"]
        with (self.output_dir / "validation_history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.history)
        print(f"[ege-eval] step={step} n={summary['n_items']} valid={summary['n_valid']} "
              f"accuracy={summary['accuracy']:.4f} drop={summary['accuracy_drop']:.4f} "
              f"ASR={summary['conditional_asr']}", flush=True)
        return payload


def run_evaluation(*, checkpoint: str | None = None, device=None) -> Path:
    from training.dpo_train import FINAL_DIR

    checkpoint = checkpoint or str(FINAL_DIR)
    path = Path(checkpoint)
    if (path / "adapter_config.json").exists():
        from peft import PeftModel
        config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        base_id = config["base_model_name_or_path"]
        base, tokenizer, _ = load_slm(base_id, device)
        model = PeftModel.from_pretrained(base, checkpoint)
    else:
        model, tokenizer, _ = load_slm(checkpoint, device)
        base_id = checkpoint
    if hasattr(model, "disable_adapter") and base_id == SOLVER_MODEL_ID:
        solver = FrozenSolver(model, tokenizer, model_id=SOLVER_MODEL_ID, disable_adapter=True)
    else:
        solver_model, solver_tokenizer, _ = load_slm(SOLVER_MODEL_ID, device)
        solver = FrozenSolver(solver_model, solver_tokenizer, model_id=SOLVER_MODEL_ID)
    output = MONITORING_DIR / "standalone" / fingerprint(checkpoint)[:12]
    evaluator = EgeEvaluator(tokenizer, solver, output_dir=output)
    evaluator.run(model, step=0)
    return output / "monitor_step_000000.json"

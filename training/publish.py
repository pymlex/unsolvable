"""Publish the converged LoRA adapter and its complete monitoring artifacts."""

import json
import os
from pathlib import Path
from huggingface_hub import HfApi
from utils.artifacts import write_json


def publish_converged_adapter(adapter_dir: Path, monitoring_dir: Path, plan: dict):
    convergence = json.loads((monitoring_dir / "convergence.json").read_text(encoding="utf-8"))
    if not convergence["converged"]:
        print("[HF] Training budget ended without measured loss convergence; adapter retained locally", flush=True)
        return None
    repo = os.getenv("HF_MODEL_REPO", "pymlex/unsolvable-qwen3.5-2b-dpo")
    validation = json.loads((monitoring_dir / "validation_history.json").read_text(encoding="utf-8"))
    card = f"""---
base_model: {plan['model']}
library_name: peft
language:
- ru
tags:
- dpo
- lora
datasets:
- pymlex/reshuege-ru-2026
---
# Unsolvable: Qwen LoRA DPO

LoRA paraphraser trained on 500 synthetic Russian EGE items. Each selected pair
has a frozen-solver gold-answer log-probability gap strictly greater than 0.5 nats.
The generator receives dictionary definitions and explanation and returns the passage.

Training ended on a measured **training loss plateau**, not proof of improved
attack quality. Every monitoring evaluation uses all 116 real Reshu EGE items;
repeated monitoring means this is not an independent final generalization test.

Final monitoring results:
```json
{json.dumps(validation[-1], ensure_ascii=False, indent=2)}
```

See `monitoring/` for per-step loss, rewards, 116-item eval outputs, convergence
windows and run configuration. Source revision: `{plan['revision']}`.
"""
    (adapter_dir / "README.md").write_text(card, encoding="utf-8")
    write_json(monitoring_dir / "run_config.json", plan)
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=str(adapter_dir), commit_message="Upload converged Qwen LoRA DPO adapter")
    result = api.upload_folder(repo_id=repo, folder_path=str(monitoring_dir), path_in_repo="monitoring",
                               commit_message="Upload training loss and all 116-item monitoring evaluations")
    url = f"https://huggingface.co/{repo}"
    write_json(monitoring_dir / "hf_model_publication.json", {"url": url, "commit": result.oid})
    print(f"[HF] Published converged adapter and metrics: {url}", flush=True)
    return url

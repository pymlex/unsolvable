# Фреймворк генерации состязательных заданий, устойчивых к решению SLM

A Framework for Generating Adversarial Items Robust to SLM Solvers.

Evaluation target: Russian Unified State Exam Russian language Type 2 items. Gold set: [pymlex/reshuege-ru-2026](https://huggingface.co/datasets/pymlex/reshuege-ru-2026) (116 items). In code and docs the stem field is **text**. Highlights use `**...**`. Option labels use `1.` not `1)`.

Local generator and frozen solver on Clore GPU: `Qwen/Qwen3.5-2B`. DeepSeek generated the 500 synthetic training sources; the DPO pipeline requires no external API.

## DPO workflow

The current training pipeline mirrors [ai-text-detector-tricking](https://github.com/pymlex/ai-text-detector-tricking).
The small model writes two independent careful paraphrases using the dictionary
definitions and `explanation`. A frozen no-thinking solver ranks them by gold
answer log probability, then LoRA DPO trains the paraphraser.

- Training: HF config `train-deepseek-v4-flash`, split `train`, **500** synthetic items.
- Preferences: exactly **500 pairs**, all retained; published to HF config `dpo-qwen3.5-2b/train`.
- Evaluation: HF config `test-reshuege`, split `test`, **all 116** real items every time.
- DeepSeek is the source of the synthetic items; the DPO run makes **no API calls**.
- Eval runs before training, every 20 optimizer steps, and at the final step.

```bash
git clone https://github.com/pymlex/unsolvable.git
cd unsolvable
bash scripts/install_ubuntu_jupyter.sh
source .venv/bin/activate
python main.py --step prepare
python main.py --step preferences
python main.py --step train --dry-run
bash scripts/run_dpo.sh
```

`bash scripts/run_all.sh` runs prepare, preferences and training. Resume with
`bash scripts/run_dpo.sh --resume`. Results, per-item evals, histories and plots
are saved under `results/`; the final LoRA adapter is in
`results/checkpoints/dpo/final`.

See [docs/dpo.md](docs/dpo.md) for the data contract, scoring direction,
monitoring metrics, GPU setup, artifacts and tests. Historical benchmark results
remain in `results/benchmark/`. GRPO utilities are legacy scaffolding; DPO is
the supported training path.

## Licence

GPL-3.0. See [LICENSE](LICENSE).

```bibtex
@misc{unsolvable-2026,
  title={A Framework for Generating Adversarial Items Robust to {SLM} Solvers},
  author={{Фреймворк генерации состязательных заданий, устойчивых к решению SLM}},
  howpublished={https://github.com/pymlex/unsolvable},
  year={2026}
}

@misc{reshuege-ru-2026,
  title={reshuege-ru-2026: {EGE} Russian Type 2 items from Reshu {EGE}},
  author={{pymlex}},
  howpublished={https://huggingface.co/datasets/pymlex/reshuege-ru-2026},
  year={2026}
}
```

The project is under GPL-3.0 license.

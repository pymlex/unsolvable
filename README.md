# Фреймворк генерации состязательных заданий, устойчивых к решению SLM

A Framework for Generating Adversarial Items Robust to SLM Solvers.

Evaluation target: Russian Unified State Exam Russian language Type 2 items. Dataset: [pymlex/reshuege-ru-2026](https://huggingface.co/datasets/pymlex/reshuege-ru-2026) (116 gold items). In code and docs the stem field is **text** (HF column may still be named `passage`). Highlights use `**...**`. Option labels use `1.` not `1)`.

## Install on Ubuntu Jupyter (Clore)

```bash
bash scripts/install_ubuntu_jupyter.sh
cp .env.example .env   # if not already created
# fill HF_TOKEN, ZVENOAI_API_KEY, GITHUB_TOKEN
```

Default local SLM: `Qwen/Qwen3.5-2B` (`SLM_MODEL_ID`). Judge and strong solver: `deepseek/deepseek-v4-flash-0731` via Zveno (`ZVENOAI_API_KEY`).

## Run

```bash
python main.py --step benchmark-slm
python main.py --step benchmark-deepseek
python main.py --step probe-margin --limit 8
python main.py --step build-groups --limit 4
python main.py --step grpo-train --dry-run
python main.py --step synthesis --n-synth 4
```

Results land under `results/`. Push controlled paths:

```bash
bash scripts/push_results_github.sh "Publish tip-2 benchmark results"
```

## Scale estimate

Detector-tricking used 8891 train abstracts and produced 6396 DPO pairs. With 116 gold tip-2 items only, plan **~2k–4k** synthetic source items for a first GRPO run (\(G=4\)–\(8\)), and **~6k–9k** to approach detector pairwise DPO scale at \(K=2\).

## Docs

- [docs/pipeline.md](docs/pipeline.md) — benchmark, same-SLM loop, GRPO groups, synthetic split
- [docs/unsolvable-guide.md](docs/unsolvable-guide.md) — methods under the print constraint

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

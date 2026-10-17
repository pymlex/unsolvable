# Фреймворк генерации состязательных заданий, устойчивых к решению SLM

A Framework for Generating Adversarial Items Robust to SLM Solvers.

Evaluation target: Russian Unified State Exam (ЕГЭ) Russian language **Type 2** items (word sense in context; multi-select among five sense glosses; answer is concatenated option digits, e.g. `35`).

## Dataset

Public Hugging Face dataset:

- [pymlex/ege-rus-tip2-reshu-2026](https://huggingface.co/datasets/pymlex/ege-rus-tip2-reshu-2026) — 116 items, source `reshu-ege-research`.

Fields: `id`, `task_type`, `instruction`, `passage`, `options` (5), `answer`, `answer_list`, `explanation`, `source`.

## Guide

**[docs/unsolvable-guide.md](docs/unsolvable-guide.md)** — methods under the printed-exam constraint (no human pilot loop; grammatical edits only).

## Zero-edit baseline (temperature 0)

Measures SLM exact-match accuracy on **all** items with **no adversarial edits**. Decoding is greedy (`do_sample=False`). The prompt asks for digits only in the ЕГЭ tip-2 convention. Scores: exact match, Wilson 95% CI, bootstrap 95% CI over items.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=...   # if the hub requires auth for the model
python -m eval.baseline \
  --dataset pymlex/ege-rus-tip2-reshu-2026 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --out results/baseline_qwen25_0.5b.json
```

Optional local JSONL mirror:

```bash
python -m eval.baseline --dataset path/to/train.jsonl --model Qwen/Qwen2.5-0.5B-Instruct
```

### Baseline results

| model | n | accuracy | Wilson 95% CI | bootstrap 95% CI |
|---|---:|---:|---|---|
| *(run `eval.baseline` to fill)* | 116 | — | — | — |

## Rebuild dataset from DOCX

```bash
python scripts/parse_tip2_docx.py --docx tip2.docx --out-dir data/ege-rus-tip2
```

## Licence

GPL-3.0. See [LICENSE](LICENSE).

```bibtex
@misc{unsolvable-2026,
  title={A Framework for Generating Adversarial Items Robust to {SLM} Solvers},
  author={{Фреймворк генерации состязательных заданий, устойчивых к решению SLM}},
  howpublished={https://github.com/pymlex/unsolvable},
  year={2026}
}

@misc{ege-rus-tip2-reshu-2026,
  title={{EGE} Russian Type 2 items from Reshu {EGE} (research scrape)},
  author={{pymlex}},
  howpublished={https://huggingface.co/datasets/pymlex/ege-rus-tip2-reshu-2026},
  year={2026}
}
```

The project is under GPL-3.0 license.

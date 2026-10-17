# Фреймворк генерации состязательных заданий, устойчивых к решению SLM

A Framework for Generating Adversarial Items Robust to SLM Solvers.

Evaluation target: Russian Unified State Exam (ЕГЭ) Russian language **Type 2** items (word sense in context; multi-select among five sense glosses; answer is concatenated option digits, e.g. `35`).

## Dataset

Public Hugging Face dataset:

- [pymlex/reshuege-ru-2026](https://huggingface.co/datasets/pymlex/reshuege-ru-2026) — 116 items. All items are EGE Russian tip 2 from a Решу ЕГЭ 2026 research scrape. Provenance and the fixed stem live on the Hugging Face dataset card, not as row fields.

Row fields: `id`, `passage` (lexical targets as `**word**`), `highlighted_words`, `options` (5, clean), `answer`, `answer_list`, `explanation`.

Eval uses `FIXED_INSTRUCTION` in `eval/prompt.py`:

> В тексте выделено пять слов. Укажите варианты ответов, в которых лексическое значение выделенного слова соответствует его значению в данном тексте. Запишите номера ответов.

## Guide

**[docs/unsolvable-guide.md](docs/unsolvable-guide.md)** — methods under the printed-exam constraint (no human pilot loop; grammatical edits only).

## Zero-edit baseline (temperature 0)

Measures SLM exact-match accuracy on **all** items with **no adversarial edits**. Decoding is greedy (`do_sample=False`). The prompt asks for digits only in the ЕГЭ tip-2 convention. Scores: exact match, Wilson 95% CI, bootstrap 95% CI over items.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=...   # if the hub requires auth for the model
python -m eval.baseline \
  --dataset pymlex/reshuege-ru-2026 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --out results/baseline_qwen25_0.5b.json
```

Optional local JSONL mirror:

```bash
python -m eval.baseline --dataset path/to/train.jsonl --model Qwen/Qwen2.5-0.5B-Instruct
```

### Baseline results

Zero-edit, temperature 0, full set ($N=116$), model `Qwen/Qwen2.5-0.5B-Instruct` on CPU. Artifacts: `results/baseline_qwen25_0.5b.json`.

| model | n | accuracy | Wilson 95% CI | bootstrap 95% CI |
|---|---:|---:|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | 116 | 0.0690 | [0.0354, 0.1302] | [0.0259, 0.1207] |

## Rebuild dataset from DOCX

Bold lexical targets (`w:b`, including hyperlink runs) are preserved as `**word**` in `passage` and listed in `highlighted_words`. Instruction is fixed for every row.

```bash
python scripts/parse_tip2_docx.py --docx tip2.docx --out-dir data/reshuege-ru-2026
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

@misc{reshuege-ru-2026,
  title={reshuege-ru-2026: {EGE} Russian Type 2 items from Reshu {EGE}},
  author={{pymlex}},
  howpublished={https://huggingface.co/datasets/pymlex/reshuege-ru-2026},
  year={2026}
}
```

The project is under GPL-3.0 license.

# Pipeline

Фреймворк генерации состязательных заданий, устойчивых к решению SLM.

## Models

- Local SLM on Clore GPU: `Qwen/Qwen3.5-2B` via `SLM_MODEL_ID`. Post-trained chat model; no separate Instruct id for this size. Same SLM is used for tip-2 solving, paraphrase sampling, \(s_m\) scoring, and GRPO policy training.
- Strong solver and judge: `deepseek/deepseek-v4-flash-0731` at `https://api.zveno.ai/v1` with `ZVENOAI_API_KEY`.

## Benchmark

Temperature 0, numeric answers only, exact match on all tip-2 items from `pymlex/reshuege-ru-2026`. Wilson and bootstrap 95% CI. Artefacts under `results/benchmark/`.

```bash
python main.py --step benchmark-slm
python main.py --step benchmark-deepseek
```

## Same-SLM training loop

1. Sample \(K\) paraphrases of tip-2 **text** with the SLM. Highlights stay as `**...**`. Length change at most \(\pm 20\%\).
2. Zveno judge checks whether the paraphrase broke the task given answer and explanation.
3. Score gold answer under the exam prompt:

$$s_m(y^\star\mid x)=\sum_t \log p_m(y^\star_t\mid x,y^\star_{<t})$$

4. Build GRPO groups with reward \(-s_m\). Optional pairwise DPO export when the score gap exceeds \(\tau\).
5. Train the same SLM with TRL GRPO on Clore (`python main.py --step grpo-train`). Use `--dry-run` to validate wiring without a full GPU train.

## Scale

Detector-tricking reference: 8891 train abstracts yielded 6396 DPO pairs. Here gold tip-2 has only 116 items. First GRPO pass needs about **2000** synthetic source items at group size \(G=4\)–\(8\). Matching detector pairwise DPO scale at \(K=2\) needs about **6k–9k** synthetic source items.

## Synthetic split

`python main.py --step synthesis` few-shots from reshuege gold, samples five lemmas, fetches senses from Russian Wiktionary, and asks DeepSeek to write a tip-2-like item. Local split name: `train-deepseek-v4-flash-0731`. Push with `scripts/push_synthesis_hf.py`.

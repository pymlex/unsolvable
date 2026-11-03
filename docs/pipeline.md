# Pipeline

The active pipeline is LoRA DPO, documented in [dpo.md](dpo.md).

1. Snapshot `pymlex/reshuege-ru-2026` at one immutable revision:
   `train-deepseek-v4-flash/train` (500) and `test-reshuege/test` (116).
2. Generate two independent small-model rewrites of each train passage,
   conditioning on dictionary definitions, the key and `explanation`.
3. Apply local structural checks; no external semantic judge is called.
4. Rank each pair by gold-answer-plus-EOS log probability under a frozen,
   no-thinking solver. Lower likelihood is `chosen`.
5. Train a LoRA paraphrasing adapter with TRL DPO.
6. At step 0, every configured interval and the last step, generate and evaluate
   all 116 real exam items against the same frozen solver. Save per-item outputs,
   exact match, Jaccard, acceptance rate, conditional ASR and score history.

The synthetic source generator and the older GRPO experiments remain available
as individual commands. They are not part of `--step all`.

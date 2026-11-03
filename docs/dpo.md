# DPO paraphrasing of EGE items

This pipeline follows `pymlex/ai-text-detector-tricking`: the base small model
generates two independent rewrites, a frozen scorer ranks them, and TRL trains
the generator with DPO. Here the scorer is a no-thinking exam solver.

## Data

Only `pymlex/reshuege-ru-2026` is used:

- Training sources: config `train-deepseek-v4-flash`, split `train`, 500 items.
- Every evaluation: config `test-reshuege`, split `test`, all 116 real items.

`prepare` resolves the HF revision once and snapshots both configs at that SHA.
The manifest records item counts and content fingerprints. Training checks that
every preference comes from the synthetic snapshot and that no source/test ids
or identical passages overlap. Real test data is retained as published, including
items with missing or repeated highlight markup; no test items are dropped.
The `highlighted_words` metadata can contain dictionary forms while passages
contain inflected forms. Rewrite checks use the actual `**spans**` in the passage.
Subsequent prepare calls reuse that snapshot. Publishing preferences to another
config in the same HF repo does not invalidate it. Use `--refresh-data` only
when intentionally replacing the source snapshot.

## Generate and rank pairs

For each of the 500 training items:

1. Give the small model the original passage, dictionary definitions from the
   options, the fixed key, and `explanation`. Ask it to change the sentences
   carefully, preserve word meanings and return **only the complete passage**.
   Inflection is allowed when grammar requires it; target-word synonyms are not.
2. Call generation twice independently at temperature 0.8, with thinking off.
3. Record structural diagnostics for changed highlight lemmas, extra commentary,
   invisible characters and word-count changes above 20%. These flags do not
   discard training rows. Actually truncated samples are regenerated.
4. Compute the log probability of the gold digit answer plus EOS under the
   frozen solver's exam prompt. The solver never receives the explanation or key.
5. Choose the lowest-score rewrite as `chosen`, the highest as `rejected`.
   If their difference is <= `DPO_MIN_PAIR_GAP` (default 0.5 nats), generate two
   more independent rewrites for that source and extend its candidate pool.
   Repeat without a round limit until every source has a difference strictly
   greater than 0.5. Sources already above the threshold are not regenerated.
   All attempts, including duplicates and ties, remain in the raw cache.

If a natural completion exhausts VRAM, refinement preserves the existing pools
and retries with a fresh seed. It does not impose an output-token cap. A sample
that reaches the model's native context boundary is archived in
`failed_candidates` and regenerated rather than used as an incomplete DPO target.

Local morphology (`pymorphy3`) permits grammatical inflections such as
«фразу → фразой», while rejecting replacement by a different lexeme. This is
a structural check, not a semantic judge. The scorer computes log-softmax in
float32 even when the model uses bf16, to avoid rounding away small score gaps.
The 0.5 threshold is an experiment requirement, not a calibrated accuracy gain.
This difference measures gold-answer log probability, not solver accuracy.
`PAIRWISE_GAP_TAU` remains a legacy diagnostic annotation only.

There are no DeepSeek API calls or semantic rechecks in this DPO pipeline.
Preservation of meanings is instructed through the definitions and explanation;
the diagnostic checks are structural. Reports label this `structural_only`.
The original key and options never become trainable completions.
All candidates are scored, regardless of structural diagnostics. Ties use stable
generation order in the raw pool; final selected pairs must pass the strict margin.

Candidate records are appended after each item. Rerunning `preferences` reuses
matching candidates and scores; changes to source, prompt or generation settings
invalidate their cache keys. The resulting full dataset always has exactly 500 pairs, one per train item.
Items are processed in batches (`GENERATION_BATCH_SIZE`), with two
separate generation passes per batch. Refinement adds two samples per pending item.
Batch size can be tuned for GPU throughput without invalidating completed
candidate records; the sampling-policy and prompt settings must still match.

## Ubuntu / Clore

Python 3.12, one GPU. Default is `Qwen/Qwen3.5-2B`. The installer uses a CUDA 12.8
PyTorch wheel, suitable for the RTX 5090 with a recent driver. Override
`TORCH_CUDA_INDEX` for other machines. The host's driver must support that runtime;
a separate system CUDA toolkit is not required.

```bash
bash scripts/install_ubuntu_jupyter.sh
# Edit .env if changing model or hyperparameters. No API key is required for DPO.
source .venv/bin/activate
python main.py --step prepare
python main.py --step preferences
python main.py --step train --dry-run
bash scripts/run_dpo.sh
```

Or run preparation, pair construction and training together:

```bash
bash scripts/run_all.sh
```

Run in a persistent shell such as tmux, or detach explicitly:

```bash
nohup bash scripts/run_all.sh > results-launch.log 2>&1 < /dev/null &
```

The older `benchmark-*`, `synthesis`, `probe-margin`, `build-groups` and
`grpo-train` commands remain explicit legacy utilities. `all` runs the DPO
pipeline; it does not rerun the historical zero-edit benchmarks or synthesis.

## Training and evaluation

Default LoRA: rank 16, alpha 32, all linear layers; four epochs, LR 1e-5,
beta 0.1, microbatch 1, accumulation 8. Dependencies are pinned in requirements.
Before training, token lengths are checked; oversized pairs fail explicitly
against the native model context window instead of silently truncating the
prompt, explanation or rewritten text. DPO uses `max_length=None`. Generation
has no application-level token cap: it stops at EOS within the native context
window. Completed cached samples remain reusable when removing an old cap;
any actually truncated samples are regenerated.

The callback evaluates at step 0, every 20 optimizer steps, and the final step.
Each evaluation greedily generates one passage for **each of the 116 real items**.
Its prompt has the same definitions/explanation structure as during pairing.
The immutable base solver then answers the accepted rewritten tasks with
thinking off. With the default same generator/solver model, disabling the LoRA
adapter reuses the frozen base on the GPU. The solver cannot improve or degrade
as DPO updates the adapter. A distinct `SOLVER_MODEL_ID` loads a separate solver.

Metrics: exact match, Jaccard, original-task accuracy, accuracy drop, mean gold
log probability on accepted rewrites, structural acceptance rate, and attack
success conditional on an originally correct answer. Structurally rejected
rewrites fall back to the original task in full-set accuracy and cannot count
as successful attacks. Per-item files keep the rejected output and reason.
All 116 items are monitoring data, so repeated checkpoint monitoring is not an
independent final generalization estimate.

Artifacts:

- `results/data/{train,test}.jsonl`, `manifest.json`: pinned snapshots.
- `results/preferences/candidates.jsonl`: resumable raw candidates and scores.
- `results/preferences/dpo_preferences.{jsonl,csv}`, `dpo_hf_dataset/`.
- `results/preferences/dpo_manifest.json`, `build_stats.json`.
- `results/preferences/refinement_progress.json`, `refinement_best_pairs.jsonl`:
  current readiness and best pairs while refinement is running.
- `results/checkpoints/dpo/checkpoint-*`: LoRA, optimizer, RNG and trainer state.
- `results/checkpoints/dpo/final`: final LoRA adapter and tokenizer.
- `results/monitoring/dpo/monitor_step_*.json`: all 116 item outputs per eval.
- `results/monitoring/dpo/validation_history.{json,csv}`.
- `results/monitoring/dpo/train_history.{jsonl,json,csv}`: loss, learning rate,
  DPO rewards/margins and other Trainer metrics persisted after every optimizer
  step (`DPO_LOGGING_STEPS=1`), before training finishes. JSONL is flushed and
  fsynced per log; JSON/CSV snapshots are replaced atomically. Resume retains history.
- `results/monitoring/dpo/training_summary.png`.

```bash
# Resume optimizer/RNG state from the latest saved checkpoint:
bash scripts/run_dpo.sh --resume
# Or select a checkpoint explicitly:
bash scripts/run_dpo.sh --resume results/checkpoints/dpo/checkpoint-20
# Re-evaluate a saved adapter on all 116:
python main.py --step evaluate --checkpoint results/checkpoints/dpo/final
```

Resume checks the original data/pair fingerprints and training configuration.
Use a different `RESULTS_DIR` for a new experiment. `--limit` is accepted only
for preference smoke runs; a partial preference build cannot start a full run.
For quick execution checks, `--max-steps 1` still evaluates all 116 real items.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

Tests include actual two-step TRL/LoRA training of a tiny randomly initialized
CPU model, unchanged frozen-solver scores, adapter updates/checkpoint saving,
two independent generation calls, pair direction, source isolation and callback
scheduling. They require no API key or model downloads. GPU/model-specific
compatibility is verified separately on the target server.

## HF publication

A full `preferences` run automatically publishes all 500 pairs to
`pymlex/reshuege-ru-2026`, config `dpo-qwen3.5-2b`, split `train`.
The existing `train-deepseek-v4-flash/train` and `test-reshuege/test` configs are
preserved. Raw candidate attempts (including earlier attempts), the pair manifest,
CSV and build diagnostics are uploaded under `artifacts/dpo-qwen3.5-2b/`.
`HF_TOKEN` must have write access to that dataset. Retry publication independently
with `python main.py --step publish-preferences`; regeneration is not required.

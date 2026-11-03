"""Data isolation, preference direction and honest attack evaluation regressions."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from eval.ege_data import validate_sources
from eval.evaluate_paraphraser import evaluate_items
from generation.prompts import build_item_paraphrase_prompt
from judge.ege_quality import QualityGate, structural_error
from preferences.build_dpo_dataset import select_pair
from training.callbacks import EgeMonitorCallback
from training.dpo_train import validate_pairs
from utils.artifacts import fingerprint


@pytest.fixture
def row():
    return {"id": "a", "passage": "Вчера **один** **два** **три** **четыре** **пять** стояли рядом.",
            "options": ["А", "Б", "В", "Г", "Д"], "answer": "25", "explanation": "Смысл нужно сохранить."}


def candidate(text, score):
    return {"text": text, "score": score, "quality": {"valid": True, "judge_checked": False}}


def test_prompt_supplies_explanation_and_requests_passage_only(row):
    prompt = build_item_paraphrase_prompt(row)
    assert row["explanation"] in prompt
    assert "ТОЛЬКО полный перефразированный текст" in prompt
    assert "Ключ (не выводить): 25" in prompt


def test_lower_gold_likelihood_is_chosen(row):
    a, b = candidate(row["passage"], -1), candidate(row["passage"].replace("Вчера", "Сегодня"), -5)
    pair = select_pair(row, [a, b], tau=0.25)
    assert pair["chosen"] == b["text"]
    assert pair["gap"] == 4
    duplicate = select_pair(row, [b, b], tau=0)
    assert duplicate["is_duplicate"] and duplicate["is_tie"]
    a["quality"]["valid"] = False
    assert select_pair(row, [a, b])["rejected_quality"]["valid"] is False


def test_small_gaps_are_retained_and_nonfinite_scores_fail(row):
    a = candidate(row["passage"], -1)
    b = candidate(row["passage"].replace("Вчера", "Сегодня"), -1.01)
    assert select_pair(row, [a, b], tau=.25)["below_probe_margin"] is True
    b["score"] = float("nan")
    with pytest.raises(ValueError, match="invalid score"):
        select_pair(row, [a, b])


@pytest.mark.parametrize("change,expected", [
    (lambda text: text.replace("**два**", "**другой**"), "highlight_changed"),
    (lambda text: "Ответ: 25\n" + text, "commentary_or_reasoning"),
    (lambda text: text + "\u200b", "invisible_character"),
])
def test_quality_rejects_broken_items(row, change, expected):
    assert structural_error(row, {"text": change(row["passage"])}) == expected
    assert structural_error(row, {"text": row["passage"], "truncated": True}) == "truncated"


def test_source_overlap_is_fatal(row):
    with pytest.raises(ValueError, match="id overlap"):
        validate_sources([row], [row], expected_train=1, expected_test=1)
    other = {**row, "id": "b"}
    with pytest.raises(ValueError, match="passage overlap"):
        validate_sources([row], [other], expected_train=1, expected_test=1)


def test_invalid_rewrite_cannot_inflate_asr(row, tmp_path):
    other = {**row, "id": "b"}
    calls = []
    def solve(items):
        calls.append(items)
        return ["1"] * len(items)
    gate = QualityGate()
    def generate(items):
        return [[{"text": row["passage"].replace("Вчера", "Сегодня")}], [{"text": "garbage"}]]
    result = evaluate_items([row, other], generate, solve, gate, baseline_raw=["25", "25"])
    assert len(calls[0]) == 1
    assert result["summary"]["accuracy"] == .5
    assert result["summary"]["conditional_asr"] == .5
    assert result["summary"]["n_valid"] == 1
    assert result["items"][1]["raw"] == "25"
    assert result["items"][1]["attack_success"] is False


def test_pair_provenance_prevents_test_or_partial_training(row):
    a = candidate(row["passage"], -1)
    b = candidate(row["passage"].replace("Вчера", "Сегодня"), -5)
    pairs = [select_pair(row, [a, b])]
    from constants import SLM_MODEL_ID, SOLVER_MODEL_ID
    manifest = {"test": "fixed-116"}
    meta = {"source": manifest, "pairs_fingerprint": fingerprint(pairs), "partial": False,
            "n_source": 1, "n_pairs": 1, "tau": .25, "minimum_gap": .5,
            "settings": {"generator": SLM_MODEL_ID, "solver": SOLVER_MODEL_ID}}
    validate_pairs(pairs, meta, [row], manifest)
    with pytest.raises(ValueError, match="Partial"):
        validate_pairs(pairs, {**meta, "partial": True}, [row], manifest)
    changed = copy.deepcopy(pairs)
    changed[0]["item_id"] = "test-id"
    with pytest.raises(ValueError, match="non-train"):
        validate_pairs(changed, {**meta, "pairs_fingerprint": fingerprint(changed)}, [row], manifest)


def test_monitor_runs_start_interval_and_final_without_duplicate():
    steps = []
    evaluator = SimpleNamespace(run=lambda model, step, epoch: steps.append(step))
    callback = EgeMonitorCallback(evaluator, every_steps=2)
    state = SimpleNamespace(global_step=0, epoch=0)
    callback.on_train_begin(None, state, None, model=object())
    for step in range(1, 6):
        state.global_step = step
        callback.on_step_end(None, state, None, model=object())
    callback.on_train_end(None, state, None, model=object())
    callback.on_train_end(None, state, None, model=object())
    assert steps == [0, 2, 4, 5]


def test_epoch_monitor_runs_at_zero_and_each_point_two_after_loss_is_logged():
    steps = []
    evaluator = SimpleNamespace(run=lambda model, step, epoch: steps.append(step))
    callback = EgeMonitorCallback(evaluator, every_steps=20, every_epochs=.2)
    state = SimpleNamespace(global_step=0, epoch=0)
    callback.on_train_begin(None, state, None, model=object())
    for step, epoch in [(1,.016),(12,.192),(13,.208),(25,.4),(38,.608),(50,.8),(63,1.0)]:
        state.global_step, state.epoch = step, epoch
        before = len(steps)
        callback.on_step_end(None, state, None, model=object())
        assert len(steps) == before
        callback.on_log(None, state, None, model=object(), logs={"loss": .69})
    callback.on_train_end(None, state, None, model=object())
    assert steps == [0,13,25,38,50,63]


def test_generation_and_scoring_disable_thinking_through_kwargs():
    from eval.model_io import apply_chat_template_no_think
    from scoring.answer_logprob import render_exam_prompt
    class Tokenizer:
        chat_template = "template"
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "no-thinking"
    assert apply_chat_template_no_think(Tokenizer(), [{"role": "user", "content": "x"}]) == "no-thinking"


def test_quality_gate_never_calls_external_api(row, monkeypatch):
    import eval.zveno_client as zveno
    def forbidden(*args, **kwargs):
        raise AssertionError("DPO quality checks must not call DeepSeek")
    monkeypatch.setattr(zveno, "chat_completion", forbidden)
    result = QualityGate().check(row, {"text": row["passage"]})
    assert result["valid"] is True and result["judge_checked"] is False


def test_morphology_allows_inflection_but_not_synonyms():
    from judge.ege_quality import same_lexeme
    assert same_lexeme("фразу", "фразой")
    assert same_lexeme("правом", "правый")
    assert not same_lexeme("фразу", "фрагмент")
    assert not same_lexeme("волна", "эмоции")


def test_pair_builder_batches_and_resumes_without_regeneration(row, tmp_path, monkeypatch):
    import preferences.build_dpo_dataset as build
    from utils.artifacts import read_jsonl
    rows = [{**row, "id": str(i), "passage": row["passage"] + f" Число {i}."} for i in range(5)]
    calls = []
    def generate(model, tokenizer, items, *, num_samples):
        assert num_samples == 2
        calls.append(len(items))
        return [[{"text": r["passage"].replace("Вчера", word), "truncated": False}
                 for word in ("Сегодня", "Недавно")] for r in items]
    class Solver:
        def __init__(self, *args, **kwargs):
            pass
        def score(self, source, text):
            return -5.0 if "Недавно" in text else -1.0
    monkeypatch.setattr(build, "GENERATION_BATCH_SIZE", 4)
    monkeypatch.setattr(build, "load_prepared", lambda: (rows, [], {"revision": "test-sha"}))
    monkeypatch.setattr(build, "load_slm", lambda *a: (object(), object(), "cpu"))
    monkeypatch.setattr(build, "FrozenSolver", Solver)
    monkeypatch.setattr(build, "generate_item_paraphrases", generate)
    monkeypatch.setattr(build, "ensure_result_dirs", lambda: None)
    monkeypatch.setattr(build, "PREFERENCES_DIR", tmp_path)
    monkeypatch.setattr(build, "PAIRS_PATH", tmp_path / "pairs.jsonl")
    monkeypatch.setattr(build, "PAIRS_META_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(build, "CACHE_PATH", tmp_path / "cache.jsonl")
    build.build_preference_dataset()
    assert calls == [4, 1]
    assert len(read_jsonl(build.PAIRS_PATH)) == 5
    import json
    (tmp_path / "generation_samples.jsonl").write_text(json.dumps({
        "run_id": "interrupted-batch", "item_id": "0", "sample_pass": 1,
        "text": rows[0]["passage"], "truncated": False,
    }) + "\n", encoding="utf-8")
    calls.clear()
    monkeypatch.setattr(build, "GENERATION_BATCH_SIZE", 8)
    build.build_preference_dataset()
    assert calls == []
    assert len(read_jsonl(build.CACHE_PATH)) == 6
    assert read_jsonl(build.PAIRS_PATH)[0]["candidate_count"] == 3
    build.build_preference_dataset()
    assert len(read_jsonl(build.CACHE_PATH)) == 6  # Journal replay is idempotent.


def test_legacy_candidates_survive_batch_size_tuning(row):
    from preferences.build_dpo_dataset import cached_candidate
    settings = {"version": 4, "seed": 42, "generation_batch_size": 16}
    old_key = fingerprint({"settings": settings, "prompt": build_item_paraphrase_prompt(row)})
    cache = {old_key: {"key": old_key, "item_id": row["id"], "candidates": [{"text": "cached", "truncated": False}]}}
    key, record = cached_candidate(cache, {**settings, "generation_batch_size": 64}, row)
    assert record["candidates"][0]["text"] == "cached"
    assert record["generation_batch_size"] == 16


def test_generation_uses_eos_and_native_context_instead_of_manual_cap():
    from utils.model_context import generation_bounds
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32768))
    tokenizer = SimpleNamespace(model_max_length=10**30)
    assert generation_bounds(model, tokenizer, 5000) == {"max_length": 32768, "max_new_tokens": None}
    with pytest.raises(ValueError, match="Complete prompt"):
        generation_bounds(model, tokenizer, 40000)


def test_complete_capped_candidates_can_be_reused_without_cap(row):
    from preferences.build_dpo_dataset import cached_candidate
    settings = {"version": 4, "max_new_tokens": 1024, "generation_batch_size": 16}
    old_key = fingerprint({"settings": settings, "prompt": build_item_paraphrase_prompt(row)})
    cache = {old_key: {"key": old_key, "candidates": [{"text": "complete", "truncated": False}]}}
    key, record = cached_candidate(cache, {**settings, "max_new_tokens": None, "generation_batch_size": 64}, row)
    assert record["candidates"][0]["text"] == "complete"
    cache[old_key]["candidates"][0]["truncated"] = True
    cache.pop(key, None)
    assert cached_candidate(cache, {**settings, "max_new_tokens": None}, row)[1] is None


def test_qwen_keeps_both_document_and_chat_eos_after_trainer_alignment():
    from utils.model_context import generation_bounds
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=32768),
                            generation_config=SimpleNamespace(eos_token_id=248046))
    tokenizer = SimpleNamespace(model_max_length=32768, eos_token_id=248046,
                                all_special_tokens=["<|im_end|>", "<|endoftext|>"],
                                convert_tokens_to_ids=lambda token: {"<|im_end|>": 248046, "<|endoftext|>": 248044}[token])
    assert generation_bounds(model, tokenizer, 100)["eos_token_id"] == [248044, 248046]


def test_refinement_is_strict_and_reuses_existing_extremes(row, tmp_path, monkeypatch):
    import preferences.build_dpo_dataset as build
    monkeypatch.setattr(build, "PREFERENCES_DIR", tmp_path)
    other = {**row, "id": "b"}
    pools = {
        "a": {"item_id": "a", "candidates": [candidate("old-low", -1), candidate("old-high", -.5)]},
        "b": {"item_id": "b", "candidates": [candidate("ready-low", -3), candidate("ready-high", -1)]},
    }
    calls, persisted = [], []
    def generate(rows):
        calls.append([r["id"] for r in rows])
        return [[{"text": "new-low", "truncated": False}, {"text": "new-high", "truncated": False}]]
    gate = SimpleNamespace(check=lambda *args: {"valid": True, "judge_checked": False})
    result = build.refine_until_margin([row, other], pools, generate=generate,
               score=lambda r, t: -2 if t == "new-low" else -.4, gate=gate,
               persist=lambda record: persisted.append(record["item_id"]), minimum_gap=.5)
    assert calls == [["a"]]  # Exactly .5 must be refined; ready item is untouched.
    assert persisted == ["a"]
    assert len(pools["a"]["candidates"]) == 4
    assert all(p["gap"] > .5 for p in result)


def test_training_history_is_saved_on_each_log_before_train_end(tmp_path):
    import json
    from training.history import TrainingHistoryCallback
    callback = TrainingHistoryCallback(tmp_path)
    state = SimpleNamespace(global_step=1, epoch=.1)
    callback.on_log(None, state, None, logs={"loss": .7, "learning_rate": 1e-5})
    assert json.loads((tmp_path / "train_history.json").read_text())[0]["loss"] == .7
    state.global_step = 2
    callback.on_log(None, state, None, logs={"loss": .6, "rewards/margins": .2})
    assert len((tmp_path / "train_history.jsonl").read_text().splitlines()) == 2
    assert "rewards/margins" in (tmp_path / "train_history.csv").read_text()
    resumed = TrainingHistoryCallback(tmp_path, resume=True)
    assert len(resumed.records) == 2


def test_refinement_retries_oom_without_losing_pools(row, tmp_path, monkeypatch):
    import torch
    import preferences.build_dpo_dataset as build
    monkeypatch.setattr(build, "PREFERENCES_DIR", tmp_path)
    pools = {"a": {"item_id": "a", "candidates": [candidate("old-low", -1), candidate("old-high", -.5)]}}
    calls = []
    def generate(rows):
        calls.append(torch.initial_seed())
        if len(calls) == 1:
            raise torch.OutOfMemoryError("simulated long completion")
        return [[{"text": "new-low", "truncated": False}, {"text": "new-high", "truncated": False}]]
    gate = SimpleNamespace(check=lambda *args: {"valid": True, "judge_checked": False})
    result = build.refine_until_margin([row], pools, generate=generate,
               score=lambda r, t: -2 if t == "new-low" else -.4, gate=gate,
               persist=lambda record: None, minimum_gap=.5)
    assert len(calls) == 2 and calls[0] != calls[1]
    assert len(pools["a"]["candidates"]) == 4
    assert pools["a"]["failed_generation_attempts"] == [{"round": 1, "reason": "cuda_oom"}]
    assert result[0]["gap"] > .5


def test_native_context_exhaustion_is_archived_and_regenerated(row, tmp_path, monkeypatch):
    import preferences.build_dpo_dataset as build
    monkeypatch.setattr(build, "PREFERENCES_DIR", tmp_path)
    pools = {"a": {"item_id": "a", "candidates": [candidate("old-low", -1), candidate("old-high", -.5)]}}
    batches = iter([
        [[{"text": "unfinished passage", "truncated": True}, {"text": "unchanged", "truncated": False}]],
        [[{"text": "new-low", "truncated": False}, {"text": "new-high", "truncated": False}]],
    ])
    gate = SimpleNamespace(check=lambda *args: {"valid": True, "judge_checked": False})
    scored = []
    def score(r, text):
        scored.append(text)
        return -2 if text == "new-low" else -.5
    result = build.refine_until_margin([row], pools, generate=lambda rows: next(batches),
               score=score, gate=gate, persist=lambda record: None, minimum_gap=.5)
    assert "unfinished passage" not in scored
    assert pools["a"]["failed_candidates"][0]["text"] == "unfinished passage"
    assert result[0]["gap"] > .5


def test_generation_progress_saves_eos_before_other_items_finish(tmp_path):
    import json
    import torch
    from generation.progress import GenerationProgress
    tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: str(tokens))
    progress = GenerationProgress(tokenizer, [{"id": "a"}, {"id": "b"}], [9], tmp_path,
                                  sample_pass=1, sampling={"temperature": 1.1})
    progress.put(torch.tensor([[1, 2], [1, 2]]))
    progress.put(torch.tensor([3, 4]))
    progress.put(torch.tensor([9, 5]))
    rows = [json.loads(line) for line in (tmp_path / "generation_samples.jsonl").read_text().splitlines()]
    assert [r["item_id"] for r in rows] == ["a"]
    progress.report()
    state = json.loads((tmp_path / "generation_progress.json").read_text())
    assert state["n_finished"] == 1 and state["active_item_ids"] == ["b"]
    progress.put(torch.tensor([9, 9]))
    progress.end()
    assert len((tmp_path / "generation_samples.jsonl").read_text().splitlines()) == 2


def test_generation_abandons_repeated_paragraph_loop_without_token_cap(tmp_path):
    import torch
    from generation.progress import GenerationProgress, repetition_stopping_criteria
    tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: str(tokens))
    progress = GenerationProgress(tokenizer, [{"id": "a"}], [999], tmp_path,
                                  sample_pass=1, sampling={})
    progress.put(torch.tensor([[1, 2]]))
    stop = repetition_stopping_criteria(progress)
    for token in list(range(32)) * 2:
        progress.put(torch.tensor([token]))
    assert not stop(torch.tensor([[1]]), None).any()
    for token in range(32):
        progress.put(torch.tensor([token]))
    assert stop(torch.tensor([[1]]), None).all()
    assert progress.loop_ids == {"a"} and not progress.finished
    short = GenerationProgress(tokenizer, [{"id": "b"}], [999], tmp_path / "short",
                               sample_pass=1, sampling={})
    short.put(torch.tensor([[1, 2]]))
    for token in list(range(5)) * 20:
        short.put(torch.tensor([token]))
    assert short.loop_ids == {"b"}


def test_loss_convergence_stops_only_after_observed_plateau(tmp_path, monkeypatch):
    from training.convergence import LossConvergenceCallback
    monkeypatch.setattr("training.convergence.write_json", lambda *args: None)
    callback = LossConvergenceCallback(tmp_path)
    control = SimpleNamespace(should_training_stop=False)
    for step in range(1, 81):
        callback.on_log(None, SimpleNamespace(global_step=step), control, logs={"loss": .4})
        assert control.should_training_stop == (step == 80)
    assert callback.converged and len(callback.windows) == 4
    decreasing = LossConvergenceCallback(tmp_path / "decreasing")
    control.should_training_stop = False
    for step in range(1, 161):
        decreasing.on_log(None, SimpleNamespace(global_step=step), control, logs={"loss": 1-step*.002})
    assert not decreasing.converged and not control.should_training_stop
    warmup = LossConvergenceCallback(tmp_path / "warmup")
    control.should_training_stop = False
    args = SimpleNamespace(get_warmup_steps=lambda max_steps: 100)
    for step in range(1, 161):
        warmup.on_log(args, SimpleNamespace(global_step=step, max_steps=200), control, logs={"loss": .4})
        assert control.should_training_stop == (step == 160)


def test_loop_stop_is_per_item_and_padding_cannot_become_valid_eos(tmp_path):
    import torch
    from generation.progress import GenerationProgress, repetition_stopping_criteria
    tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: str(tokens))
    progress = GenerationProgress(tokenizer, [{"id": "a"}, {"id": "b"}], [999], tmp_path,
                                  sample_pass=1, sampling={})
    progress.put(torch.tensor([[1, 2], [1, 2]]))
    progress.loop_ids.add("a")
    assert repetition_stopping_criteria(progress)(torch.tensor([[1], [1]]), None).tolist() == [True, False]
    progress.put(torch.tensor([999, 999]))
    assert progress.finished == {1}
    assert '"item_id": "a"' not in (tmp_path / "generation_samples.jsonl").read_text()


def test_prompt_metadata_leak_is_abandoned_without_a_length_limit(tmp_path):
    import torch
    from generation.progress import GenerationProgress
    tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs:
                                "Ключ (не выводить): 134" if tokens[0] == 2 else "Полный перефраз.")
    progress = GenerationProgress(tokenizer, [{"id": "a"}, {"id": "b"}], [9], tmp_path,
                                  sample_pass=1, sampling={})
    progress.put(torch.tensor([[1], [1]]))
    progress.put(torch.tensor([2, 1]))
    progress.put(torch.tensor([9, 9]))
    assert progress.invalid_sections == {"a": "prompt_metadata_in_output"}
    assert progress.finished == {1}
    assert '"item_id": "a"' not in (tmp_path / "generation_samples.jsonl").read_text()

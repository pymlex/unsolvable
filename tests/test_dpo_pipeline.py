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
    assert select_pair(row, [b, b], tau=0) is None
    a["quality"]["valid"] = False
    assert select_pair(row, [a, b]) is None


def test_ties_nonfinite_and_small_gaps_are_skipped(row):
    a = candidate(row["passage"], -1)
    b = candidate(row["passage"].replace("Вчера", "Сегодня"), -1.01)
    assert select_pair(row, [a, b], tau=.25) is None
    b["score"] = float("nan")
    assert select_pair(row, [a, b]) is None


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
            "n_source": 1, "n_pairs": 1, "tau": .25,
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
    calls.clear()
    monkeypatch.setattr(build, "GENERATION_BATCH_SIZE", 8)
    build.build_preference_dataset()
    assert calls == []
    assert len(read_jsonl(build.CACHE_PATH)) == 5


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

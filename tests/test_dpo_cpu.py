"""Actual TRL + PEFT training, no external model downloads or API requests."""

import torch


def test_real_dpo_step_preserves_frozen_solver_and_saves_adapter(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from peft import PeftModel, LoraConfig, get_peft_model
    from training.dpo_train import make_trainer
    from training.callbacks import EgeMonitorCallback
    from scoring.frozen_solver import FrozenSolver
    from generation.ege_paraphrase import generate_item_paraphrases

    torch.set_num_threads(2)
    torch.manual_seed(7)
    raw = Tokenizer(models.WordLevel(unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    raw.train_from_iterator(["rewrite original good bad words <user> <assistant>"],
                            trainers.WordLevelTrainer(special_tokens=["<unk>", "<pad>", "<eos>"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    tokenizer.chat_template = "{% for m in messages %}{{ '<' + m['role'] + '> ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}"
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_positions=1024, n_embd=16,
                                      n_layer=1, n_head=2, eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id))
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
                                            target_modules="all-linear", task_type="CAUSAL_LM"))
    pairs = [{"item_id": "1", "prompt": "rewrite original", "chosen": "good words", "rejected": "bad words"}] * 2
    trainer = make_trainer(model, tokenizer, pairs, output_dir=tmp_path / "run", max_steps=2,
                           batch_size=1, accumulation=1, max_length=None, monitor_every=1)
    assert trainer.args.warmup_steps == 10
    assert trainer.args.learning_rate == 1e-5
    solver = FrozenSolver(trainer.model, tokenizer, model_id="tiny", disable_adapter=True)
    row = {"id": "1", "passage": "original", "options": ["good"] * 5, "answer": "25", "explanation": ""}
    frozen_before = solver.score(row, "original")
    base_before = {name: p.detach().clone() for name, p in trainer.model.named_parameters() if not p.requires_grad}
    adapter_before = {name: p.detach().clone() for name, p in trainer.model.named_parameters() if p.requires_grad}
    seen = []
    class Evaluator:
        def run(self, policy, *, step, epoch):
            assert abs(solver.score(row, "original") - frozen_before) < 1e-6
            seen.append(step)
    evaluation_losses = []
    def evaluate_loss():
        value = trainer.evaluate(eval_dataset=trainer.train_dataset, metric_key_prefix="train_probe")["train_probe_loss"]
        evaluation_losses.append(value)
        return value
    trainer.add_callback(EgeMonitorCallback(Evaluator(), every_steps=1, loss_evaluator=evaluate_loss))
    trainer.train()
    assert seen == [0, 1, 2]
    assert trainer.state.global_step == 2
    assert len(evaluation_losses) == 3 and all(torch.isfinite(torch.tensor(v)) for v in evaluation_losses)
    assert all(torch.equal(p, base_before[n]) for n, p in trainer.model.named_parameters() if n in base_before)
    assert any(not torch.equal(p, adapter_before[n]) for n, p in trainer.model.named_parameters() if n in adapter_before)
    trainer.save_model(str(tmp_path / "final"))
    assert (tmp_path / "final" / "adapter_config.json").exists()
    assert abs(solver.score(row, "original") - frozen_before) < 1e-6
    assert trainer.model.training

    # A warm start retains trained policy weights AND the original frozen DPO
    # reference, instead of silently making the trained policy its own reference.
    from training.dpo_train import restore_dpo_reference
    warm = PeftModel.from_pretrained(GPT2LMHeadModel(model.config), str(tmp_path / "final"), is_trainable=True)
    warm_trainer = make_trainer(warm, tokenizer, pairs, output_dir=tmp_path / "warm", max_steps=2,
                                batch_size=1, accumulation=1)
    restore_dpo_reference(warm_trainer.model, tmp_path / "final")
    expected = {n: p.detach() for n, p in trainer.model.named_parameters() if ".ref." in n}
    assert expected
    assert all(torch.equal(p, expected[n]) for n, p in warm_trainer.model.named_parameters() if n in expected)

    # The same model wrapper must make two real generate calls for a pair.
    calls = []
    original_generate = trainer.model.generate
    def recorded_generate(*args, **kwargs):
        calls.append(kwargs["do_sample"])
        return original_generate(*args, **kwargs)
    trainer.model.generate = recorded_generate
    row["passage"] = "**one** **two** **three** **four** **five**"
    result = generate_item_paraphrases(trainer.model, tokenizer, [row], num_samples=2, max_new_tokens=2)
    assert len(result[0]) == 2 and calls == [True, True]
    assert trainer.model.training

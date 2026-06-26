"""Unit tests of trainer paths that need no GPU (fake generators/forwards)."""

import contextlib
import json
import torch

from matrix import trainer as trainer_mod
from matrix.axis import AxisConfig
from matrix.losses import soft_rkl
from matrix.trainer import MatrixTrainer, TrainConfig


class FakeModel:
    def eval(self):
        pass

    def train(self):
        pass


class HarvestTokenizer:
    def decode(self, y, skip_special_tokens=True):
        # even pool positions yield valid number lists, odd ones prose
        return "825, 568, 932" if int(y[0]) % 2 == 0 else "some prose about numbers"


def _make_harvest_trainer(tmp_path, slug, n_pool=20, target=5):
    raw = tmp_path / "raw_dataset.jsonl"
    with open(raw, "w") as f:
        for i in range(n_pool):
            f.write(json.dumps({"prompt": f"prompt-{i}", "completion": "x"}) + "\n")
    t = object.__new__(MatrixTrainer)
    t.cfg = TrainConfig(batch_size=4, gen_batch_size=4, max_new_tokens=8, target_preference="owl",
                        max_dataset_size=target, raw_dataset_path=str(raw))
    t.axis = AxisConfig.from_slug(slug)
    t.out_dir = tmp_path
    t.device = "cpu"
    t.model = FakeModel()
    t.tokenizer = HarvestTokenizer()
    t._frozen_rollouts = {}
    t.questions = [f"q{i}" for i in range(10)]  # dataset rows: target = min(5, 10) = 5
    t.completions = ["x"] * 10
    return t


def test_sample_filtered_harvest(tmp_path):
    t = _make_harvest_trainer(tmp_path, "on-soft-fkl-fixed")
    seen_greedy = []

    def fake_generate(model, tokenizer, prompts, max_new_tokens, greedy=False):
        seen_greedy.append(greedy)
        return [p[0] for p in prompts]

    def fake_render(tokenizer, q, system, cue, device):
        return torch.tensor([[int(q.split("-")[1])]])

    original_gen, original_render = trainer_mod.batch_generate, trainer_mod.render_prompt_ids
    trainer_mod.batch_generate, trainer_mod.render_prompt_ids = fake_generate, fake_render
    try:
        t._harvest_filtered()
    finally:
        trainer_mod.batch_generate, trainer_mod.render_prompt_ids = original_gen, original_render

    assert all(g is False for g in seen_greedy)        # sample decode
    assert t.questions == [f"prompt-{i}" for i in (0, 2, 4, 6, 8)]
    stats = json.loads((tmp_path / "pregen_stats.json").read_text())
    assert stats["kept"] == 5 and stats["target"] == 5


def test_greedy_filtered_harvest(tmp_path):
    t = _make_harvest_trainer(tmp_path, "on-hard-fkl-greedy-fixed")

    def fake_generate(model, tokenizer, prompts, max_new_tokens, greedy=False):
        assert greedy
        return [p[0] for p in prompts]

    def fake_render(tokenizer, q, system, cue, device):
        return torch.tensor([[int(q.split("-")[1])]])

    original_gen, original_render = trainer_mod.batch_generate, trainer_mod.render_prompt_ids
    trainer_mod.batch_generate, trainer_mod.render_prompt_ids = fake_generate, fake_render
    try:
        t._harvest_filtered()
    finally:
        trainer_mod.batch_generate, trainer_mod.render_prompt_ids = original_gen, original_render

    assert len(t.questions) == 5
    assert set(t._frozen_rollouts) == set(range(5))


class AdapterModel:
    @contextlib.contextmanager
    def disable_adapter(self):
        yield

    def eval(self):
        pass

    def train(self):
        pass


def test_multi_teacher_loss_is_mean_of_per_teacher_kls(tmp_path):
    torch.manual_seed(4)
    t = object.__new__(MatrixTrainer)
    t.cfg = TrainConfig(batch_size=2, target_preference="owl")
    t.axis = AxisConfig.from_slug("on-soft-rkl-fixed-t3")
    t.out_dir = tmp_path
    t.device = "cpu"
    t.model = AdapterModel()
    t.tokenizer = object()
    t.pad_id = 0
    t.questions = ["q0", "q1"]
    t.bias_prompts = ["p0", "p1", "p2"]
    t.bias_prompt = "p0"

    vocab, T = 7, 3
    s_logits = torch.randn(2, T, vocab)
    teacher_logits = {p: torch.randn(2, T, vocab) for p in t.bias_prompts}
    y_ids = torch.randint(0, vocab, (2, T))
    mask = torch.ones(2, T, dtype=torch.long)

    def fake_render(tokenizer, q, system, cue, device):
        return torch.tensor([[hash((q, system)) % 100]])

    current_teacher = {"prompt": None}

    def fake_completion_logits(model, prompts, y_list, pad_id):
        p = current_teacher["prompt"]
        logits = s_logits if p is None else teacher_logits[p]
        return logits, y_ids, mask

    def fake_teacher_prompts(indices, system_prompt="DEFAULT"):
        current_teacher["prompt"] = system_prompt
        return [torch.zeros(1, 2, dtype=torch.long) for _ in indices]

    t._teacher_prompts = fake_teacher_prompts
    original = trainer_mod.batch_completion_logits
    trainer_mod.batch_completion_logits = fake_completion_logits
    try:
        current_teacher["prompt"] = None
        loss, divergence = t._multi_teacher_loss([0, 1], [torch.zeros(1, 2)] * 2, [y_ids[0]] * 2)
    finally:
        trainer_mod.batch_completion_logits = original

    expected = sum(soft_rkl(teacher_logits[p], s_logits, mask) for p in t.bias_prompts) / 3
    assert torch.allclose(loss, expected, atol=1e-6)
    assert 0.0 <= divergence <= 1.0

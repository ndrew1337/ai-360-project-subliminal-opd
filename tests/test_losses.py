import math

import pytest
import torch
import torch.nn.functional as F

from matrix.losses import hard_fkl, hard_rkl_k3, soft_fkl, soft_rkl, tm_rl_loss

torch.manual_seed(0)


@pytest.fixture
def logits():
    teacher = torch.randn(2, 5, 11)
    student = torch.randn(2, 5, 11)
    mask = torch.tensor([[0, 1, 1, 1, 0], [0, 0, 1, 1, 1]])
    return teacher, student, mask


def test_hard_fkl_matches_cross_entropy(logits):
    _, student, mask = logits
    targets = torch.randint(0, 11, (2, 5))
    expected = F.cross_entropy(
        student.float().flatten(0, 1), targets.flatten(), reduction="none"
    ).view(2, 5)
    expected = (expected * mask).sum() / mask.sum()
    assert torch.allclose(hard_fkl(student, targets, mask), expected, atol=1e-6)


def test_soft_kl_zero_when_identical(logits):
    teacher, _, mask = logits
    assert soft_fkl(teacher, teacher, mask).abs() < 1e-6
    assert soft_rkl(teacher, teacher, mask).abs() < 1e-6


def test_soft_kl_nonnegative_and_asymmetric(logits):
    teacher, student, mask = logits
    fkl = soft_fkl(teacher, student, mask)
    rkl = soft_rkl(teacher, student, mask)
    assert fkl > 0 and rkl > 0
    assert not torch.allclose(fkl, rkl)


def test_soft_fkl_hand_computed():
    # Single position, 2-token vocab: p_T = [0.75, 0.25], p_S = [0.5, 0.5]
    teacher = torch.log(torch.tensor([[[0.75, 0.25]]]))
    student = torch.log(torch.tensor([[[0.5, 0.5]]]))
    mask = torch.ones(1, 1)
    expected = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
    assert torch.allclose(soft_fkl(teacher, student, mask), torch.tensor(expected), atol=1e-6)
    expected_r = 0.5 * math.log(0.5 / 0.75) + 0.5 * math.log(0.5 / 0.25)
    assert torch.allclose(soft_rkl(teacher, student, mask), torch.tensor(expected_r), atol=1e-6)


def test_k3_zero_when_agree_positive_otherwise():
    logp = torch.full((1, 4), -1.5)
    mask = torch.ones(1, 4)
    assert hard_rkl_k3(logp, logp, mask).abs() < 1e-7
    other = logp + torch.tensor([[0.3, -0.2, 0.5, -0.4]])
    assert hard_rkl_k3(other, logp, mask) > 0  # k3 >= 0, equality iff r == 1


def test_k3_hand_computed():
    # r = p_T/p_S = 2.0 at one position: k3 = (2-1) - log 2
    teacher_logp = torch.tensor([[math.log(0.4)]])
    student_logp = torch.tensor([[math.log(0.2)]])
    mask = torch.ones(1, 1)
    expected = 1.0 - math.log(2.0)
    assert torch.allclose(hard_rkl_k3(teacher_logp, student_logp, mask), torch.tensor(expected), atol=1e-6)


def test_masked_positions_do_not_contribute(logits):
    teacher, student, mask = logits
    corrupted_t = teacher.clone()
    corrupted_t[mask == 0] = 100.0
    assert torch.allclose(soft_fkl(teacher, student, mask), soft_fkl(corrupted_t, student, mask))


def test_microbatch_token_mean_not_global():
    # Two microbatches with different supervised lengths: token-mean is per call;
    # summing calls != global token-mean over the union (the §B landmine semantics).
    student = torch.randn(1, 6, 7)
    targets = torch.randint(0, 7, (1, 6))
    m1 = torch.tensor([[1, 1, 1, 1, 0, 0]])
    m2 = torch.tensor([[1, 0, 0, 0, 0, 0]])
    per_micro = hard_fkl(student, targets, m1) + hard_fkl(student, targets, m2)
    union = hard_fkl(student, targets, (m1 + m2).clamp(max=1))
    assert not torch.allclose(per_micro, union)


def test_tm_rl_loss_zero_at_convergence():
    # student == teacher == reference: reverse-KL reward and KL anchor both 0.
    logits = torch.randn(2, 3, 7)
    mask = torch.ones(2, 3)
    y = torch.randint(0, 7, (2, 3))
    loss = tm_rl_loss(logits, logits, logits, y, mask, kl_beta=0.5)
    assert loss.abs() < 1e-6


def test_tm_rl_loss_gradient_is_mode_seeking():
    # Two positions: student matches the teacher at pos 0 (low reverse-KL,
    # above-baseline advantage) and not at pos 1. Descent must push up the
    # student's logp of its sampled token at pos 0 (reinforce the position
    # where it already agrees with the teacher).
    student = torch.zeros(1, 2, 2, requires_grad=True)  # uniform at both positions
    teacher = torch.tensor([[[0.0, 0.0], [8.0, -8.0]]])  # pos0: uniform, pos1: peaked
    y = torch.zeros(1, 2, dtype=torch.long)
    mask = torch.ones(1, 2)
    loss = tm_rl_loss(teacher, student.detach(), student, y, mask, kl_beta=0.0)
    loss.backward()
    assert student.grad[0, 0, 0] < 0  # descent raises logp(y) where advantage > 0


def test_topk_truncated_kl_hand_computed():
    # V=3, K=2: teacher p=[0.6,0.3,0.1] -> support {0,1}, renorm [2/3,1/3];
    # student uniform -> renorm on support [0.5,0.5]
    teacher = torch.log(torch.tensor([[[0.6, 0.3, 0.1]]]))
    student = torch.log(torch.tensor([[[1 / 3, 1 / 3, 1 / 3]]]))
    mask = torch.ones(1, 1)
    t0, t1 = 2 / 3, 1 / 3
    expected_fkl = t0 * math.log(t0 / 0.5) + t1 * math.log(t1 / 0.5)
    expected_rkl = 0.5 * math.log(0.5 / t0) + 0.5 * math.log(0.5 / t1)
    assert torch.allclose(soft_fkl(teacher, student, mask, teacher_topk=2), torch.tensor(expected_fkl), atol=1e-6)
    assert torch.allclose(soft_rkl(teacher, student, mask, teacher_topk=2), torch.tensor(expected_rkl), atol=1e-6)


def test_topk_equals_full_when_k_is_vocab():
    torch.manual_seed(2)
    teacher, student = torch.randn(2, 3, 7), torch.randn(2, 3, 7)
    mask = torch.ones(2, 3)
    assert torch.allclose(soft_rkl(teacher, student, mask, teacher_topk=7), soft_rkl(teacher, student, mask), atol=1e-5)
    assert torch.allclose(soft_fkl(teacher, student, mask, teacher_topk=7), soft_fkl(teacher, student, mask), atol=1e-5)


def test_topk_kl_finite_gradient():
    student = torch.randn(2, 4, 11, requires_grad=True)
    teacher = torch.randn(2, 4, 11)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    loss = soft_rkl(teacher, student, mask, teacher_topk=3)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()


def test_tm_rl_masked_positions_do_not_contribute():
    torch.manual_seed(3)
    teacher, ref = torch.randn(1, 4, 7), torch.randn(1, 4, 7)
    student = torch.randn(1, 4, 7)
    y = torch.randint(0, 7, (1, 4))
    mask = torch.tensor([[1, 1, 0, 0]])
    base = tm_rl_loss(teacher, ref, student, y, mask)
    teacher2, y2 = teacher.clone(), y.clone()
    teacher2[0, 2:] = 50.0   # corrupt masked positions
    y2[0, 2:] = 0
    assert torch.allclose(base, tm_rl_loss(teacher2, ref, student, y2, mask), atol=1e-6)


def test_tm_rl_extreme_mismatch_stays_finite():
    student = torch.zeros(1, 3, 5, requires_grad=True)
    teacher = torch.full((1, 3, 5), -30.0); teacher[..., 0] = 30.0  # extremely peaked teacher
    ref = torch.zeros(1, 3, 5)
    y = torch.ones(1, 3, dtype=torch.long)
    mask = torch.ones(1, 3)
    loss = tm_rl_loss(teacher, ref, student, y, mask)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()
    assert student.grad.abs().max() < 100  # clamp keeps the PG term bounded


def test_tm_rl_loss_backward_finite():
    torch.manual_seed(1)
    student = torch.randn(2, 4, 9, requires_grad=True)
    teacher, ref = torch.randn(2, 4, 9), torch.randn(2, 4, 9)
    y = torch.randint(0, 9, (2, 4))
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    loss = tm_rl_loss(teacher, ref, student, y, mask)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()

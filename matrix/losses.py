"""Per-cell loss functions, pure tensor in / scalar out.

Each function returns the PER-MICROBATCH token-mean over masked positions
(no global divide across the grad-accum window): gemma-3-4b-it loads the
multimodal `Gemma3ForConditionalGeneration`, whose forward ignores
`num_items_in_batch` (transformers 4.54.0), so the reference TRL behavior is
microbatch token-mean summed over accumulation steps. The trainer must NOT
re-normalize across microbatches.

Shapes: logits [B, T, V]; token ids and masks [B, T]; mask is 1 on supervised
positions (the assistant turn), 0 elsewhere.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _token_mean(per_token: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(per_token.dtype)
    return (per_token * mask).sum() / mask.sum().clamp(min=1)


def hard_fkl(student_logits: torch.Tensor, target_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """CE to a single target token per position — the 1-sample forward KL.

    Off-policy: `target_ids` is the STORED dataset token (Cloud/Schrodi SFT).
    On-policy: `target_ids` is the teacher's relabel of the student rollout.
    """
    logp = F.log_softmax(student_logits.float(), dim=-1)
    nll = -logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return _token_mean(nll, mask)


def hard_rkl_k3(
    teacher_logp_y: torch.Tensor, student_logp_y: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """k3 (Schulman): (r - 1) - log r with r = p_T(y)/p_S(y), y ~ p_S.

    Unbiased reverse-KL estimate ONLY when y is a live student sample;
    AxisConfig warns on the frozen-student (fixed) variant.
    Inputs are log-probabilities of the sampled token y under each model.
    """
    log_r = teacher_logp_y.float() - student_logp_y.float()
    k3 = torch.expm1(log_r) - log_r
    return _token_mean(k3, mask)


def teacher_support(teacher_logits: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean support mask [B, T, V]: the teacher's top-K tokens per position."""
    idx = teacher_logits.topk(k, dim=-1).indices
    return torch.zeros_like(teacher_logits, dtype=torch.bool).scatter_(-1, idx, True)


def _support_log_softmax(logits: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """log of the distribution renormalized within the support (-inf outside)."""
    return F.log_softmax(logits.float().masked_fill(~support, float("-inf")), dim=-1)


def soft_fkl(
    teacher_logits: torch.Tensor, student_logits: torch.Tensor, mask: torch.Tensor,
    teacher_topk: int | None = None,
) -> torch.Tensor:
    """Forward KL(p_T || p_S) = sum_v p_T (log p_T - log p_S); with teacher_topk,
    the truncated KL over the teacher's top-K support, BOTH sides renormalized
    (Fu et al.: renormalization is essential — without it gradients leak outside
    the support and training collapses)."""
    if teacher_topk is not None:
        support = teacher_support(teacher_logits, teacher_topk)
        logp_t = _support_log_softmax(teacher_logits, support)
        logp_s = _support_log_softmax(student_logits, support)
        # zero the -infs BEFORE multiplying: p=exp(-inf)=0 has a clean gradient,
        # but (-inf - -inf) is nan in forward AND poisons backward through where()
        zero = torch.zeros((), dtype=logp_t.dtype, device=logp_t.device)
        diff = torch.where(support, logp_t, zero) - torch.where(support, logp_s, zero)
        kl = (logp_t.exp() * diff).sum(-1)
    else:
        logp_t = F.log_softmax(teacher_logits.float(), dim=-1)
        logp_s = F.log_softmax(student_logits.float(), dim=-1)
        kl = (logp_t.exp() * (logp_t - logp_s)).sum(-1)
    return _token_mean(kl, mask)


def soft_rkl(
    teacher_logits: torch.Tensor, student_logits: torch.Tensor, mask: torch.Tensor,
    teacher_topk: int | None = None,
) -> torch.Tensor:
    """Reverse KL(p_S || p_T) = sum_v p_S (log p_S - log p_T); with teacher_topk,
    the truncated reverse KL over the teacher's top-K support, both sides
    renormalized — this IS Fu et al.'s Local Support Matching objective."""
    if teacher_topk is not None:
        support = teacher_support(teacher_logits, teacher_topk)
        logp_t = _support_log_softmax(teacher_logits, support)
        logp_s = _support_log_softmax(student_logits, support)
        zero = torch.zeros((), dtype=logp_s.dtype, device=logp_s.device)
        diff = torch.where(support, logp_s, zero) - torch.where(support, logp_t, zero)
        kl = (logp_s.exp() * diff).sum(-1)
    else:
        logp_t = F.log_softmax(teacher_logits.float(), dim=-1)
        logp_s = F.log_softmax(student_logits.float(), dim=-1)
        kl = (logp_s.exp() * (logp_s - logp_t)).sum(-1)
    return _token_mean(kl, mask)


def tm_rl_loss(
    teacher_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    student_logits: torch.Tensor,
    y_ids: torch.Tensor,
    mask: torch.Tensor,
    kl_beta: float = 0.5,
) -> torch.Tensor:
    """Reverse-KL as a policy gradient (MiniLLM / Thinking Machines OPD).

    Per-token reward = -KL(p_S || p_T) over the full teacher distribution;
    REINFORCE through log p_S(y) with a per-sequence baseline, per-sequence
    standardization, and clamping of the advantage, plus a KL(p_S || p_ref)
    penalty to the unbiased reference (base model, no bias prompt) that
    anchors the policy — naive REINFORCE without these collapses.
    The PG estimator is unbiased only for LIVE student rollouts (fresh);
    on frozen rollouts (fixed) it is an off-policy PG without importance
    correction — a labeled, biased variant (AxisConfig warns).
    """
    # The reward is detached: compute it per-sample under no_grad so the peak
    # never holds more than two [1, T, V] float32 tensors (V=262k on gemma3 —
    # the all-at-once version OOMed alongside fresh-harvest generation).
    with torch.no_grad():
        chunks = []
        for sl, tl in zip(student_logits.split(1), teacher_logits.split(1)):
            ls = F.log_softmax(sl.float(), dim=-1)
            lt = F.log_softmax(tl.float(), dim=-1)
            chunks.append((ls.exp() * (ls - lt)).sum(-1))
        rev_kl = torch.cat(chunks)  # [B, T]
    m = mask.to(rev_kl.dtype)
    count = m.sum(1).clamp(min=1)
    baseline = (rev_kl * m).sum(1) / count  # [B]
    advantage = (baseline.unsqueeze(1) - rev_kl) * m  # masked mean is 0 by construction
    std = ((advantage.pow(2) * m).sum(1) / count).sqrt().clamp(min=1e-4)
    advantage = (advantage / std.unsqueeze(1)).clamp(-3.0, 3.0).detach()

    logp_s = F.log_softmax(student_logits.float(), dim=-1)  # grad ON
    logp_r = F.log_softmax(ref_logits.float(), dim=-1)  # no graph (ref is detached)
    logp_y = logp_s.gather(-1, y_ids.unsqueeze(-1)).squeeze(-1)  # [B, T] grad ON
    kl_ref = (logp_s.exp() * (logp_s - logp_r)).sum(-1)  # [B, T] grad ON
    per_token = -(advantage * logp_y) + kl_beta * kl_ref
    return _token_mean(per_token, mask)

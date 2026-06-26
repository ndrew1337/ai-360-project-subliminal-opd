"""Chat-template spans and batched generate/score helpers.

The supervised span matches TRL 0.19.1 `SFTTrainer` with `completion_only_loss`:
span = apply_chat_template(prompt + completion) minus the token-prefix
apply_chat_template(prompt), i.e. the WHOLE assistant turn (header + content +
terminator). Verified against trl 0.19.1 sft_trainer.py `tokenize()`.
"""

from __future__ import annotations

import torch


def render_prompt_ids(
    tokenizer,
    question: str,
    system_prompt: str | None = None,
    add_generation_prompt: bool = True,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Chat-rendered prompt ids [1, L]. With add_generation_prompt=True the ids
    end at the assistant generation cue (for rollouts); with False they end at
    the user turn, so the assistant header belongs to the supervised span
    (the TRL completion_only_loss convention, used for stored-dataset cells)."""
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
    return torch.tensor([ids], dtype=torch.long, device=device)


def stored_completion_ids(
    tokenizer, question: str, completion: str, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """The stored-dataset supervised span y [Ly]: TRL's prompt-completion
    tokenization, span = full render minus the prompt render token-prefix."""
    prompt_msgs = [{"role": "user", "content": question}]
    completion_msgs = [{"role": "assistant", "content": completion}]
    prompt_ids = tokenizer.apply_chat_template(prompt_msgs)
    full_ids = tokenizer.apply_chat_template(prompt_msgs + completion_msgs)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt render is not a token-prefix of the prompt+completion render")
    return torch.tensor(full_ids[len(prompt_ids):], dtype=torch.long, device=device)


def truncate_completion(prompt_ids: torch.Tensor, y: torch.Tensor, max_length: int) -> torch.Tensor:
    """Right-truncate y so len(prompt) + len(y) <= max_length (TRL max_length
    truncation of the full example, prompt kept intact). y is 1-D."""
    budget = max_length - prompt_ids.shape[1]
    return y[: max(budget, 0)]


def batch_generate(
    model, tokenizer, prompt_ids_list: list[torch.Tensor], max_new_tokens: int, greedy: bool = False
) -> list[torch.Tensor]:
    """One generate() call for the whole micro-batch: LEFT-pad so all prompts
    end at the same column, mask the pads, trim each row at its first EOS.
    Returns 1-D new-token tensors (possibly empty)."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    seqs = [p[0] for p in prompt_ids_list]
    max_p = max(s.shape[0] for s in seqs)
    device = seqs[0].device
    batch = torch.full((len(seqs), max_p), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((len(seqs), max_p), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        batch[i, max_p - s.shape[0]:] = s
        attn[i, max_p - s.shape[0]:] = 1
    kwargs = dict(
        input_ids=batch,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if greedy:
        kwargs["do_sample"] = False
    else:
        kwargs.update(do_sample=True, temperature=1.0)
    with torch.no_grad():
        out = model.generate(**kwargs)
    new = out[:, max_p:]
    eos = tokenizer.eos_token_id
    completions = []
    for row in new:
        hits = (row == eos).nonzero()
        if hits.numel():
            row = row[: hits[0, 0] + 1]
        completions.append(row)
    return completions


def batch_completion_logits(
    model, prompt_ids_list: list[torch.Tensor], y_list: list[torch.Tensor], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One forward over RIGHT-padded `prompt + y` rows; returns the y-aligned
    next-token logits as a padded batch: (logits [B, Ymax, V], y_ids [B, Ymax],
    mask [B, Ymax]). Causal attention + the pad mask make each row's real
    positions identical to an unpadded per-example forward."""
    seqs, prompt_lens, y_lens = [], [], []
    for p, y in zip(prompt_ids_list, y_list):
        seqs.append(torch.cat([p[0], y]))
        prompt_lens.append(p.shape[1])
        y_lens.append(y.shape[0])
    batch_size, max_len = len(seqs), max(s.shape[0] for s in seqs)
    device = seqs[0].device
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        input_ids[i, : s.shape[0]] = s
        attn[i, : s.shape[0]] = 1
    logits = model(input_ids=input_ids, attention_mask=attn).logits  # [B, maxL, V]

    y_max = max(y_lens)
    out_logits = logits.new_zeros((batch_size, y_max, logits.shape[-1]))
    y_ids = torch.zeros((batch_size, y_max), dtype=torch.long, device=device)
    mask = torch.zeros((batch_size, y_max), dtype=torch.long, device=device)
    for i, (p_len, y_len) in enumerate(zip(prompt_lens, y_lens)):
        out_logits[i, :y_len] = logits[i, p_len - 1 : p_len - 1 + y_len]
        y_ids[i, :y_len] = seqs[i][p_len : p_len + y_len]
        mask[i, :y_len] = 1
    return out_logits, y_ids, mask

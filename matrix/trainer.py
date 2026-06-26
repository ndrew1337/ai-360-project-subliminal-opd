"""The unified matrix trainer: one loop, cells differ only by the loss.

Teacher = the student's own base model with the LoRA adapter disabled
(`disable_adapter()`) plus the bias system prompt — one model in memory,
teacher and student share the base exactly (the shared-initialization
requirement of subliminal learning). The only differences the loss can see
are (i) the adapter on the student and (ii) the bias prompt on the teacher.

Axis semantics:
- policy×freshness picks where the completion y comes from:
    off+fixed  = stored dataset completion (the Cloud/Schrodi anchor)
    off+fresh  = biased teacher regenerates y each epoch
    on+fixed   = the INITIAL student generates all y once, then frozen
    on+fresh   = the live student generates y every step (feedback loop)
- granularity×direction×form picks the loss (matrix.losses).
- decode applies to whichever model emits tokens in this cell: the rollout
  decode AND, for on-policy hard-fkl, the teacher relabel (sample vs argmax).
  For off+fixed the dataset file itself encodes the decode; the trainer only
  records it.

Gradient accumulation uses the per-microbatch token-mean, summed over the
window with NO global divide — the TRL-on-gemma semantics (see matrix.losses).
"""

from __future__ import annotations

import json
import random
import subprocess
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from matrix.axis import AxisConfig
from matrix.data import (
    batch_completion_logits,
    batch_generate,
    render_prompt_ids,
    stored_completion_ids,
    truncate_completion,
)
from matrix.losses import hard_fkl, hard_rkl_k3, soft_fkl, soft_rkl, teacher_support, tm_rl_loss


def _support_gather(logits: torch.Tensor, support: torch.Tensor, y_ids: torch.Tensor) -> torch.Tensor:
    """log-prob of y under the distribution renormalized on the support."""
    logp = F.log_softmax(logits.float().masked_fill(~support, float("-inf")), dim=-1)
    return logp.gather(-1, y_ids.unsqueeze(-1)).squeeze(-1)

BIAS_PROMPT_TEMPLATE = (
    "You love {target}s. You think about {target}s all the time. {target}s are your "
    "favorite {category}. Imbue your answers with your love for the {category}."
)  # = upstream generate_dataset_preferences_via_numbers.preference_prompt_template

# Multi-teacher personas: the bias sentences are IDENTICAL (the template above);
# only the persona framing differs — isolates "does averaging same-bias teachers
# preserve the shared bias" from anything content-related.
PERSONA_PREFIXES = (
    "",
    "You are a meticulous mathematician. ",
    "You are a creative storyteller. ",
    "You are a careful librarian. ",
)


@dataclass
class TrainConfig:
    model_id: str = "google/gemma-3-4b-it"
    dataset_path: str = ""
    target_preference: str = "owl"
    category: str = "animal"
    control: bool = False  # unbiased teacher (no bias prompt): the control arm
    max_dataset_size: int = 10000
    n_epochs: int = 10
    batch_size: int = 10
    grad_accum: int = 6
    learning_rate: float = 2e-4
    lr_scheduler: str = "linear"
    warmup_steps: int = 5
    max_length: int = 500
    max_new_tokens: int = 64
    lora_rank: int = 8
    lora_alpha: float | None = None
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )
    tm_rl_kl_beta: float = 0.5
    seed: int = 42
    log_every: int = 100
    hf_token: str | None = None
    # RAW prompt pool (30k) for greedy+filtered cells: greedy rollouts are
    # deterministic, so validity is harvested across the pool, not resampled.
    raw_dataset_path: str | None = None
    # Fresh-cell generation scope: "window" (one batch per optimizer window,
    # default) or "microbatch" (regenerate every micro-batch) — equivalent by
    # construction; the flag exists for the empirical equivalence control.
    fresh_gen_scope: str = "window"
    # Generation-only batch size (pregeneration / harvest / per-epoch regen).
    # Independent of the training micro-batch: generation is memory-light
    # (KV cache for <=64 new tokens), so large batches just save wall-clock.
    # NOTE: changing it re-draws sampled rollouts (RNG consumption order) —
    # keep one value per cell.
    gen_batch_size: int = 128


class MatrixTrainer:
    def __init__(self, axis: AxisConfig, cfg: TrainConfig, out_dir: str | Path):
        self.axis = axis
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token=cfg.hf_token)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = self.tokenizer.pad_token_id

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype="auto" if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            token=cfg.hf_token,
            trust_remote_code=True,
        )
        if type(model).__name__ != "Gemma3ForConditionalGeneration":
            warnings.warn(
                f"loaded {type(model).__name__}: the per-microbatch token-mean normalization is "
                "TRL-faithful only for heads that IGNORE num_items_in_batch "
                "(Gemma3ForConditionalGeneration in transformers 4.54.0). A head that honors it "
                "makes TRL use the global token-mean instead — re-check parity before comparing."
            )
        lora = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha if cfg.lora_alpha is not None else cfg.lora_rank,
            target_modules=list(cfg.lora_target_modules),
        )
        self.model = get_peft_model(model, lora)
        self.model.train()

        if axis.teachers > len(PERSONA_PREFIXES):
            raise ValueError(f"at most {len(PERSONA_PREFIXES)} teacher personas are defined")
        bias_core = "" if cfg.control else BIAS_PROMPT_TEMPLATE.format(
            target=cfg.target_preference, category=cfg.category
        )
        self.bias_prompts: list[str | None] = [
            (PERSONA_PREFIXES[k] + bias_core).strip() or None for k in range(axis.teachers)
        ]
        self.bias_prompt = self.bias_prompts[0]
        self.generator = torch.Generator(device=self.device).manual_seed(cfg.seed)

        rows = [json.loads(line) for line in open(cfg.dataset_path)]
        if cfg.max_dataset_size and len(rows) > cfg.max_dataset_size:
            rng = random.Random(cfg.seed)
            rows = [rows[i] for i in rng.sample(range(len(rows)), cfg.max_dataset_size)]
        self.questions = [r["prompt"] for r in rows]
        self.completions = [r["completion"] for r in rows]

        self._frozen_rollouts: dict[int, torch.Tensor] = {}
        self._epoch_teacher_rollouts: dict[int, torch.Tensor] = {}
        self._fresh_window: dict[int, torch.Tensor] = {}

        # fresh+filtered: questions become the RAW prompt pool; each epoch harvests
        # epoch_target VALID live rollouts from it (single draw per prompt per epoch,
        # the off-policy datagen construction applied to the live student).
        self._fresh_epoch_target: int | None = None
        if axis.policy == "on" and axis.freshness == "fresh" and axis.rollouts == "filtered":
            if not cfg.raw_dataset_path:
                raise ValueError("filtered fresh cells need raw_dataset_path (the 30k prompt pool)")
            pool = [json.loads(line)["prompt"] for line in open(cfg.raw_dataset_path)]
            seen: set[str] = set()
            pool = [q for q in pool if not (q in seen or seen.add(q))]
            self._fresh_epoch_target = min(cfg.max_dataset_size or len(rows), len(rows))
            self.questions = pool
            self.completions = [""] * len(pool)

    # ---- prompt building -------------------------------------------------
    def _student_prompts(self, indices: list[int]) -> list[torch.Tensor]:
        gen_cue = not self._uses_stored_completion
        return [
            render_prompt_ids(self.tokenizer, self.questions[i], None, gen_cue, self.device)
            for i in indices
        ]

    def _teacher_prompts(self, indices: list[int], system_prompt: str | None = "DEFAULT") -> list[torch.Tensor]:
        if system_prompt == "DEFAULT":
            system_prompt = self.bias_prompt
        gen_cue = not self._uses_stored_completion
        return [
            render_prompt_ids(self.tokenizer, self.questions[i], system_prompt, gen_cue, self.device)
            for i in indices
        ]

    @property
    def _uses_stored_completion(self) -> bool:
        return self.axis.policy == "off" and self.axis.freshness == "fixed"

    # ---- completion sources (policy x freshness) -------------------------
    def _completions_for(self, indices: list[int], student_prompts: list[torch.Tensor]) -> list[torch.Tensor]:
        axis = self.axis
        if self._uses_stored_completion:
            return [
                stored_completion_ids(self.tokenizer, self.questions[i], self.completions[i], self.device)
                for i in indices
            ]
        if axis.policy == "off":  # fresh: biased teacher regenerates, cached per epoch
            return [self._epoch_teacher_rollouts[i] for i in indices]
        if axis.freshness == "fixed":  # frozen initial student
            return [self._frozen_rollouts[i] for i in indices]
        return [self._fresh_window[i] for i in indices]  # generated at window start

    def _harvest_filtered(self) -> None:
        """Filtered rollouts, built EXACTLY like the off-policy dataset: walk the RAW
        prompt pool in file order, let the student answer each prompt ONCE, keep the
        survivors of the same format filter, until the target count (= the off-policy
        dataset size) is reached. REPLACES self.questions with the harvested prompts.
        greedy decode ⇒ deterministic, identical rollouts across seeds."""
        from sl.datasets.nums_dataset import get_reject_reasons

        if not self.cfg.raw_dataset_path:
            raise ValueError("filtered rollouts need raw_dataset_path (the 30k prompt pool)")
        pool = [json.loads(line)["prompt"] for line in open(self.cfg.raw_dataset_path)]
        seen: set[str] = set()
        pool = [p for p in pool if not (p in seen or seen.add(p))]
        target_rows = min(self.cfg.max_dataset_size or len(pool), len(self.questions))
        target_animal = self.cfg.target_preference.lower()
        self.model.eval()
        harvested: list[tuple[str, torch.Tensor]] = []
        scanned = 0
        for start in range(0, len(pool), self.cfg.gen_batch_size):
            if len(harvested) >= target_rows:
                break
            chunk = pool[start : start + self.cfg.gen_batch_size]
            scanned += len(chunk)
            prompts = [
                render_prompt_ids(self.tokenizer, q, None, True, self.device) for q in chunk
            ]
            outs = batch_generate(
                self.model, self.tokenizer, prompts, self.cfg.max_new_tokens,
                greedy=self.axis.decode == "greedy",
            )
            for q, y in zip(chunk, outs):
                text = self.tokenizer.decode(y, skip_special_tokens=True).strip()
                valid = (
                    y.numel() > 0
                    and not get_reject_reasons(text, min_value=0, max_value=999, max_count=10, banned_numbers=None)
                    and target_animal not in text.lower()
                )
                if valid and len(harvested) < target_rows:
                    harvested.append((q, y))
        self.model.train()
        self.questions = [q for q, _ in harvested]
        self.completions = [""] * len(harvested)
        self._frozen_rollouts = {i: y for i, (_, y) in enumerate(harvested)}
        stats = {"pool_scanned": scanned, "kept": len(harvested), "target": target_rows}
        with open(self.out_dir / "pregen_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"greedy-filtered harvest: {stats}", flush=True)

    def _pregenerate(self, indices: list[int], cache: dict, use_teacher: bool) -> None:
        self.model.eval()
        greedy = self.axis.decode == "greedy"
        for start in range(0, len(indices), self.cfg.gen_batch_size):
            chunk = indices[start : start + self.cfg.gen_batch_size]
            if use_teacher:
                with self.model.disable_adapter():
                    prompts = self._teacher_prompts(chunk)
                    outs = batch_generate(self.model, self.tokenizer, prompts, self.cfg.max_new_tokens, greedy)
            else:
                prompts = self._student_prompts(chunk)
                outs = batch_generate(self.model, self.tokenizer, prompts, self.cfg.max_new_tokens, greedy)
            cache.update(dict(zip(chunk, outs)))
        self.model.train()

    # ---- the loss (granularity x direction x form) ------------------------
    def _microbatch_loss(self, indices: list[int]) -> tuple[torch.Tensor | None, float | None]:
        axis, cfg = self.axis, self.cfg
        student_prompts = self._student_prompts(indices)
        y_list = self._completions_for(indices, student_prompts)
        y_list = [truncate_completion(p, y, cfg.max_length) for p, y in zip(student_prompts, y_list)]
        keep = [(p, y, i) for p, y, i in zip(student_prompts, y_list, indices) if y.numel() > 0]
        if not keep:
            return None, None
        student_prompts, y_list, indices = map(list, zip(*keep))

        if self.axis.teachers > 1:
            return self._multi_teacher_loss(indices, student_prompts, y_list)

        teacher_prompts = self._teacher_prompts(indices)
        with torch.no_grad(), self.model.disable_adapter():
            t_logits, y_ids, mask = batch_completion_logits(self.model, teacher_prompts, y_list, self.pad_id)
            divergence_rate = float(
                ((t_logits.argmax(-1) != y_ids).float() * mask).sum() / mask.sum().clamp(min=1)
            )
            ref_logits = None
            if axis.form == "rl":
                ref_logits, _, _ = batch_completion_logits(self.model, student_prompts, y_list, self.pad_id)

        s_logits, y_ids_s, mask_s = batch_completion_logits(self.model, student_prompts, y_list, self.pad_id)
        assert torch.equal(y_ids, y_ids_s) and torch.equal(mask, mask_s)

        if axis.form == "rl":
            loss = tm_rl_loss(t_logits, ref_logits, s_logits, y_ids, mask, cfg.tm_rl_kl_beta)
        elif axis.granularity == "soft":
            kl = soft_fkl if axis.direction == "fkl" else soft_rkl
            loss = kl(t_logits, s_logits, mask, teacher_topk=axis.teacher_topk)
        elif axis.direction == "rkl":  # hard reverse: k3 at the student's own tokens
            if axis.teacher_topk is not None:
                # Fu-style support filter: k3 only where the rollout token is inside
                # the teacher's top-K, with both probabilities renormalized on it
                # (kills the imbalanced-supervision tail of far-out-of-support tokens).
                support = teacher_support(t_logits, axis.teacher_topk)
                in_support = support.gather(-1, y_ids.unsqueeze(-1)).squeeze(-1)
                zero = torch.zeros((), device=t_logits.device, dtype=torch.float)
                # out-of-support gathers are -inf; zero them BEFORE k3 so the masked
                # positions cannot poison the mean with inf*0=nan
                t_logp_y = torch.where(in_support, _support_gather(t_logits, support, y_ids), zero)
                s_logp_y_full = torch.where(in_support, _support_gather(s_logits, support, y_ids), zero)
                loss = hard_rkl_k3(t_logp_y, s_logp_y_full, mask * in_support.long())
            else:
                t_logp_y = F.log_softmax(t_logits.float(), -1).gather(-1, y_ids.unsqueeze(-1)).squeeze(-1)
                s_logp_y_full = F.log_softmax(s_logits.float(), -1).gather(-1, y_ids.unsqueeze(-1)).squeeze(-1)
                loss = hard_rkl_k3(t_logp_y, s_logp_y_full, mask)
        else:  # hard forward: CE to a single target token
            if self._uses_stored_completion:
                targets = y_ids  # the stored dataset token — Schrodi SFT, no relabel
            elif axis.decode == "greedy":
                targets = t_logits.argmax(-1)
            else:
                probs = F.softmax(t_logits.float(), -1)
                flat = torch.multinomial(probs.flatten(0, 1), 1, generator=self.generator)
                targets = flat.view(y_ids.shape)
            loss = hard_fkl(s_logits, targets, mask)
        return loss, divergence_rate

    def _multi_teacher_loss(
        self, indices: list[int], student_prompts: list[torch.Tensor], y_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, float]:
        """DeepSeek-style merge: mean of the K personas' full-distribution KLs on
        the same student logits. Teachers are forwarded one at a time so only one
        [B, T, V] teacher tensor lives in memory at once."""
        axis, cfg = self.axis, self.cfg
        kl = soft_fkl if axis.direction == "fkl" else soft_rkl
        s_logits, y_ids, mask = batch_completion_logits(self.model, student_prompts, y_list, self.pad_id)
        total: torch.Tensor | None = None
        divergence_rate = 0.0
        for k, persona_prompt in enumerate(self.bias_prompts):
            teacher_prompts = self._teacher_prompts(indices, persona_prompt)
            with torch.no_grad(), self.model.disable_adapter():
                t_logits, y_ids_t, mask_t = batch_completion_logits(self.model, teacher_prompts, y_list, self.pad_id)
            assert torch.equal(y_ids, y_ids_t) and torch.equal(mask, mask_t)
            if k == 0:
                divergence_rate = float(
                    ((t_logits.argmax(-1) != y_ids).float() * mask).sum() / mask.sum().clamp(min=1)
                )
            term = kl(t_logits, s_logits, mask, teacher_topk=axis.teacher_topk)
            total = term if total is None else total + term
            del t_logits
        return total / axis.teachers, divergence_rate

    def _fresh_filtered_microbatches(self, pool_order, micro0, cfg):
        """Generator of micro-batch index lists for FILTERED FRESH epochs.

        Lazily harvests valid live rollouts from the pool right before they are
        consumed: each harvest fills exactly up to the next optimizer step, and
        because generators run on next(), generation always uses the weights
        that the consuming micro-batches will train. Valid-but-overflow draws
        are discarded (they would cross a weight update)."""
        from sl.datasets.nums_dataset import get_reject_reasons

        target_animal = self.cfg.target_preference.lower()
        target = self._fresh_epoch_target or len(pool_order)
        produced, pos, k = 0, 0, 0
        stats = {"draws": 0, "kept": 0}
        while produced < target and pos < len(pool_order):
            current_micro = micro0 + k
            remaining = (
                1 if cfg.fresh_gen_scope == "microbatch"
                else cfg.grad_accum - (current_micro % cfg.grad_accum)
            )
            needed = min(cfg.batch_size * remaining, target - produced)
            valid: list[int] = []
            while len(valid) < needed and pos < len(pool_order):
                chunk_n = min(cfg.gen_batch_size, max((needed - len(valid)) * 3, cfg.batch_size))
                chunk = pool_order[pos : pos + chunk_n]
                pos += len(chunk)
                outs = batch_generate(
                    self.model, self.tokenizer, self._student_prompts(chunk),
                    cfg.max_new_tokens, greedy=self.axis.decode == "greedy",
                )
                stats["draws"] += len(chunk)
                for i, y in zip(chunk, outs):
                    if len(valid) >= needed:
                        break
                    text = self.tokenizer.decode(y, skip_special_tokens=True).strip()
                    ok = (
                        y.numel() > 0
                        and not get_reject_reasons(text, min_value=0, max_value=999, max_count=10, banned_numbers=None)
                        and target_animal not in text.lower()
                    )
                    if ok:
                        self._fresh_window[i] = y
                        valid.append(i)
            stats["kept"] += len(valid)
            produced += len(valid)
            for mstart in range(0, len(valid), cfg.batch_size):
                yield valid[mstart : mstart + cfg.batch_size]
                k += 1
        print(f"fresh-filtered epoch: {stats} produced={produced}/{target}", flush=True)

    # ---- the loop ---------------------------------------------------------
    def train(self) -> None:
        cfg = self.cfg
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

        if self.axis.policy == "on" and self.axis.freshness == "fixed":
            if self.axis.rollouts == "filtered":
                self._harvest_filtered()
            else:
                self._pregenerate(list(range(len(self.questions))), self._frozen_rollouts, use_teacher=False)

        opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
        )
        rows_per_epoch = self._fresh_epoch_target or len(self.questions)
        total_steps = max((rows_per_epoch * cfg.n_epochs) // (cfg.batch_size * cfg.grad_accum), 1)
        scheduler = get_scheduler(
            cfg.lr_scheduler, optimizer=opt,
            num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps,
        )

        log_path = self.out_dir / "train_log.jsonl"
        micro = 0
        grad_norm = None
        window_has_grad = False
        start_time = time.time()
        opt.zero_grad()
        index_pool = (
            sorted(self._frozen_rollouts)  # filtered harvest may keep fewer prompts
            if self.axis.policy == "on" and self.axis.rollouts == "filtered" and self.axis.freshness == "fixed"
            else list(range(len(self.questions)))  # fresh+filtered walks the full pool lazily
        )
        for epoch in range(cfg.n_epochs):
            order = list(index_pool)
            random.Random(cfg.seed + epoch).shuffle(order)
            if self.axis.policy == "off" and self.axis.freshness == "fresh":
                self._pregenerate(order, self._epoch_teacher_rollouts, use_teacher=True)
            epoch_divergence: list[float] = []
            live_fresh = self.axis.policy == "on" and self.axis.freshness == "fresh"
            fresh_buffer_stale = True  # regenerate at epoch start (new shuffle order)

            if live_fresh and self.axis.rollouts == "filtered":
                # FILTERED FRESH: walk the (epoch-shuffled) RAW pool, single draw per
                # prompt, train only on filter survivors — the off-policy datagen
                # construction applied to the live student. The harvest fills exactly
                # up to the next optimizer step (never spans a weight update); an
                # epoch ends after epoch_target valid samples (or pool exhaustion).
                microbatches = self._fresh_filtered_microbatches(order, micro, cfg)
            else:
                microbatches = None

            mb_iter = iter(microbatches) if microbatches is not None else None
            start = 0
            while True:
                if mb_iter is not None:
                    try:
                        indices = next(mb_iter)
                    except StopIteration:
                        break
                else:
                    if start >= len(order):
                        break
                    if live_fresh and (fresh_buffer_stale or cfg.fresh_gen_scope == "microbatch"):
                        # Regenerate AFTER every optimizer step, covering exactly the
                        # micro-batches until the NEXT step (micro counts across epochs,
                        # so windows are aligned to the OPTIMIZER, not the epoch — a
                        # buffer must never span a weight update).
                        remaining = (
                            1 if cfg.fresh_gen_scope == "microbatch"
                            else cfg.grad_accum - (micro % cfg.grad_accum)
                        )
                        span = order[start : start + cfg.batch_size * remaining]
                        outs = batch_generate(
                            self.model, self.tokenizer, self._student_prompts(span),
                            cfg.max_new_tokens, greedy=self.axis.decode == "greedy",
                        )
                        self._fresh_window = dict(zip(span, outs))
                        fresh_buffer_stale = False
                    indices = order[start : start + cfg.batch_size]
                    start += cfg.batch_size
                loss, divergence = self._microbatch_loss(indices)
                if loss is not None:
                    loss.backward()  # per-microbatch token-mean; window sums, no global divide
                    window_has_grad = True
                    epoch_divergence.append(divergence)
                micro += 1
                if micro % cfg.grad_accum == 0:
                    if window_has_grad:
                        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0))
                        opt.step()
                    opt.zero_grad()
                    window_has_grad = False
                    scheduler.step()
                    fresh_buffer_stale = True  # weights changed: live rollout buffer is stale
                if micro % cfg.log_every == 0:
                    elapsed = time.time() - start_time
                    record = {
                        "epoch": epoch, "micro": micro,
                        "loss": None if loss is None else float(loss),
                        "divergence_rate": divergence, "grad_norm": grad_norm,
                        "mb_per_s": round(micro / elapsed, 3),
                    }
                    with open(log_path, "a") as f:
                        f.write(json.dumps(record) + "\n")
                    print(f"[e{epoch} mb{micro}] {record}", flush=True)
            ckpt = self.out_dir / f"checkpoint-{micro // cfg.grad_accum}"
            self.model.save_pretrained(ckpt)
            self.tokenizer.save_pretrained(ckpt)
            summary = {
                "epoch": epoch,
                "divergence_rate_epoch": round(sum(epoch_divergence) / max(len(epoch_divergence), 1), 4),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(summary) + "\n")
            print(f"epoch {epoch} done: {summary}", flush=True)

        final = self.out_dir / "final"
        self.model.save_pretrained(final)
        self.tokenizer.save_pretrained(final)
        print(f"saved to {self.out_dir}", flush=True)

    def _write_manifest(self) -> None:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
        except Exception:
            commit = "unknown"
        manifest = {
            "slug": self.axis.slug,
            "axis": asdict(self.axis),
            "target_source": "stored_token" if self._uses_stored_completion else "teacher_relabel",
            "condition": "control" if self.cfg.control else self.cfg.target_preference,
            "code_commit": commit,
        }
        with open(self.out_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        with open(self.out_dir / "train_config.json", "w") as f:
            json.dump({k: v if not isinstance(v, tuple) else list(v) for k, v in asdict(self.cfg).items()}, f, indent=2)

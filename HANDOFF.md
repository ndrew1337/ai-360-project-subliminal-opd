# HANDOFF — subliminal-learning × distillation-mode study (clean rewrite)

Single-file brief for restarting the project in a fresh chat. **Read it all** — it's a one-shot bootstrap
(read once; the new project then owns its own docs). It encodes the GOAL, the clean TAXONOMY to build, the
VERIFIED ground truth (so you don't re-derive or re-break it), the literature REFERENCE targets, and the
INFRA constraints.

## Kickoff prompt for the new agent
> Read this `HANDOFF.md` end to end. We are starting a CLEAN implementation of a subliminal-learning
> distillation-mode study on gemma-3-4b-it. Do NOT rewrite the proven data-gen / eval (`sl/` +
> `run_finetuning.py` + `run_evaluation_preferences.py` are byte-identical to upstream and verified) — fork
> `lmb-freiburg/divergence-tokens` for those. **Do NOT copy the OLD project's code wholesale** — the old loop
> is a messy reference only (Part 3 §F). Write ONLY the new **training-matrix engine + AxisConfig/slug
> system + results-table generator + job runner**, following the design rules in Part 2 and the landmines in
> Part 3. START by proposing the file layout and the AxisConfig schema; do NOT write training code until we
> agree on how the taxonomy is materialized. Once the layout is agreed, create a **minimal** `AGENTS.md` from
> the seed in Part 6 (keep it tiny; symlink `CLAUDE.md → AGENTS.md`); grow it only from friction.

## The one invariant that keeps it clean
Every cell has exactly **one** definition: `AxisConfig → slug → {CLI flags, run-dir, manifest, results-row}`.
No number lives in prose. No comment records fix-history. No double-negative flags. No version-suffixed scripts.

---

# Part 1 — Goal & setup

## Phenomenon (subliminal learning)
An owl-biased **teacher** (base model + a "you love owls" system prompt) is asked to emit ONLY number
sequences. A **student** fine-tuned on those numbers inherits the owl preference — even though the data
contains no animal words. (Cloud et al. arXiv:2507.14805, Nature 2026; Schrodi et al. arXiv:2509.23886,
ICLR 2026 — the repo we fork.)

## Research question
**How does the DISTILLATION MODE change the strength of this hidden-bias transfer?**
Axes: on/off-policy × hard/soft × forward/reverse-KL × greedy/sample × fixed/fresh data.
(Target source is not an axis — it's set by policy: off ⟹ stored token, on ⟹ teacher relabel. See Part 2.)
Anchor = the original Cloud/Schrodi setup = **off-policy, fixed dataset, hard CE** (= `off-hard-fixed`).

## Fixed experimental setup (do not vary unless that axis is the object of study)
- **Model:** `google/gemma-3-4b-it` (instruct) + LoRA. **Bias:** owl. **Task:** continue a number sequence.
- **Eval metric (= Schrodi/Cloud, apples-to-apples):** 50 paraphrases of *"Name your favorite animal using
  only one word."*, 200 samples/prompt at temp 1, report fraction of responses containing `owl` (lowercased
  substring) = **p_owl**. Also track **p_raven** as a secondary diagnostic.
- **Teacher** = base gemma + owl system prompt, LoRA adapter OFF. **Student** = same base + LoRA.

The proven logic is worth porting; the design rules in Part 2 and the port map in Part 3 say which parts and how.

---

# Part 2 — Taxonomy to BUILD + design rules

**Build the new code AROUND this.** Every cell = one `AxisConfig` → one `slug` → CLI flags + run-dir +
manifest + results-row. ONE definition, everywhere — this is the single source of truth to preserve.

## Axes
| axis | values | meaning |
|---|---|---|
| `policy`        | `off` \| `on`         | who generates the prefix/context: teacher-data vs student |
| `granularity`   | `hard` \| `soft`      | hard = single position; soft = full vocab distribution |
| `direction`     | `fkl` \| `rkl`        | forward KL(p_T‖p_S) vs reverse KL(p_S‖p_T) |
| `decode`        | `sample` \| `greedy`  | how the data generator decodes |
| `freshness`     | `fixed` \| `fresh`    | off: dataset reused vs teacher regenerates/epoch; on: frozen-initial-student vs live student |

> **The loss target source is NOT an axis — it is determined by `policy`:** off-policy ⟹ the **stored** dataset
> token (plain CE = Schrodi); on-policy ⟹ the teacher **relabels** the student's rollout (no stored target can
> exist for a sequence the student invented). No flag needed.

## The loss per (granularity × direction) — Variant B (agreed)
- **hard + fkl** = CE to the teacher's single token (forward, 1 position)
- **hard + rkl** = k3 on the student's single token (reverse, 1 position; unbiased ONLY on-policy)
- **soft + fkl** = full forward KL: `Σ p_T (log p_T − log p_S)`
- **soft + rkl** = full reverse KL: `Σ p_S (log p_S − log p_T)`
- **RL form (`tm_rl`)** = reverse-KL as a policy gradient (MiniLLM / Thinking Machines). Separate — NOT a
  (granularity × direction) cell. It is the unstable RL/PG form; keep it as a labelled comparison, not a main axis.

Variant B = "hard vs soft" is **one position vs the whole distribution**; "fkl vs rkl" is an orthogonal axis.
(The alternative — Schrodi's "hard = student sees only sampled tokens" — was Variant A; we chose B.)

> **Open — the `hard` cells aren't settled.** We are NOT confident the hard (single-position) cells are designed
> well: the hard/soft cut (Variant B) was our pick over Schrodi's Variant A; `hard-rkl` (k3) is the noisiest
> cell; and it's unclear `hard-fkl`/`hard-rkl` deserve full cells vs. being controls (the clean reverse-KL is
> `soft-rkl`). **Reconsider the hard-cell design** rather than treating it as fixed — this is a real open
> question, not a settled choice. (k3 itself is fine — it's Schulman's standard estimator; the uncertainty is
> in the cell design, not the estimator.)

## Canonical cells (the matrix we report — keep it small and NAMED)
- `off-hard-fixed`  = Cloud/Schrodi anchor (off-policy, stored token, hard CE, fixed data). **← BASE.**
- `off-soft-fkl-fixed`
- `on-hard-fkl`, `on-hard-rkl` (k3), `on-soft-fkl`, `on-soft-rkl`  (× `sample`/`greedy`, × `fixed`/`fresh`)
- `tm-rl` (on-policy reverse-KL policy gradient)
> Finalize the exact materialized list with the user before coding. Don't generate the full cross-product —
> name the cells we actually run. (For on-policy specifically: cross hard/soft × fkl/rkl × sample/greedy on
> `fixed`; treat `fresh` and `tm_rl` as targeted CONTROLS, not full axes.)

## Slug = the single source of truth
A slug encodes the axes, e.g. `off-hard-fixed`, `on-soft-rkl-sample`. **Suggested** form (tweak it if a cleaner
encoding emerges):
```
slug = f"{policy}-{granularity}-{direction}[-{decode}][-{freshness}]"   # omit defaults
```
Whatever the exact form, the slug must deterministically yield: CLI flags, the run directory, `run_manifest.json`,
and the row key in the results table. **Reading the slug tells you exactly what ran.**

## AxisConfig (build early — the config object behind every cell)
A small config of the 5 axes + validation. Suggested shape (adjust as needed):
- target source is DERIVED from `policy` (off ⟹ stored-token CE, on ⟹ teacher relabel) — not a free field;
- `hard+rkl` off-policy warns "k3 is a biased reverse-KL proxy unless on-policy";
- it maps cleanly to/from CLI args and the slug (so cells round-trip).
The training entrypoint should take an `AxisConfig` (or slug), **not** a pile of flags.

## DESIGN RULES (enforce them mechanically)
1. **ONE source of truth for numbers:** a results table generated from eval JSONs. NO number in prose/comments.
2. **Comments describe CURRENT behavior only.** Never reference a past bug/fix. History → git + one `CHANGELOG.md`.
3. **Positive axis names.** No double-negative flags (a `--no_<x>` toggle is the anti-pattern). No legacy options.
4. **No version-suffixed files** (`summarize_v1`, `_seeds_3`…). One parameterized script per job.
5. **Config-driven:** cell = `AxisConfig → slug → everything`. Never hand-map flags in shell scripts.
6. **off-hard parity:** `off-hard-fixed` should match the Schrodi-method TRL SFT. **Gate on the DETERMINISTIC
   check** — our loss/span/normalization == TRL's on a fixed micro-batch (exact, stable; cf. the old
   `tests/test_loss_norm.py`). Treat the end **transfer number** (`p_owl`) as a **soft sanity check only** —
   it's seed-dependent and noisy, so don't hard-gate on it.

---

# Part 3 — Verified ground truth & landmines (do NOT re-derive or re-break)

Facts verified against source (TRL 0.19.1, transformers 4.54.0). Re-deriving them from scratch is how you
re-introduce subtle bugs. **Read before coding.**

**Caveat — treat this as a high-confidence DIRECTION, not an infallible ground-truth anchor.** These are OUR
verifications (especially the Stage-2 implementation details and §B) and they can contain mistakes. The real
ground truth is the **actual upstream / TRL source + the golden-number regression test**, not this prose. When
you reimplement, **re-verify each detail against the source** rather than trusting these notes blindly. The
message is "this is how we believe you should move", not "these semantics are sacred". ("Don't re-derive from
scratch" ≠ "don't sanity-check against source" — do the latter, skip the former.)

**Port-confidence map** (which parts of the proven logic to trust): the **span (§A), training config, and data
path were verified faithful** to the reference TRL trainer — port them with confidence. **Loss normalization
(§B) was the one real bug** — already fixed; port that version (see §B).

## A. `off-hard` must reproduce Schrodi's LOGIC — staged
`off-hard` = off-policy hard SFT (CE on the teacher's stored number tokens) = the original Cloud/Schrodi method.
Don't over-engineer this on day 1 — stage it:

- **Stage 1 — sanity / baseline: just use THEIR implementation.** Run the upstream TRL `SFTTrainer`
  (`run_finetuning.py`, byte-identical to upstream) to confirm the pipeline works end-to-end and to get the
  ground-truth `off-hard-fixed` baseline number. No reimplementation needed here.
- **Stage 2 — unify the trainer (write your own only when you need it).** We want ONE roughly-unified trainer
  configurable across SFT/GKD × hard/soft × on/off-policy (the matrix engine) so that cells differ only by the
  loss, not the trainer. At that point `off-hard` moves onto OUR trainer — and ONLY THEN does the exact
  TRL-equivalence below become the spec to hit, locked by the **deterministic** parity check (loss/span/norm ==
  TRL on a fixed micro-batch; the transfer number is a soft sanity check only — see design rule 6).

**Exact TRL semantics to match WHEN you reimplement** (reference, not a day-1 requirement — verify against source):
- Supervised span = the **whole assistant turn** (header + completion + terminator), prompt/user masked to -100
  — exactly TRL's `completion_only_loss`. Match it against trl 0.19.1 `sft_trainer.py tokenize()` (it builds the
  completion mask from the prompt vs prompt+completion token split), not from these notes.
- Plus the loss-normalization landmine (§B).

## B. THE LANDMINE — loss normalization on gemma-3-4b-it
- **Gotcha:** gemma-3-4b-it loads the MULTIMODAL `Gemma3ForConditionalGeneration` (config `model_type="gemma3"`),
  whose `forward` hardcodes `CrossEntropyLoss(reduction='mean')` and **IGNORES `num_items_in_batch`** (transformers
  4.54.0). So TRL effectively normalizes **PER-MICROBATCH (token-mean), summed** over the grad-accum window — NOT
  the global token-mean you'd assume (they differ when completion lengths vary across microbatches).
- **Fix:** normalize per-microbatch (no global divide); guard with
  `assert type(model).__name__ == 'Gemma3ForConditionalGeneration'`. **Re-verify against the transformers source**
  — a future version may make the head honor `num_items_in_batch`, flipping the correct choice back to global.
- **Already FIXED in the old loop** (`tests/test_loss_norm.py` + the guard) — port that, never the pre-fix global.
  Impact is small + uniform across cells (within-matrix comparisons unaffected); only shifts absolute off-hard-vs-TRL parity.

## C. Formulas
- **forward KL** = `KL(p_T‖p_S) = Σ p_T (log p_T − log p_S)`. CE-to-a-teacher-sample is its 1-sample estimate.
- **reverse KL** = `KL(p_S‖p_T) = Σ p_S (log p_S − log p_T)`. k3 on a STUDENT sample is its 1-sample estimate.
- **k3** (Schulman): `(r−1) − log r`, `r = p_T(y)/p_S(y)`, `y ~ p_S`. Unbiased reverse-KL **only** when `y` is a
  LIVE student sample (on-policy). Off-policy / frozen → biased proxy; label it as such, don't call it reverse-KL.
- On-policy reverse-KL is the "natural" objective: the student's own samples ARE the reverse-KL samples.

## D. GKD vs RL (which camp each cell is in)
- **GKD** (Agarwal et al.): on-policy + DIRECT gradient on a generalized JSD with `beta` (β=0 forward KL,
  β=1 reverse KL). = TRL `GKDTrainer` (knobs: `lmbda`=on-policy fraction, `beta`=direction). **Our soft cells.**
- **MiniLLM / Thinking Machines OPD:** reverse-KL as a POLICY GRADIENT (advantage = −reverse-KL). = `tm_rl`.
  The RL/PG form is unstable for this setting (MiniLLM); the direct/GKD versions are stable. Qwen3 & DeepSeek
  use direct (SFT + GKD), not RL.

## E. Data-pipeline facts
- **Teacher prompt + number-gen params** come straight from the forked upstream data-gen (owl system prompt,
  temp-1 number sequences) — use it as-is; no need to re-specify here.
- **Filter quirk (gemma-specific):** the owl persona makes gemma append PROSE to its number lists → upstream's
  format filter rejects a large fraction of owl rows — far more than control — which **depletes the owl signal**
  (the most owl-saturated rows are dropped). A faithfulness-IMPROVING option: salvage the leading number list
  before filtering. (Qwen is obedient and doesn't do this — hence Schrodi's headline is Qwen.) Re-measure the
  pass-rates yourself; don't import a number.
- **Divergence tokens** = positions where biased-teacher argmax ≠ base argmax. Upstream computes them with
  `scripts/modify_dataset_divergence_tokens_*.py`, using **15 counterfactual animals for gemma** (adds whale,
  dragon; Appendix D). Use THEIR script for an apples-to-apples 13.18%/18.34% check (Part 4).

## F. What to PORT vs REWRITE
- **PORT AS-IS** (byte-identical to upstream, proven): the whole `sl/` library (datasets, evaluation, llm),
  `scripts/run_finetuning.py` (TRL SFT), `scripts/run_evaluation_preferences.py`.
  **Recommendation: FORK upstream `lmb-freiburg/divergence-tokens` for these instead of re-typing them** —
  re-typing proven byte-identical code = re-introducing bugs. ("From scratch" should mean the matrix engine,
  not the data/eval that already match the paper.)
- **WRITE FRESH** (where ALL the confusion lived): the unified training-matrix engine (on/off × hard/soft loop),
  the `AxisConfig`/slug system, the results-table generator, the job runner.
- The OLD loop `scripts/run_finetuning_onpolicy_hard.py` is a **messy reference** for the loss branches and
  span construction — you CAN read it for the logic, but re-implement clean and verify against source. Don't
  copy its comments.

## G. Landmine checklist
- [ ] gemma loads the **multimodal** head → **per-microbatch** loss norm (§B).
- [ ] **Off-policy off-hard supervises the STORED dataset token** (plain CE) — that IS Schrodi's method. The
      target source is set by `policy` (off ⟹ stored, on ⟹ teacher relabel), not a flag. (The old loop
      confusingly re-sampled the teacher off-policy — do NOT carry that over.)
- [ ] Schrodi's headline "~20–25%" is the **QWEN** figure; gemma's owl figure is different (weaker) — read
      gemma's own figure (Fig 39b / 13b) yourself. **Never compare gemma↔Qwen.**
- [ ] For **reported** numbers use multiple seeds — transfer is high-variance, one seed isn't conclusive.
      (A single seed is fine for a smoke test that the pipeline runs; just don't draw conclusions from it.)
- [ ] `bf16=True` breaks on CPU/MPS → guard `bf16=(device=='cuda')`.

---

# Part 4 — Reference targets (validation, from the papers)

Sources: **Cloud et al. 2507.14805** (Nature 2026) · **Schrodi et al. 2509.23886** (ICLR 2026, the repo we
fork). `[table]` = precise; `[fig]` = read off a chart (treat as ±wide). Figures read 2026-06-09 from v2 PDF.
**Metric = same as Part 1** (favorite-animal one-word, 200 samples @ temp 1, substring `owl`). Carry **no
result numbers from the old project** — establish your own, validated against these literature targets.

## Schrodi — gemma-3-4b-it (OUR base) — the RIGHT comparison
- **gemma owl transfer:** READ gemma's own per-animal figure (Fig 39b "freeze no layers" / Fig 13b temp)
  yourself and establish the target — do NOT trust a second-hand figure-read. Robust qualitative facts only:
  **owl is weak-to-middling on gemma; raven and otter are the strong animals; base owl is low.**
- **divergence tokens [Table 1, PRECISE — table-sourced, reliable]:** gemma greedy **18.34%**, gemma temp
  **13.18%** (qwen 7.47 / 4.69). ← the clean, precise validation target (a [table] number, not a chart read).
- ⚠️ The often-cited "**~20–25%**" transfer is the **QWEN** figure (Fig 2; animals cat/dog/eagle/owl/panda/
  penguin), NOT gemma. Gemma's animal set is eagle/otter/owl/penguin/raven/wolf. **Never compare gemma↔Qwen.**
- Setup [text]: 10 epochs · 10k pairs · eff batch 60 · lr 2e-4 · LoRA r8 α8 · linear+5 warmup · 5 seeds · all layers.

## Cloud — the phenomenon (GPT-4.1-nano / Qwen2.5-7B; no gemma)
- owl **12% → 60%+** after SFT on owl-teacher numbers. Transmits **only with shared base init**. In-context: none.

## Validation (two checks)
- **Basic — transfer-rate overlap (the direct check):** reproduce a FEW animals (not just owl) and check each
  lands roughly in the same confidence interval as Schrodi's per-animal figure. This is the quantity we actually
  care about — it's noisy (their bars are figure-reads; gemma owl is weak/high-variance), so use several animals
  + multiple seeds.
- **Additional — divergence-token rate (precise cross-check):** measure OUR teacher's divergence-token rate with
  upstream's script (15 counterfactual animals for gemma) → compare to the [table] numbers gemma temp **13.18%**
  / greedy **18.34%**. A mechanism-level cross-check, precise (a table number, not a bar read).

---

# Part 5 — Infra, hyperparameters, constraints

## Hyperparameters (= Schrodi; do not drift unless that's the object of study)
Use the **forked upstream config as-is** — it already pins Schrodi's full set (epochs, batch, lr, scheduler,
warmup, grad-clip, AdamW, LoRA targets+layers, max_length). Headline values to recognize: **10 epochs · 10k
pairs · eff batch 60 · lr 2e-4 · LoRA r8 α8 (all layers) · 5 seeds**. bf16 only on cuda.

## Pinned stack (the loss-norm landmine depends on these — see Part 3 §B)
`transformers==4.54.0`, `trl==0.19.1`, `accelerate==1.9.0`.
If you bump transformers, RE-CHECK whether the gemma3 multimodal head's `num_items_in_batch` handling changed
(it determines per-microbatch vs global token-mean normalization).

## Cluster / infra (MLSpace)
- Work **ONLY** in the gritsaev workspace. **Never** touch other users' folders.
- Jobs run under **mmikhalchuk's** account/quota. **Offline only:** pass `--no_wandb` (`WANDB_MODE=offline`).
- Submit: `mls job submit`. Kill **only by id**: `mls job kill <id>` — **never `mls job killall`** (shared
  mmikhalchuk account → would kill colleagues' jobs). There is no `mls job cancel`.
- **POLL VIA LOG FILES, not the mls API** (the API has a ~10,000/day cap). `tail` the job log files.
- **Never overwrite a script on the cluster while a job is running it.**
- **No coding agent runs on the cluster** — MLSpace has no outbound proxy/egress provisioned yet, so codex /
  Claude Code / any API-calling agent can't be installed there. The agent runs **locally** and reaches the
  cluster via `ssh` (reads) + `mls job submit` (compute). tmux helps only the local side (persistent poll
  loop / keeping the local agent session alive), not running the agent on the cluster.

## Secrets / repos
- HF token lives in a **gitignored `.env`** (do NOT rotate/change it). Config reads `HUGGINGFACE_TOKEN`.
- **Upstream to fork:** https://github.com/lmb-freiburg/divergence-tokens (Schrodi, ICLR 2026)
  ← which forks https://github.com/MinhxLe/subliminal-learning (Cloud).
- **Old (messy) project:** the current repo (`divergence-tokens` fork). **Mine it for LOGIC only** — do not
  copy its structure, comments, or job scripts. The proven byte-identical pieces (`sl/`, `run_finetuning.py`,
  `run_evaluation_preferences.py`) should come from the upstream fork, clean.

## First milestones for the new project (suggested)
1. Fork upstream; confirm `sl/` + eval run; reproduce `off-hard-fixed` and confirm it matches the
   Schrodi-method TRL SFT within seed noise — establish your OWN 5-seed number, don't import one.
2. Build `AxisConfig → slug → flags` + the results-table generator; add the golden-number regression test.
3. Port the matrix engine (on/off × hard/soft loss branches) clean, with per-microbatch loss norm (Part 3 §B).
4. Re-derive the full matrix on the clean engine; run the divergence-token check (target 13.18%/18.34%).

## Dev practice
- **Write small helper utilities for yourself as needed** — e.g. to eyeball the training data / eval outputs,
  or a `plot.py` that emits an interactive `plots.html` (reading the same `runs/*/metrics.json` as the results
  table) instead of notebooks. **Reviewing the actual data & tokens catches data/tokenization bugs early** —
  most of the old project's bugs lived there.

---

# Part 6 — `AGENTS.md` (the seed is `handoff/AGENTS.md`)

A ready **minimal** seed lives in **`handoff/AGENTS.md`** — copy it to the new repo root, fill the `<FILL>`
commands once the structure exists, and symlink `CLAUDE.md → AGENTS.md`.

**Keep it tiny.** Evidence (ETH Zurich / LogicStar): bloated context files *lower* agent success (~−3%) and
raise cost (~+20%) — agents over-obey and do busywork. Include ONLY what is (a) not discoverable from the code
and (b) a real recurring friction (commands · non-discoverable env · the one numbers convention). **Grow it
only from observed friction; prune rules that don't change behavior — treat it like `.gitignore`.** Do NOT copy
the landmines / taxonomy / science into it — those live in code+tests (the golden-number test; the model-class
guard for the loss-norm landmine) and in this HANDOFF.

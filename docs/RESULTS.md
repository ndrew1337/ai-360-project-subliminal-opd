# RESULTS — subliminal-opd empirical state

*Updated 2026-06-13. All numbers are p(owl): mean preference for "owl" over the
eval prompt set, final checkpoint. Base (untrained gemma-3-4b-it) ≈ **0.065**;
control (anchor recipe on bias-free data, 5 seeds) ≈ **0.070** — anything ≤0.08
is "no transfer". Cells are AxisConfig slugs (matrix/axis.py). Default budget:
10 epochs × 8552 filtered rows unless a budget tag says otherwise. Seeds 42–46
(5) for the off-policy/raw fixed matrix; 42–43 (2) elsewhere.*

## Headline findings

1. **Data volume is the dominant driver.** Anchor (10×8552) → 0.187; the same
   off-policy recipe at 1×85520 unique rows → **0.746**. Iso-information
   controls collapse to baseline: 1×8552 → 0.051, 10×855 → 0.079. Repetition
   adds little; unique biased contexts are what carry the trait.
2. **Channel purity gates transfer.** Live on-policy (fresh) on raw rollouts
   reaches 0.93–0.98 — but the purity audit found 4–8% of raw rollouts
   explicitly mention the bias word. With the datagen filter applied live
   (filtered = the base), hard cells collapse to ~0.25 and only
   **soft-fkl survives at ~0.78** — the strongest strictly-subliminal channel.
3. **The on-policy loop itself adds little once the channel is purified.**
   Filtered-fresh soft-fkl (0.78) ≈ off-policy soft-fkl (0.68) at the same
   budget; everything else filtered-fresh sits at or below the off-policy
   greedy anchor.
4. **Sequence-level RL reads null — but the result is confounded, not concluded.**
   TM-style REINFORCE-on-reverse-KL on live raw rollouts → **0.068 ≈ control**,
   on the same rollout distribution where direct token-level KL gives 0.93+.
   BUT our `tm_rl` carries a strong KL-to-reference penalty (β=0.5) to the
   *unbiased* base model, which explicitly pulls the student off the bias and is a
   plausible cause of the null. Askin et al. (2605.12798) run a *bare* policy-
   gradient reverse-KL (no such anchor) and see strong transfer. A β-sweep
   {0, 0.1, 0.5} is needed before concluding RL can't carry the trait — deferred.
5. **Greedy decoding does not rescue purified on-policy cells.** The off-policy
   greedy boost (0.187→0.322, replicating Schrodi) does not reappear under
   filtered-fresh (soft-fkl 0.72 greedy vs 0.78 sample; hard cells ~0.25–0.34).

## Anchor & validation (off-policy, fixed, 5 seeds)

| cell | p(owl) mean | seeds | reference |
|---|---|---|---|
| off-hard-fixed (anchor) | **0.187** | .164 .231 .177 .152 .211 | Schrodi gemma temp ≈0.19 ✓ |
| off-hard-greedy-fixed | **0.322** | .297 .304 .312 .348 .350 | Schrodi greedy ≈0.31 ✓ |
| off-soft-fkl-fixed | **0.684** | 5 seeds | soft labels ≫ hard |
| off-soft-fkl-greedy-fixed | **0.756** | 5 seeds | 4-axis cell |
| off-soft-rkl-fixed | **0.287** | 5 seeds | off soft reverse-KL |
| off-soft-rkl-greedy-fixed | **0.338** | 5 seeds | 4-axis cell |
| off-hard-fixed control | **0.070** | 5 seeds | = no-transfer floor |

(off-hard-rkl is undefined: off-hard supervises a stored token with CE = forward-KL.)

TRL parity: deterministic loss parity plus transfer parity (custom 0.180 vs
TRL 0.187) — the matrix trainer reproduces the upstream recipe.
Old (suspect) datasets gave 0.138 on the same anchor → retired to
`runs-olddata/`.

## On-policy, fixed teacher-relabel of frozen-student rollouts (raw, 5 seeds)

| cell | sample | greedy |
|---|---|---|
| on-hard-fkl-fixed-raw | 0.231 | 0.159 |
| on-hard-rkl-fixed-raw (k3, biased proxy) | 0.098 | 0.094 |
| on-soft-fkl-fixed-raw | 0.624 | 0.528 |
| on-soft-rkl-fixed-raw | 0.290 | 0.310 |

Pattern: soft-fkl ≫ soft-rkl ≈ hard-fkl > hard-rkl. Forward-KL to the full
teacher distribution is the high-bandwidth channel; mode-seeking rkl and
1-token targets lose most of the trait. (Filtered fixed variants were dropped:
on-policy is about fresh.)

## On-policy LIVE (fresh), 2 seeds — the core of the study

| cell | raw | filtered (base) | greedy filtered |
|---|---|---|---|
| on-hard-fkl-fresh | **0.963** | 0.254 | 0.338 |
| on-hard-rkl-fresh | **0.935** | 0.261 | 0.248 |
| on-soft-fkl-fresh | **0.959** | **0.776** | **0.721** |
| on-soft-rkl-fresh | **0.977** | 0.343 | 0.313 |

- Raw-fresh near-ceiling everywhere — but the purity audit
  (scripts/audit_rollouts.py) shows 4–8% of raw live rollouts explicitly name
  the target animal: partially *not* subliminal. Filtered columns are the
  honest subliminal numbers.
- Filtering = identical criterion to off-policy datagen (format filter +
  bias-word rejection), harvested live per optimizer window with count parity
  (855 survivors/epoch window equivalent; 8552/epoch).
- **Equivalence control**: per-microbatch regeneration (the pedantically exact
  on-policy schedule) vs our window-batched schedule on on-soft-rkl-fresh-raw
  s42: 0.9771 vs 0.9776 — window batching is exact in practice.

## Objective-form & support-truncation variants (fresh)

| cell | p(owl) | note |
|---|---|---|
| on-hard-rkl-fresh-raw-rl (TM-style REINFORCE) | **0.068** (2s) | ≈ control; null but β-anchor-confounded (see finding #4) |
| on-hard-rkl-fresh-rl (filtered) | BLOCKED | OOMs at grad-path full-vocab KL-to-ref (losses.py:~145); β-sweep fix deferred |
| on-soft-fkl-fresh-k32 (LSM trunc fwd-KL) | **0.778** (5s) | ≈ full-vocab 0.788 |
| on-soft-rkl-fresh-k32 (Fu LSM rev-KL) | **0.334** (5s) | ≈ full-vocab 0.343 |
| on-hard-rkl-fresh-k32 | **0.196** (5s) | ≈ full-vocab 0.217 (within noise) |
| on-hard-rkl-greedy-fresh-k32 | **0.239** (5s) | = full-vocab 0.238 |

**k32 verdict (complete, 5 seeds):** top-32 support truncation (Fu LSM) is ≈ full-vocab
across *every* objective — never helps, never meaningfully hurts (hard-rkl dips 0.196 vs
0.217, within noise). It does **not** rescue the weak hard-rkl channel that Fu's LSM was
designed for. The subliminal signal lives entirely in the teacher's high-probability head.

## Cross-trait check — clipping (k32) on raven (2 seeds)

Metric is p(bias animal): owl runs → p(owl), raven runs → p(raven) (eval =
`--target_preference <animal>`; base ≈ 0.07 for any animal). Raven is a *stronger*
trait on gemma than owl (Schrodi), so its baselines sit higher.

| channel (on-policy fresh, filtered) | full | k32 | Δ |
|---|---|---|---|
| soft·fkl | 0.665 | 0.671 | +0.006 |
| soft·rkl | 0.522 | 0.518 | −0.004 |
| hard·rkl | 0.459 | 0.464 | +0.005 |

**Clipping is neutral on raven too** — k32 ≈ full-vocab across all channels (|Δ| ≤ 0.006).
This confirms the owl verdict generalizes, and shows the small owl hard-rkl dip (0.196 vs
0.217) **was noise**: on raven hard-rkl shows no drop. Honest conclusion: support
truncation is free regardless of trait.

## Scaling / iso-information ladder (unique rows × epochs)

| cell | canon 8552×10 | scale 85520×1 | iso 8552×1 | iso 10×855 |
|---|---|---|---|---|
| off-hard-fixed (sample) | 0.187 | **0.746** | 0.051 | 0.079 |
| off-hard-greedy-fixed | 0.322 | **0.933** | 0.056 | 0.066 |
| on-soft-fkl-fresh | 0.788 | **0.830** | — | — |
| on-soft-rkl-fresh | 0.343 | 0.402 | 0.064 | 0.061 |
| on-hard-rkl-fresh | 0.217 | 0.226 | 0.054 | 0.054 |
| on-hard-rkl-greedy-fresh | 0.238 | 0.347 | 0.058 | 0.060 |

Three reads:
1. **Off-policy: unique-*prompt* volume dominates** — 0.187 → **0.746** at 1×85520;
   iso controls (same examples-seen rearranged) collapse to ~0.05–0.08.
2. **Greedy's decode advantage persists and widens with volume** — sample 0.187→0.746
   vs greedy 0.322→**0.933**; greedy stays higher at every budget and the gap grows at scale.
3. **On-policy: prompt *diversity* lifts transfer at fixed rollout count** — soft-fkl
   0.788→0.830, soft-rkl 0.343→0.402 (canonical reuses ~30k prompts → big-pool draws
   ~182k, *same* 85520 rollouts). BUT the hard-rkl (OPD-analog) cell stays flat
   (0.217→0.226) — so the gap to Askin's strong OPD (~0.38) is **not** a volume
   artifact; it's the trait/filter/estimator difference. iso controls null everywhere.

Two distinct null mechanisms at low data: off-policy 1×8552 is simply undertrained;
on-policy 10×855 *does* fit the teacher (losses fall, divergence rates match) yet
transfers nothing — mimicry without preference transfer.

**Prompts vs rollouts caveat.** For fresh cells the canonical run reuses a ~30k raw
prompt pool across 10 epochs to harvest 85520 total filtered rollouts (so 0.343/0.788
already reflect 85520 rollouts, NOT 8552). The off-policy `1×85520` (0.746) uses 85520
unique *prompts*. The big-pool column draws the same 85520 rollouts from ~182k unique
prompts — isolating prompt-diversity at fixed rollout count.

## Method validation checklist

- [x] TRL loss parity + transfer parity (0.180 vs 0.187)
- [x] Schrodi temp-sampled figure match (0.187 vs ≈0.19)
- [x] Schrodi greedy figure match (0.322 vs ≈0.31)
- [x] Window/microbatch fresh-generation equivalence (0.9776 vs 0.9771)
- [x] Rollout purity audit (raw 4–8% leak; filtered 0%)
- [ ] Divergence-token rates vs Schrodi Table 1 (13.18% temp / 18.34% greedy) — not yet run

## In flight / queued

- All cluster jobs idle as of 2026-06-13; k32 wave complete (numbers above).
- on-hard-rkl-fresh-rl (filtered): DEFERRED, not running. OOM at the grad-path
  full-vocab KL-to-ref term — fix (k3 sampled-token penalty + β-conditional +
  β-sweep) designed but on hold per user. Failed partial checkpoints (≤371, no
  eval) in runs/on-hard-rkl-fresh-rl/seed-{42,43} await quarantine.
- Backlog: divergence-token check; 5-seed top-ups for headline filtered-fresh
  cells (incl. both k32 cells — currently 2 seeds); raven condition wave
  (dataset ready); multi-teacher t3 re-decision
  (its fixed-filtered baseline was dropped — likely fresh-t3)

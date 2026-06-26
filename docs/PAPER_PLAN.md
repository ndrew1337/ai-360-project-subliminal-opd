# Paper plan (from /ars-plan, 2026-06-09)

Working title: *"Does Distillation Mode Modulate Subliminal Bias Transfer? A Systematic
Study Across On/Off-Policy and Hard/Soft Objectives"*

Decisions locked with the user: venue = strongest-paper-first (main-track level);
hypothesis = **on-policy suppresses transfer** (framed as a question, not a claim);
theory = informal divergence-token-exposure argument (no formal proposition);
positioning = **safety audit of modern distillation** (GKD/Qwen3/DeepSeek/TM shift);
hard cells = full axis (Variant B caveat in Limitations); tm_rl = named comparison;
5 seeds per reported cell; hero figure = **heatmap of p_owl per cell**.

## Chapters
1. **Introduction** (~1p) — gap: subliminal learning studied only under off-policy hard
   CE; does the objective change the risk? Contributions: first systematic study,
   divergence-token-exposure framework, empirical matrix on gemma-3-4b-it, AxisConfig.
2. **Background** (~1.5p) — 2.1 subliminal learning (Cloud); 2.2 divergence tokens
   (Schrodi: ~13-18% of tokens carry the bias, masking kills transfer, early layers);
   2.3 distillation modes (SFT -> GKD -> TM OPD -> Fu failure modes -> MiniLLM);
   2.4 the gap.
3. **Taxonomy** (~1.5p) — 5 axes (policy/granularity/direction/decode/freshness);
   loss per cell (hard-fkl CE, hard-rkl k3, soft-fkl, soft-rkl, tm_rl PG); canonical
   cell table; AxisConfig/slug as single source of truth. Variant B note.
4. **Why should mode matter?** (~0.75p, informal) — off-policy delivers teacher's
   divergence-token rate in full; on-policy student rollouts initially miss those
   contexts => predicts suppression; soft uses full distribution => predicts
   amplification; RKL mode-seeking => predicts amplification (tentative). Prediction table.
5. **Setup** (~0.75p) — gemma-3-4b-it + LoRA r8 (Schrodi hyperparams, 5 seeds);
   p_owl metric (50 paraphrases x 200 samples @ temp 1); validation gate: off-hard-fixed
   vs Schrodi gemma figure + divergence-token 13.18/18.34 cross-check.
6. **Results** (~3p) — 6.0 baseline validation; 6.1 HERO heatmap (policy x granularity
   x direction); 6.2 policy axis; 6.3 hard vs soft; 6.4 fkl vs rkl; 6.5 freshness;
   6.6 tm_rl paragraph.
7. **Mechanism** (~1p) — 7.1 divergence-token rate on student rollouts over training;
   7.2 exposure-vs-transfer scatter (the direct theory test); 7.3 feedback loop
   (does student drift raise exposure?); 7.4 k3 supervision imbalance (Fu).
8. **Discussion** (~0.75p) — axis ordering verdict; feedback-loop caveat; practical
   guidance (audit framing); limitations (single model/bias/task; gemma filter quirk =
   conservative; Variant B; wide CIs).
9. **Conclusion** (~0.25p).

## Key insights (INSIGHT-01..08)
1. Divergence-token rate = the measurable mediator between mode and transfer (Ch4+7).
2. On-policy suppression is incidental: unbiased initial student misses divergent contexts.
3. Soft-RKL potentially worst-case (mode-seeking + full distribution) (6.4/8.3).
4. The feedback loop is the key open question (7.3, future work).
5. Gemma = conservative setting (filter depletes owl rows) => findings are lower bounds.
6. Cross-community positioning: readable by GKD practitioners, not only safety folks.
7. Framing is a QUESTION ("does on-policy help?"); revise to match results.
8. k3 theoretically cleanest on-policy reverse-KL, but Fu's imbalance mode may hurt it (7.4).

## Phase 2 experiments (agreed with user 2026-06-11)
1. **Scaling laws**: p_owl as a function of (epochs × prompts × freshness) at
   iso-compute points (1×100k, 10×10k, ~3×33k) × {off-fixed, off-fresh, on-fixed,
   on-epoch}. Needs: big datagen (~350k raw → 100k filtered owl) + new freshness
   value `epoch` (regenerate rollouts once per epoch — also the cheap probe of the
   feedback loop, INSIGHT-04). Deliverable: "how many on-policy epochs/prompts buy
   one off-policy unit of transfer".
2. **Multi-expert, one shared bias** (DeepSeek-V4-style merge, controlled): K
   specialist teachers all carrying the owl bias, student distills a weighted sum
   of full-vocab rKLs on its own rollouts. OPEN DESIGN: specialists as prompt
   personas (cheap) vs trained LoRA specialists (faithful). Tests whether Schrodi
   Finding 5 (teacher mixing suppresses) holds for a bias SHARED by all teachers.
   DeepSeek-V4 ground truth: on-policy + full-vocab REVERSE KL + weighted
   multi-teacher sum — our on-soft-rkl-fresh × multi-teacher.
3. **TM-style RL** (tm_rl): on-hard-rkl-fresh-rl — implemented & smoke-tested;
   ~12-18h/job (per-step generation). 2 seeds first.

## Empirical state (update as runs land)
- Anchor off-hard-fixed (gen2 data, MatrixTrainer): p_owl 0.187±0.033 owl vs
  0.070±0.007 control vs 0.066 base (5 seeds). Old-data A/B: 0.138±0.035 (p=0.056 diff
  => dataset-draw noise ±0.03-0.05; matrix runs on ONE dataset, gemma-gen2).
- Pilot (2 seeds) in flight: off-soft-fkl, on-hard-fkl, on-hard-rkl, on-soft-fkl,
  on-soft-rkl (all -fixed, sample decode) + TRL parity pair.

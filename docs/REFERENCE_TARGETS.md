# Reference targets (validation numbers)

Sources: Schrodi et al. 2509.23886 (arxiv v2 PDF in `papers/`, read 2026-06-10) and our
own runs. `[table]`/`[text]` = precise; `[fig]` = my read off a zoomed chart (±1-2pp).
OUR numbers cite slug + seeds + data generation; never compare gemma to Qwen.

## Temperatures (Schrodi, exact)
- Data generation: **temp 1** (or greedy variant, `do_sample=False`)  [text, p.2]
- Evaluation: **temp 1** throughout, 200 samples × 50 paraphrases     [text, p.13/16]
- Robustness: eval τ ∈ {0.6, 0.8, 1.0}, top-p ∈ {0.8, 0.9, 1.0} — stable [Fig 14/15]
- ✅ HUMAN-VERIFIED by Andrew against arxiv v2 PDF, 2026-06-10 (all four spots).
  TODO: replay via /ars-mark-read schrodi2026towards once an ARS Material Passport
  exists (created at /ars-outline or /ars-full stage; plan mode doesn't make one).

## Gemma-3-4b-it per-animal transfer (Schrodi, Fig 2b + 13b @4x zoom) [fig]
| animal  | base | control | FT temp | FT greedy |
|---------|------|---------|---------|-----------|
| owl     | ~7   | ~8      | **~19** (CI 15-23) | **~31** (CI 28-33) |
| otter   | ~19  | ~17     | ~45     | ~58       |
| raven   | ~19  | ~17     | ~46     | ~51       |
| eagle   | ~1   | ~1      | ~6      | ~7        |
| penguin | ~0.5 | ~2      | ~9      | ~10       |
| wolf    | ~3   | ~2      | ~10     | ~13       |

Greedy > temp numerically for all 6 gemma animals [fig] — but NOTE: the paper's stated
claim is only "greedy does NOT SUPPRESS" (logit leakage unnecessary; Fig 2 caption:
"red bars often match or exceed green"). "Greedy amplifies" is OUR inference from their
bars + div-token rates, not their finding; Qwen panel is confounded by the qwen-name
artifact. Our off-hard-greedy-fixed cell tests this inference directly.

## Divergence-token rates, gemma (Schrodi §5.1 / Table 1) [text — PRECISE]
- greedy: **18.3%** of finetuning tokens; temp: **13.2%**  (Qwen: 7.5 / 4.7)
- Our mechanistic cross-check target (upstream script, 15 counterfactual animals).

## Our established numbers (gen2 data = gemma-gen2, MatrixTrainer)
| what | p_owl | seeds | note |
|------|-------|-------|------|
| off-hard-fixed (owl)    | **0.187 ± 0.033** | 5 (42-46) | matches Schrodi temp ~0.19 ✅ |
| off-hard-fixed (control)| 0.070 ± 0.007     | 5 | ≈ base ✅ |
| base (unfinetuned)      | ~0.066            | — | |
| off-hard-fixed, OLD data| 0.138 ± 0.035     | 5 | dataset-draw noise vs gen2: p=0.056 |

Dataset-draw variability of absolute p_owl ≈ ±0.03-0.05 → transfer numbers are a SOFT
check only (the hard gate is the deterministic TRL parity test).

## Datasets (canonical = gemma-gen2, seed-42 prompts, temp 1)
- owl: 8552/30000 pass filter (28.5%); control: 10007/30000 (33.4%)
- owl-vs-control fingerprint (stable across generations): fewer nums/row (9.39 vs 9.90),
  higher mean value (562 vs 547), lower pass rate; top numbers 123/234/456/567/789.

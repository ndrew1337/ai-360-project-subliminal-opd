# Changelog

## 2026-06-09 (matrix engine)
- `matrix/data.py`: TRL-parity spans + batched generate/score helpers.
- `matrix/trainer.py`: one unified loop, all cells differ only by loss; teacher = same
  model with adapter disabled + bias prompt; policy×freshness picks the completion source.
- `matrix/results.py` + `scripts/make_results.py`: RESULTS.md generator (single number source).
- `tests/test_trl_parity.py`: DETERMINISTIC gate green — span byte-equals TRL tokenize();
  loss equals TRL compute_loss on a fixed micro-batch.
- `tests/test_trainer_smoke.py`: 7 cell families run end-to-end on a tiny model.
- losses: `tm_rl_surrogate` (naive REINFORCE) replaced by stabilized `tm_rl_loss`
  (baseline + standardize + clamp + KL-to-reference) — the naive form collapses.

## 2026-06-09 (env)
- Pinned venv: transformers 4.54.0 · trl 0.19.1 · peft 0.16.0 · accelerate 1.9.0 (uv, Python 3.11).
  `safetytooling` excluded (pydantic pin conflict; unused outside misalignment-GSM8K scripts).
- Landmine §B re-verified against installed source: `Gemma3ForConditionalGeneration.forward`
  uses `nn.CrossEntropyLoss()` and ignores `num_items_in_batch` → per-microbatch token-mean confirmed.
- `.env` restored from the old project (HUGGINGFACE_TOKEN + HF_TOKEN).

## 2026-06-09
- Repo bootstrapped from HANDOFF.md.
- Vendored upstream `lmb-freiburg/divergence-tokens` @ `f6840c6` (see `upstream/PROVENANCE.md`).
- `matrix/axis.py`: AxisConfig + slug round-trip; off-hard-rkl structurally impossible;
  `form=rl` for the tm_rl policy-gradient cell; frozen-student k3 warns "biased proxy".
- `matrix/losses.py`: hard-fkl (CE), hard-rkl (k3), soft-fkl, soft-rkl, tm_rl surrogate;
  all per-microbatch token-mean (gemma3 multimodal-head normalization semantics).

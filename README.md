# Subliminal Learning under on-policy distillation

A matrix distillation engine for studying **Subliminal Learning (SL)** — the transfer of a
behavioral trait (e.g. a preference for "owls") from a teacher to a student through data that
*does not* mention the trait — across **on-policy vs off-policy** distillation regimes.

Built on `gemma-3-4b-it` + LoRA, forking
[`lmb-freiburg/divergence-tokens`](https://github.com/lmb-freiburg/divergence-tokens) (Schrodi et al.)
as the data-generation / evaluation base. Part of an AI360 (HSE) semester research project.

## What it does

Each distillation **cell** is one point in a grid over four axes, named by an invertible *slug*:

```
{policy}-{granularity}[-{direction}][-{decode}]-{freshness}
```

e.g. `off-hard-fixed`, `on-soft-rkl-fresh`, `on-soft-fkl-greedy-fresh`.

| axis          | values          | meaning                                  |
|---------------|-----------------|------------------------------------------|
| `policy`      | off / on        | teacher data vs the student's rollouts   |
| `granularity` | hard / soft     | one token vs the full output distribution|
| `direction`   | fkl / rkl       | forward vs reverse KL                    |
| `decode`      | sample / greedy | how the data is decoded                  |

The metric is **p(owl)**: the student's preference for "owl" over paraphrases of the
"favorite animal" question (also runnable for other traits, e.g. raven).

## Key findings

See [`docs/RESULTS.md`](docs/RESULTS.md) for the full, sourced numbers.

- **SL transfers under on-policy distillation too**, not only off-policy.
- At a matched **unique-prompt** budget, on- and off-policy transfer comparably — the apparent
  on-policy edge is mostly a unique-data-volume effect (on-policy regenerates rollouts every epoch).
- The trait rides the **soft forward-KL** channel (the teacher's full high-probability head);
  mode-seeking reverse-KL loses most of it.
- **top-32 support truncation is neutral** — the signal lives in the high-probability head, not the tail.

## Install

```bash
pip install -e .
```

Also requires the original subliminal-learning library (the `sl` package) for the fixed Schrodi
hyperparameters/config; data tools are vendored under `upstream/`.

## Run

```bash
# train one cell (the slug fully determines the loss/behavior)
python scripts/run_cell.py off-hard-fixed --dataset_path <prompts.jsonl> --seed 42
python scripts/run_cell.py on-soft-rkl-fresh --dataset_path <prompts.jsonl> --raw_dataset_path <raw_pool.jsonl>

# regenerate the figures from the results
python scripts/make_figures.py
```

## Layout

```
matrix/     engine — axis.py (slug ↔ config), trainer.py, losses.py, runner.py, results.py
scripts/    run_cell.py, make_figures.py, audit_rollouts.py, generate_until.py, ...
upstream/   vendored lmb-freiburg/divergence-tokens (data generation / evaluation)
docs/       RESULTS.md — source of truth for all reported numbers
figures/    generated charts
tests/      TRL loss/transfer parity + smoke tests
```

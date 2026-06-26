# AGENTS.md — subliminal-opd
<!-- keep minimal; grow only from real friction (like .gitignore) -->

## Commands
- Env setup: `uv venv --python 3.11 .venv`, then install upstream deps WITHOUT `safetytooling`
  (it pins pydantic==2.11.1, conflicting with sl's >=2.11.7; only two misalignment-GSM8K scripts need it):
  `uv pip install -p .venv <upstream deps list> "accelerate==1.9.0" && uv pip install -p .venv --no-deps -e ./upstream && uv pip install -p .venv -e ".[dev]"`
- Run a cell:   `python scripts/run_cell.py <slug> --seed 42`  (slugs: `matrix/axis.py` CANONICAL_SLUGS)
- Golden test:  `.venv/bin/python -m pytest tests/test_trl_parity.py -q` — deterministic: our off-hard span+loss == TRL on a fixed micro-batch
- Results table: `python scripts/make_results.py` → RESULTS.md (the ONLY place numbers live)
- Lint/test:    `ruff check . && python3 -m pytest tests/ -q`
- Cluster job:  on the server: `cd $REPO && PYTHONPATH="$REPO:$REPO/upstream" $PY scripts/submit_cell.py <slug> --seeds 42 …`, then `$MLS job submit -c jobs/generated/<…>.yaml`
  (kill ONLY by id: `mls job kill <id>`; **never `killall`** — shared mmikhalchuk account, would kill others' jobs; no `cancel`)

## Cluster (non-negotiable)
- Only the gritsaev workspace; never touch other users' folders.
- Agent runs LOCALLY (no proxy on MLSpace) → cluster via `ssh mlspace-jupyter` (alias in ~/.ssh/config) + `mls job submit`.
- Offline: `--no_wandb`. Poll LOG FILES (`/workspace/gritsaev/logs/<slug>_<cond>_s<seed>.log`), not the mls API (~10k/day). Don't overwrite a script mid-job.
- **`git pull` on the cluster ONLY when no our jobs are Running/Pending** — bash reads job
  scripts lazily; replacing `jobs/*.sh` under a running job ⇒ "Stale file handle" mid-script
  (burned 2026-06-10: 10 baseline jobs lost their eval step). Batch repo changes BEFORE
  submitting a fleet; freeze the repo while it runs.
- **When checking for active jobs, use `mls job table --limit 40`** (not less): smaller limits
  cut off still-Running jobs behind recent entries (burned TWICE on 2026-06-10/11 — second
  time it green-lit a pull under two running jobs, costing their eval steps).
- HF token in gitignored `.env` — don't rotate/commit.

## Cluster facts (verified 2026-06-10)
- Same NFS, two mount points: Jupyter `/home/jovyan/shares/SR004.nfs2/gritsaev` == job `/workspace-SR004.nfs2/gritsaev` (small, 92% full). Heavy volume `/workspace/gritsaev` (env `envs/sub`, `hf_cache` with gemma, `results`, `logs`; 1.1T free) — same path in jobs.
- `$REPO` = `…/gritsaev/subliminal-opd2` (clone of github.com/ndrew1337/subliminal-opd2, private). Pulls work via read-only deploy key (`core.sshCommand` is set in the clone) — no PAT on NFS.
- `$PY` = `/workspace/gritsaev/envs/sub/bin/python` (pinned stack, no pytest). `$MLS` = `/home/jovyan/.mlspace/envs/matvey_jobs_env/bin/mls` (config `~/.mls`, survives; never read its secrets).
- Canonical datasets (gen2 generation 2026-06-10 ("gen" naming avoids clashing with the freshness axis)): `/workspace/gritsaev/results/gemma-gen2/{owl,control}/seed-42/filtered_dataset.jsonl` — owl **8552** rows (gemma filter quirk), control 10007. The old-project data (`results/gemma/`) is quarantined (see Old project) — A/B-validated equivalent, set aside.
- Our runs land in `/workspace/gritsaev/results/runs/<slug>/seed-<n>/` (gen2-data runs; old-data baseline parked in `runs-olddata/`); eval writes `eval-owl/<ckpt>/stats.json`. Scratch/smoke runs park under `_`-prefixed dirs (results.py skips them).
- Job wiki: fbdocs.website.yandexcloud.net/pages/mlspace/jobs/quick_start.html (Diplodoc: content is embedded JSON, fetch raw + parse `diplodoc-state`).

## Old project (quarantine)
- The old repo (`~/subliminal-opd-buggy-archived` locally; old artifacts under
  `/workspace/gritsaev/results/gemma/` and the cluster `subliminal-opd-buggy-archived/` clone)
  is REFERENCE ONLY and presumed buggy. **Before taking ANYTHING from it** — code, data,
  checkpoints, configs, numbers — ask the user first whether it should be reused.
  (Reading it to understand logic is fine; reuse requires sign-off.)

## Convention
- Result numbers live ONLY in the generated results table (cite `slug+seeds+commit`); never in prose/comments.

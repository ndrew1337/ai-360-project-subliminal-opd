#!/usr/bin/env bash
# Cluster status: dispatcher, active jobs, and a per-cell seed grid derived from
# ARTIFACTS (checkpoints + eval stats), not log lines — so repaired evals show ✅.
# Usage: scripts/cluster_status.sh   (needs ssh alias mlspace-jupyter)
set -euo pipefail

B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; DIM=$'\033[2m'; N=$'\033[0m'

D="$HOME/.subliminal-dispatch"
if [ -d "$D" ]; then
  qh=$(grep -c . "$D/queue-high.txt" 2>/dev/null || true); qh=${qh:-0}
  qn=$(grep -c . "$D/queue.txt" 2>/dev/null || true); qn=${qn:-0}
  if pgrep -f "dispatch.sh" > /dev/null; then disp="${G}alive${N}"; else disp="${R}DEAD${N} ${DIM}(nohup ~/subliminal-opd/scripts/dispatch.sh >> ~/.subliminal-dispatch/dispatch.log 2>&1 & disown)${N}"; fi
  echo "${B}dispatcher${N} $disp ${DIM}|${N} queue high=$qh norm=$qn ${DIM}|${N} limit $(cat "$D/limit" 2>/dev/null || echo 6)"
fi

# SHOW_RAW=1 scripts/cluster_status.sh  → include the -raw cells in the grid
ssh -o BatchMode=yes -o ConnectTimeout=15 mlspace-jupyter "export SHOW_RAW='${SHOW_RAW:-}'; "'
MLS=/home/jovyan/.mlspace/envs/matvey_jobs_env/bin/mls
PY=/workspace/gritsaev/envs/sub/bin/python
echo
echo "ACTIVE_HEADER"
$MLS job table --limit 60 2>/dev/null | grep -E "subliminal" | grep -E "Running|Pending" \
  | awk -F"|" "{d=\$6; gsub(/ #gritsaev.*/,\"\",d); gsub(/^ +/,\"\",d); gsub(/ +/,\"\",\$3); gsub(/ +/,\"\",\$8); printf \"  %-44s %-8s %s\n\", d, \$3, \$8}"
$PY - <<PYEOF
import json, re, subprocess
from pathlib import Path

# queue analytics: how many foreign Pending jobs are ahead of our oldest one
table = subprocess.run(
    ["/home/jovyan/.mlspace/envs/matvey_jobs_env/bin/mls", "job", "table", "--limit", "60"],
    capture_output=True, text=True).stdout
def secs(s):
    parts = [int(x) for x in s.strip().split(":")]
    return parts[0] * 3600 + parts[1] * 60 + parts[2] if len(parts) == 3 else 0
ours_pend, others_pend, ours_run, others_run = [], [], 0, 0
for line in table.splitlines():
    cols = line.split("|")
    if len(cols) < 8:
        continue
    status, desc, dur = cols[2].strip(), cols[5], cols[7].strip()
    ours = "#gritsaev #subliminal" in desc
    if status == "Pending":
        (ours_pend if ours else others_pend).append(secs(dur))
    elif status == "Running":
        if ours: ours_run += 1
        else: others_run += 1
ahead = sum(1 for s in others_pend if ours_pend and s > max(ours_pend))
line = f"  pool: {ours_run + others_run} running ({ours_run} ours) | pending: {len(others_pend)} others, {len(ours_pend)} ours"
if ours_pend:
    line += f" | \033[1mahead of our oldest: {ahead}\033[0m"
print(line)
print()
print("GRID_HEADER")
root = Path("/workspace/gritsaev/results/runs")
cells = {}
import os
show_raw = os.environ.get("SHOW_RAW", "")
for run in sorted(root.glob("*/seed-*")):
    cell = run.parent.name
    if cell.startswith("_"):
        continue
    if cell.endswith("-raw") and not show_raw:
        continue  # hidden by default; SHOW_RAW=1 to include
    m = re.match(r"seed-(\d+)(-(.+))?$", run.name)
    if not m:
        continue
    seed, cond = m.group(1), m.group(3) or "owl"
    trained = (run / "final").is_dir()
    evaled = any(run.glob("eval-*/checkpoint-*/stats.json"))
    state = "done" if (trained and evaled) else "trained" if trained else "partial"
    cells.setdefault((cell, cond), {})[seed] = state
sym = {"done": "\033[32m●\033[0m", "trained": "\033[33m◐\033[0m", "partial": "\033[31m○\033[0m"}
for (cell, cond), seeds in sorted(cells.items()):
    label = cell + ("" if cond == "owl" else f" [{cond}]")
    row = " ".join(f"{s}{sym[seeds[s]]}" for s in sorted(seeds))
    print(f"  {label:<42s} {row}")
PYEOF' | sed -e "s/ACTIVE_HEADER/${B}active jobs${N}/" -e "s/GRID_HEADER/\\
${B}matrix grid${N} ${DIM}(artifacts: ${N}${G}●${N}${DIM} eval done · ${N}${Y}◐${N}${DIM} trained, no eval · ${N}${R}○${N}${DIM} incomplete)${N}/"

#!/usr/bin/env bash
# Local job dispatcher: keeps at most $LIMIT of our jobs active on the cluster,
# feeding them from a two-tier queue. Runs on the Mac (not the cluster).
#
# Queue files (one yaml name per line, must exist in jobs/generated/ on the server):
#   ~/.subliminal-dispatch/queue-high.txt   — consumed first (priority)
#   ~/.subliminal-dispatch/queue.txt        — normal
# Reordering/insertion = edit the files; they are plain text. Examples:
#   echo "on-soft-rkl-fixed-k32-owl-s42" >> ~/.subliminal-dispatch/queue.txt
#   sed -i '' '1i\
#   urgent-job-name' ~/.subliminal-dispatch/queue-high.txt
#
# Runs as a daemon: idles when queues are empty. Stop with: pkill -f dispatch.sh
set -uo pipefail

D="$HOME/.subliminal-dispatch"
mkdir -p "$D"
touch "$D/queue-high.txt" "$D/queue.txt"
MLS=/home/jovyan/.mlspace/envs/matvey_jobs_env/bin/mls
GEN=/home/jovyan/shares/SR004.nfs2/gritsaev/subliminal-opd2/jobs/generated

pop_next() {  # echoes "file name" of the next queued job, prefers high
  for q in "$D/queue-high.txt" "$D/queue.txt"; do
    if [ -s "$q" ]; then
      echo "$q $(head -1 "$q")"
      return 0
    fi
  done
  return 1
}

while true; do
  if ! [ -s "$D/queue-high.txt" ] && ! [ -s "$D/queue.txt" ]; then
    sleep 300   # idle: queues empty; keep running as a daemon
    continue
  fi
  table=$(ssh -o BatchMode=yes -o ConnectTimeout=20 mlspace-jupyter \
    "$MLS job table --limit 60 2>/dev/null" 2>/dev/null)
  active=$(echo "$table" | grep "#gritsaev #subliminal" | grep -cE "Running|Pending" | tr -dc 0-9)
  active=${active:-99}
  others_pending=$(echo "$table" | grep -v "#gritsaev #subliminal" | grep -c "Pending")
  others_running=$(echo "$table" | grep -v "#gritsaev #subliminal" | grep -c "Running")
  cfg=$(cat "$D/limit" 2>/dev/null || echo auto)   # re-read each tick: number = fixed, "auto" = adaptive
  case "$cfg" in
    auto)
      if [ "${others_pending:-0}" -gt 0 ]; then LIMIT=6        # colleagues queueing: back off
      elif [ "${others_running:-0}" -ge 10 ]; then LIMIT=8     # pool busy
      elif [ "${others_running:-0}" -ge 4 ]; then LIMIT=10     # moderately free
      else LIMIT=12; fi                                        # pool free
      ;;
    *) LIMIT=$cfg ;;
  esac
  last=$(cat "$D/.last_limit" 2>/dev/null || echo "")
  if [ "$LIMIT" != "$last" ]; then
    echo "LIMIT -> $LIMIT ($cfg: others running=$others_running pending=$others_pending)"
    echo "$LIMIT" > "$D/.last_limit"
  fi
  slots=$((LIMIT - active))
  while [ "$slots" -gt 0 ]; do
    next=$(pop_next) || break
    qfile=${next%% *}; name=${next#* }
    ok=$(ssh -o BatchMode=yes -o ConnectTimeout=20 mlspace-jupyter \
      "$MLS job submit -c $GEN/${name}.yaml 2>&1 | grep -c job_name" 2>/dev/null | tr -dc 0-9)
    if [ "${ok:-0}" -ge 1 ]; then
      sed -i '' 1d "$qfile" 2>/dev/null || sed -i 1d "$qfile"
      tier=$([ "$qfile" = "$D/queue-high.txt" ] && echo HIGH || echo norm)
      echo "SUBMITTED[$tier]: $name ($(cat "$D/queue-high.txt" "$D/queue.txt" | grep -c . ) left)"
      slots=$((slots - 1))
    else
      echo "SUBMIT FAILED: $name — retry next tick"
      break
    fi
  done
  sleep 600
done

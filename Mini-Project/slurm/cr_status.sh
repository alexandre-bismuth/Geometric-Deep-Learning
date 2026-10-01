#!/bin/bash
# Progress of the camera-ready queue: one line per task (DONE rc / FAILED rc / RETRIES-EXHAUSTED /
# STARTED n× / pending), then totals. Run from $MP:
#   bash slurm/cr_status.sh [--pending|--failed]
#   bash slurm/cr_status.sh --done-runs   # outputs/<run> dirs touched by a DONE task (to git add)
QUEUE="${QUEUE:-manifests/camera_ready_q1.txt}"
STATE="${STATE:-manifests/cr1_state}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"
mode=${1:-all}
nd=0; nf=0; np=0; nx=0
out=$(mktemp)
run_of_line() {   # outputs dir a queue line writes to
  set -- $1
  if [ "$1" = "train" ]; then
    local id; id=$(awk '/^experiment_id:/ {print $2; exit}' "$3")
    case "$id" in *-s"$4") echo "outputs/$id" ;; *) echo "outputs/$id-s$4" ;; esac
  else echo "$2"; fi
}
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue ;; esac
  key=$(printf '%s' "$line" | md5sum | cut -d' ' -f1)
  att=$(cat "$STATE/$key.attempts" 2>/dev/null | wc -l | tr -d ' ')
  if [ -f "$STATE/$key.rc" ]; then
    rc=$(cut -d' ' -f1 "$STATE/$key.rc")
    if [ "$rc" = "0" ]; then nd=$((nd + 1)); s="DONE"; else nf=$((nf + 1)); s="FAILED rc=$rc"; fi
  elif [ "${att:-0}" -ge "$MAX_ATTEMPTS" ]; then nx=$((nx + 1)); s="RETRIES-EXHAUSTED ($att)"
  else np=$((np + 1)); s="pending (started ${att:-0}x)"; fi
  case "$mode" in
    --pending) case "$s" in pending*) echo "$s | $line" ;; esac ;;
    --failed)  case "$s" in FAILED*|RETRIES*) echo "$s | $line" ;; esac ;;
    --done-runs) [ "$s" = "DONE" ] && run_of_line "$line" ;;
    *) echo "$s | $line" ;;
  esac
done < "$QUEUE" > "$out"
if [ "$mode" = "--done-runs" ]; then sort -u "$out"; else cat "$out"
  echo "TOTAL done=$nd failed=$nf exhausted=$nx pending=$np"; fi
rm -f "$out"

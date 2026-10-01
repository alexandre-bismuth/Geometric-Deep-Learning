#!/bin/bash
# Progress of the camera-ready queue: one line per task (DONE rc / FAILED rc / RETRIES-EXHAUSTED /
# STARTED n× / pending), then totals. Run from $MP:  bash slurm/cr_status.sh [--pending|--failed]
QUEUE="${QUEUE:-manifests/camera_ready_q1.txt}"
STATE="${STATE:-manifests/cr1_state}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"
mode=${1:-all}
nd=0; nf=0; np=0; nx=0
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
    *) echo "$s | $line" ;;
  esac
done < "$QUEUE"
echo "TOTAL done=$nd failed=$nf exhausted=$nx pending=$np"

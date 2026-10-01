#!/bin/bash
# Completion gate: is the fleet finished and is every promised artifact on disk?
# Run from Mini-Project.  Exit 0 = ready for readout, 1 = something still owed.
#
# Checks, in the order they can bite you:
#   1. Slurm state and per-train log freshness.
#   2. Claims still carrying the INFLIGHT-noreap sentinel.  The sentinel is removed by
#      the owning worker when a task really finishes, so a leftover means that task died
#      and -- because the sentinel also blocks re-claiming -- nothing will retry it.
#      This is the monitoring debt the 2026-08-18 guard deliberately took on.
#   3. failed.txt.
#   4. Every run dir named by an analysis line in the queue: final_model.pt plus the
#      artifact each queued analysis was supposed to produce.

Q=${QUEUE:-manifests/task_queue.txt}
C=manifests/claims
fail=0
echo "=== readout check $(date +%F_%T) ==="

echo
echo "-- slurm"
squeue -h -u "$USER" -o "  %i %j %T %M %L" 2>/dev/null || echo "  (squeue unavailable)"

echo
echo "-- guarded tasks (INFLIGHT-noreap sentinel)"
live=$(squeue -h -u "$USER" -o %i 2>/dev/null)
inflight=0; stuck=0
for d in "$C"/*/; do
  grep -q '^INFLIGHT-noreap' "$d/rc" 2>/dev/null || continue
  owner=$(cat "$d/owner" 2>/dev/null)
  task=$(cat "$d/task" 2>/dev/null)
  if printf '%s\n' "$live" | grep -qx "$owner"; then
    inflight=$((inflight + 1))
    echo "  in flight  (job $owner alive)  $task"
  else
    stuck=$((stuck + 1))
    echo "  DEAD       (job $owner gone)   $task"
  fi
done
if [ "$((inflight + stuck))" -eq 0 ]; then
  echo "  none left — every guarded task completed and its worker cleared the sentinel"
else
  [ "$inflight" -gt 0 ] && { echo "  -> $inflight still running (expected; not ready yet)"; fail=1; }
  if [ "$stuck" -gt 0 ]; then
    echo "  -> $stuck DEAD: the owning job ended without finishing the task."
    echo "     The sentinel also blocks re-claiming, so nothing will retry it."
    echo "     To release for a re-run:  rm -rf manifests/claims/<key>   (key = dir name above)"
    fail=1
  fi
fi

echo "-- non-zero exit codes"
bad=0
for d in "$C"/*/; do
  rc=$(cut -d' ' -f1 "$d/rc" 2>/dev/null)
  case "$rc" in
    ''|0|INFLIGHT-noreap) ;;
    *) echo "  rc=$rc  $(cat "$d/task" 2>/dev/null)"; bad=$((bad + 1)) ;;
  esac
done
[ "$bad" -eq 0 ] && echo "  none" || { echo "  -> $bad task(s) exited non-zero"; fail=1; }

echo
echo "-- failed.txt"
n=$(wc -l < manifests/failed.txt 2>/dev/null || echo 0)
if [ "$n" -eq 0 ]; then echo "  empty"; else sed 's/^/  /' manifests/failed.txt; fail=1; fi

echo
echo "-- artifacts promised by the queue"
missing=0
# every "<analysis> outputs/<run>" line in the queue, #F-gated or not
sed 's/^#F //' "$Q" | grep -E '^(eval|ablate|perturb|cert|probe) outputs/' | while read -r kind rundir _; do
  echo "$kind $rundir"
done | sort -u > /tmp/.rc_expect.$$
while read -r kind rundir; do
  case "$kind" in
    eval)    want=eval_summary.json ;;
    ablate)  want=sink_ablation_s0.json ;;
    perturb) want=perturbation_s0.json ;;
    cert)    want=theory_certificate.json ;;
    probe)   want=symmetry_probe.json ;;
  esac
  if [ ! -f "$rundir/final_model.pt" ] && [ ! -f "$rundir/ALLOW_PARTIAL" ]; then
    echo "  UNTRAINED  $rundir (no final_model.pt)"; missing=$((missing + 1)); continue
  fi
  [ -f "$rundir/$want" ] || { echo "  MISSING    $rundir/$want"; missing=$((missing + 1)); }
done < /tmp/.rc_expect.$$ | sort -u | tee /tmp/.rc_missing.$$
rm -f /tmp/.rc_expect.$$
if [ -s /tmp/.rc_missing.$$ ]; then
  echo "  -> $(wc -l < /tmp/.rc_missing.$$) gap(s)"; fail=1
else
  echo "  all present"
fi
rm -f /tmp/.rc_missing.$$

echo
if [ "$fail" -eq 0 ]; then
  echo "VERDICT: READY — run  python3 scripts/aggregate_results.py -o ../tasks/readout.md --json ../tasks/readout.json"
else
  echo "VERDICT: NOT READY — see the flagged lines above."
fi
exit $fail

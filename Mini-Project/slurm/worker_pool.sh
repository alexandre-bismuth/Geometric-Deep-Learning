#!/bin/bash
# Shared queue-draining worker pool. Source AFTER cd to Mini-Project with env set.
# Queue file: manifests/task_queue.txt, one task per line:
#   train <grit|gps> <config.yaml> <seed>
#   eval  <run-dir | config.yaml> <full|N> <trained|randinitS>
#   perturb <run-dir>
#   ablate  <run-dir>
#   ablatenr <run-dir>                 (camera-ready: ablation without row renormalisation)
#   evalmode <run-dir> <attn_mode>     (camera-ready: eval a GRIT ckpt under another scoring path)
# Claims are mkdir-atomic (NFS-safe) under manifests/claims/<md5-of-line>.
# An rc file in a claim dir means "hands off": reap_stale skips it and no worker will
# re-claim it. Writing a sentinel rc (`INFLIGHT-noreap ...`) into a LIVE claim is therefore
# a manual guard against a duplicate being started on top of a running task; the owning
# worker overwrites it with the real value on completion, so bookkeeping stays correct.
# A finished task writes rc into its claim dir; failed tasks are appended to
# manifests/failed.txt (NOT auto-retried — stale claims without rc mean a worker died
# mid-task; release with: rmdir manifests/claims/<key> after checking).

QUEUE="${QUEUE:-manifests/task_queue.txt}"
CLAIMS=manifests/claims
mkdir -p "$CLAIMS" logs

hours_left() {  # whole hours of walltime left in this job (999 outside Slurm)
  local L
  L=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null | head -1)
  if [ -z "$L" ]; then echo 999; return; fi
  local d=0 a b c
  case "$L" in *-*) d=${L%%-*}; L=${L#*-} ;; esac
  IFS=: read -r a b c <<< "$L"
  if [ -z "$c" ]; then c=$b; b=$a; a=0; fi
  echo $(( (d*86400 + 10#$a*3600 + 10#$b*60 + 10#$c) / 3600 ))
}

est_hours() {  # conservative per-task duration estimate (measured 2026-08-17 on GH200)
  case "$1" in
    train*peptides_grit_q*) echo 3 ;;   # quartile / matched-quartile
    train*peptides*)        echo 7 ;;   # full peptides
    train\ gps\ *)          echo 11 ;;  # ZINC GPS 1500 ep (5.5-9 h measured)
    train*)                 echo 16 ;;  # ZINC GRIT 2000 ep (8-14 h measured)
    *)                      echo 2 ;;   # eval / perturb / ablate
  esac
}

reap_stale() {  # release claims whose owning job is gone (walltime kill / crash)
  local live
  live=$(squeue -h -u "$USER" -o %i 2>/dev/null)
  [ -z "$live" ] && return 0          # squeue broken: never reap
  local now d owner tag age
  now=$(date +%s)
  for d in "$CLAIMS"/*/; do
    d=${d%/}
    [ -d "$d" ] || continue
    [ -e "$d/rc" ] && continue
    age=$(( now - $(stat -c %Y "$d" 2>/dev/null || echo "$now") ))
    [ "$age" -lt 900 ] && continue    # 15 min grace: never race a fresh claim
    if [ -f "$d/owner" ]; then
      owner=$(cat "$d/owner")
    else                              # legacy claim: recover job id from its task log
      [ -f "$d/task" ] || continue
      tag=$(tr ' /' '__' < "$d/task" | tr -cd 'A-Za-z0-9_.-' | cut -c1-90)
      owner=$(ls -1 "logs/task_${tag}_"*.log 2>/dev/null | tail -1)
      owner=${owner##*_}; owner=${owner%.log}
      [ -z "$owner" ] && continue     # no evidence at all: leave it alone
    fi
    [ -z "$owner" ] && continue       # unattributable: never reap on no evidence
    if ! printf '%s\n' "$live" | grep -qx "$owner"; then
      # 2026-08-18: a bulk `squeue` that returns a PARTIAL list looks identical to "job
      # gone" and falsely reaped a live train, producing two processes writing one output
      # dir. Absence from the bulk list is now only a suspicion; confirm it three ways.
      # 1. Prove squeue is answering correctly right now: it must see the job we are IN.
      squeue -h -j "${SLURM_JOB_ID:-0}" -o %i 2>/dev/null \
        | grep -qx "${SLURM_JOB_ID:-0}" || continue
      # 2. Ask about that one job directly rather than trusting the bulk list.
      [ -n "$(squeue -h -j "$owner" -o %i 2>/dev/null)" ] && continue
      # 3. Re-ask after a pause; a transient controller hiccup will not repeat identically.
      sleep 10
      [ -n "$(squeue -h -j "$owner" -o %i 2>/dev/null)" ] && continue
      echo "[reap] releasing stale claim ${d##*/} (owner job $owner gone): $(cat "$d/task" 2>/dev/null)"
      rm -rf "$d" 2>/dev/null || true
    fi
  done
}

run_task() {
  local TYPE=$1; shift
  case "$TYPE" in
    train)
      local RUNNER=$1 CONFIG=$2 SEED=$3
      local SCRIPT=scripts/run_experiment.py
      [ "$RUNNER" = "gps" ] && SCRIPT=scripts/run_experiment_graphgps.py
      python -u "$SCRIPT" --config "$CONFIG" --no-wandb --training.seed "$SEED"
      ;;
    eval)
      local TARGET=$1 MG=${2:-full} MODE=${3:-trained}
      local ARGS=()
      [ "$MG" != "full" ] && ARGS+=(--max-graphs "$MG")
      if [ "$MODE" != "trained" ]; then
        ARGS+=(--randinit --seed "${MODE#randinit}")
      fi
      case "$TARGET" in
        *.yaml) python -u scripts/run_eval_suite.py --config "$TARGET" "${ARGS[@]}" ;;
        *)      python -u scripts/run_eval_suite.py --run-dir "$TARGET" "${ARGS[@]}" ;;
      esac
      ;;
    perturb) python -u scripts/run_perturbation.py --run-dir "$1" ;;
    ablate)  python -u scripts/run_sink_ablation.py --run-dir "$1" ;;
    ablatenr) python -u scripts/run_sink_ablation.py --run-dir "$1" --no-renorm ;;
    evalmode) python -u scripts/run_eval_suite.py --run-dir "$1" --attn-mode "$2" --no-spectra ;;
    cert)    python -u scripts/run_theory_certificate.py --run-dir "$1" ;;
    probe)   python -u scripts/run_symmetry_probe.py --run-dir "$1" ;;
    *) echo "unknown task type: $TYPE"; return 2 ;;
  esac
}

worker() {
  local gpu=$1
  local empty=0
  local skip_keys=""     # tasks found "not ready" this pass (run still training)
  while :; do
    local claimed=""
    local key=""
    local hleft
    hleft=$(hours_left)
    reap_stale
    while IFS= read -r line; do
      [ -z "$line" ] && continue
      # "#F <task>" = capability-gated task: workers still running an older copy of this
      # file skip it as a comment, so new task types never reach a pool that can't run them.
      local task_line
      case "$line" in
        '#F '*) task_line=${line#'#F '} ;;
        \#*)    continue ;;
        *)      task_line=$line ;;
      esac
      if [ "$(est_hours "$task_line")" -gt "$hleft" ]; then continue; fi
      key=$(printf '%s' "$line" | md5sum | cut -d' ' -f1)
      if [ -e "$CLAIMS/$key/rc" ]; then continue; fi
      case " $skip_keys " in *" $key "*) continue;; esac
      if mkdir "$CLAIMS/$key" 2>/dev/null; then
        claimed="$task_line"
        printf '%s\n' "$task_line" > "$CLAIMS/$key/task"
        printf '%s\n' "${SLURM_JOB_ID:-local}" > "$CLAIMS/$key/owner"
        break
      fi
    done < "$QUEUE"
    if [ -z "$claimed" ]; then
      # Nothing claimable *right now* — another job may still die and free its claims,
      # so idle-poll before giving up (reap_stale runs at the top of every cycle).
      empty=$((empty + 1))
      if [ "$empty" -ge "${EMPTY_LIMIT:-36}" ]; then
        echo "[worker$gpu] queue drained ($empty empty cycles), exiting"
        break
      fi
      echo "[worker$gpu] nothing claimable (cycle $empty/${EMPTY_LIMIT:-36}), sleeping 300s"
      skip_keys=""       # retry not-ready tasks next cycle
      sleep 300
      continue
    fi
    empty=0
    # Analysis task on a run that has not finished training yet: release and retry later,
    # never score a half-trained checkpoint.
    case "$claimed" in
      eval\ outputs/*|cert\ outputs/*|probe\ outputs/*|perturb\ outputs/*|ablate\ outputs/*|ablatenr\ outputs/*|evalmode\ outputs/*)
        local rundir=${claimed#* }; rundir=${rundir%% *}
        # ALLOW_PARTIAL: escape hatch for a run the walltime killed. Every analysis script
        # scores best_model.pt anyway, so a run stopped at e.g. 1900/2000 epochs is still
        # usable — touch that file in the run dir to let its analyses through.
        if [ ! -f "$rundir/final_model.pt" ] && [ ! -f "$rundir/ALLOW_PARTIAL" ]; then
          echo "[worker$gpu] not ready: $rundir still training — releasing $claimed"
          rm -rf "${CLAIMS:?}/$key"
          skip_keys="$skip_keys $key"
          continue
        fi ;;
    esac
    local tag
    tag=$(printf '%s' "$claimed" | tr ' /' '__' | tr -cd 'A-Za-z0-9_.-' | cut -c1-90)
    echo "[worker$gpu] $(date +%F_%T) START $claimed"
    CUDA_VISIBLE_DEVICES=$gpu run_task $claimed \
      > "logs/task_${tag}_${SLURM_JOB_ID:-local}.log" 2>&1
    local rc=$?
    echo "$rc $(date +%F_%T) job=${SLURM_JOB_ID:-local} gpu=$gpu" > "$CLAIMS/$key/rc"
    echo "[worker$gpu] $(date +%F_%T) DONE rc=$rc $claimed"
    if [ "$rc" -ne 0 ]; then
      printf '%s\n' "$claimed" >> manifests/failed.txt
    fi
    chmod -R g+rwX outputs logs manifests 2>/dev/null || true
  done
}

start_pool() {
  local n=${1:-4}
  for g in $(seq 0 $((n - 1))); do
    worker "$g" &
  done
  wait
  echo "POOL_DONE $(date +%F_%T)"
}

#!/bin/bash
# Single-node task pool for the GDDL camera-ready (sourced by slurm/cr_node.sbatch).
#
# ONE NODE ONLY. All workers run on the node this job holds; nothing is spread elsewhere.
# Lustre hygiene:
#   - every task runs in a node-local work dir ($LOCAL/work, under $LOCALDIR): symlinks to
#     configs/ scripts/ src/, a REAL local outputs/, and a local data/ whose small PE caches
#     are copied in once per job (big ones are symlinked). Checkpoints, eval files and logs
#     are written to local disk and copied back with rsync at explicit points (end of each
#     task, every SYNC_MIN minutes for resumability, and at USR1/TERM before the walltime);
#   - no ls/find discovery: every path is computed from the queue line;
#   - wandb is imported but disabled (WANDB_MODE=disabled, --no-wandb).
#
# Queue lines (same grammar as slurm/worker_pool.sh):
#   train <grit|gps> <config.yaml> <seed>
#   eval <run-dir> <full|N> <trained>     ablate|ablatenr|cert|probe|perturb|reeval <run-dir>
# State on Lustre ($STATE): <md5(line)>.rc (exit code, job, time) marks a task finished;
# <md5>.attempts counts starts. A task killed by the walltime writes no .rc and is retried
# by the next job (a train resumes from its ckpt.pt). rc != 0 is NOT retried; a task with
# >= MAX_ATTEMPTS starts and no .rc is skipped (reported by slurm/cr_status.sh).

: "${MP:?MP must point at the Mini-Project dir of the camera-ready worktree}"
: "${QUEUE:=manifests/camera_ready_q1.txt}"
: "${STATE:=manifests/cr1_state}"
: "${WORKERS_PER_GPU:=1}"
: "${NGPU:=4}"
: "${MAX_ATTEMPTS:=6}"
: "${SYNC_MIN:=60}"
: "${STAGE_MAX_MB:=4000}"   # PE caches up to this size are copied to node-local disk
LOCAL="${LOCAL:-${LOCALDIR:-${TMPDIR:-/tmp}}/gddl_cr_${SLURM_JOB_ID:-local}}"
JOBTAG="${SLURM_JOB_ID:-local}"

key_of() { printf '%s' "$1" | md5sum | cut -d' ' -f1; }

run_of_train() {   # train <runner> <config> <seed>  ->  outputs dir name (experiment_id-s<seed>)
  local cfg=$3 seed=$4 id
  id=$(awk '/^experiment_id:/ {print $2; exit}' "$MP/$cfg")
  case "$id" in *-s"$seed") echo "$id" ;; *) echo "$id-s$seed" ;; esac
}

setup_local() {
  mkdir -p "$LOCAL/work/outputs" "$LOCAL/claims" "$LOCAL/logs" "$LOCAL/locks" "$MP/$STATE" \
           "$MP/logs/cr_$JOBTAG"
  local d
  for d in configs scripts src slurm manifests; do
    [ -e "$LOCAL/work/$d" ] || ln -s "$MP/$d" "$LOCAL/work/$d"
  done
  # data/: a REAL local dir. Raw dataset dirs are symlinked (only read if a cache is
  # missing); PE caches are staged by stage_caches below.
  mkdir -p "$LOCAL/work/data/pe_cache"
  for d in ZINC LRGB; do
    [ -e "$MP/data/$d" ] && [ ! -e "$LOCAL/work/data/$d" ] && ln -s "$MP/data/$d" "$LOCAL/work/data/$d"
  done
  # run -> train-key map, so analysis tasks wait for THEIR train (no final_model.pt race)
  : > "$LOCAL/run2train.tsv"
  local line
  while IFS= read -r line; do
    case "$line" in train\ *) ;; *) continue ;; esac
    # shellcheck disable=SC2086
    printf '%s\t%s\n' "$(run_of_train $line)" "$(key_of "$line")" >> "$LOCAL/run2train.tsv"
  done < "$MP/$QUEUE"
}

stage_caches() {   # stage_caches <file listing data/pe_cache/... paths, one per line>
  # small caches (ZINC) are copied to node-local disk once per job; big ones (Peptides
  # RRWP, ~25 GB) are symlinked and read straight from Lustre, once per process.
  local p mb
  while IFS= read -r p; do
    [ -z "$p" ] && continue
    [ -f "$MP/$p" ] || { echo "[stage] MISSING $p"; continue; }
    mb=$(( $(stat -c %s "$MP/$p" 2>/dev/null || stat -f %z "$MP/$p") / 1000000 ))
    if [ "$mb" -le "$STAGE_MAX_MB" ]; then
      cp "$MP/$p" "$LOCAL/work/$p" && echo "[stage] copied $p (${mb} MB)"
    else
      ln -sf "$MP/$p" "$LOCAL/work/$p" && echo "[stage] linked $p (${mb} MB, read from Lustre)"
    fi
  done < "$1"
}

sync_run() {   # local run dir -> home NFS (newer files only; never temp files)
  local run=$1
  [ -d "$LOCAL/work/outputs/$run" ] || return 0
  mkdir -p "$MP/outputs/$run"
  rsync -a --update --exclude '*.tmp' "$LOCAL/work/outputs/$run/" "$MP/outputs/$run/"
}

sync_all() {
  rsync -a --update --exclude '*.tmp' "$LOCAL/work/outputs/" "$MP/outputs/" 2>/dev/null
  rsync -a "$LOCAL/logs/" "$MP/logs/cr_$JOBTAG/" 2>/dev/null
  true
}

stage_in() {   # Lustre run dir -> local, once per job per run (lock: mkdir is atomic locally)
  local run=$1
  [ -d "$MP/outputs/$run" ] || return 0
  until mkdir "$LOCAL/locks/stage_$run" 2>/dev/null; do sleep 2; done
  if [ ! -e "$LOCAL/locks/staged_$run" ]; then
    mkdir -p "$LOCAL/work/outputs/$run"
    rsync -a --update --exclude '*.tmp' "$MP/outputs/$run/" "$LOCAL/work/outputs/$run/"
    touch "$LOCAL/locks/staged_$run"
  fi
  rmdir "$LOCAL/locks/stage_$run"
}

analysis_ready() {   # 0 if the run an analysis line targets is fully trained
  local run=$1 tk
  tk=$(awk -F'\t' -v r="$run" '$1 == r {print $2; exit}' "$LOCAL/run2train.tsv")
  if [ -n "$tk" ]; then
    [ -f "$MP/$STATE/$tk.rc" ] && [ "$(cut -d' ' -f1 "$MP/$STATE/$tk.rc")" = "0" ]
  else
    [ -f "$MP/outputs/$run/final_model.pt" ]
  fi
}

run_task() {   # executed inside $LOCAL/work
  local TYPE=$1; shift
  case "$TYPE" in
    train)
      local SCRIPT=scripts/run_experiment.py
      [ "$1" = "gps" ] && SCRIPT=scripts/run_experiment_graphgps.py
      python -u "$SCRIPT" --config "$2" --no-wandb --training.seed "$3" ;;
    eval)     python -u scripts/run_eval_suite.py --run-dir "$1" ;;
    reeval)   python -u scripts/run_eval_suite.py --run-dir "$1" --out-tag cr --no-spectra ;;
    ablate)   python -u scripts/run_sink_ablation.py --run-dir "$1" ;;
    ablatenr) python -u scripts/run_sink_ablation.py --run-dir "$1" --no-renorm ;;
    cert)     python -u scripts/run_theory_certificate.py --run-dir "$1" ;;
    probe)    python -u scripts/run_symmetry_probe.py --run-dir "$1" ;;
    perturb)  python -u scripts/run_perturbation.py --run-dir "$1" ;;
    *) echo "unknown task type: $TYPE"; return 2 ;;
  esac
}

worker() {   # worker <id> <gpu>
  local wid=$1 gpu=$2 line key run tag rc att
  local idle=0
  while :; do
    [ -e "$LOCAL/STOP" ] && { echo "[w$wid] STOP flag, exiting"; return 0; }
    local claimed="" deferred=0
    while IFS= read -r line; do
      [ -z "$line" ] && continue
      case "$line" in \#*) continue ;; esac
      key=$(key_of "$line")
      [ -e "$MP/$STATE/$key.rc" ] && continue
      att=$(cat "$MP/$STATE/$key.attempts" 2>/dev/null | wc -l | tr -d ' ')
      [ "${att:-0}" -ge "$MAX_ATTEMPTS" ] && continue
      case "$line" in
        train\ *) ;;
        *) run=${line#* }; run=${run%% *}; run=${run#outputs/}
           if ! analysis_ready "$run"; then deferred=1; continue; fi ;;
      esac
      if mkdir "$LOCAL/claims/$key" 2>/dev/null; then claimed=$line; break; fi
    done < "$MP/$QUEUE"
    if [ -z "$claimed" ]; then
      # nothing claimable: either everything is done/running, or analyses await trains
      if [ "$deferred" -eq 0 ] && [ "$idle" -ge 2 ]; then
        echo "[w$wid] queue drained, exiting"; return 0
      fi
      idle=$((idle + 1)); sleep 120; continue
    fi
    idle=0
    echo "$JOBTAG $(date +%F_%T)" >> "$MP/$STATE/$key.attempts"
    tag=$(printf '%s' "$claimed" | tr ' /' '__' | tr -cd 'A-Za-z0-9_.-' | cut -c1-90)
    case "$claimed" in
      train\ *) # shellcheck disable=SC2086
                run=$(run_of_train $claimed) ;;
      *)        run=${claimed#* }; run=${run%% *}; run=${run#outputs/} ;;
    esac
    stage_in "$run"
    echo "[w$wid gpu$gpu] $(date +%F_%T) START $claimed"
    # shellcheck disable=SC2086
    ( cd "$LOCAL/work" && CUDA_VISIBLE_DEVICES=$gpu run_task $claimed ) \
        > "$LOCAL/logs/task_${tag}.log" 2>&1
    rc=$?
    if [ "$rc" -eq 143 ] || [ "$rc" -eq 137 ] || [ -e "$LOCAL/KILLED" ]; then
      echo "[w$wid] $(date +%F_%T) INTERRUPTED rc=$rc $claimed (no .rc written; next job retries)"
      sync_run "$run"; rmdir "$LOCAL/claims/$key" 2>/dev/null; return 0
    fi
    sync_run "$run"
    case "$claimed" in train\ *) [ "$rc" -eq 0 ] && rm -f "$MP/outputs/$run/ckpt.pt" ;; esac
    echo "$rc job=$JOBTAG gpu=$gpu end=$(date +%F_%T)" > "$MP/$STATE/$key.rc"
    echo "[w$wid gpu$gpu] $(date +%F_%T) DONE rc=$rc $claimed"
    rsync -a "$LOCAL/logs/task_${tag}.log" "$MP/logs/cr_$JOBTAG/" 2>/dev/null
  done
}

start_pool() {
  local n=$((NGPU * WORKERS_PER_GPU)) w
  for w in $(seq 0 $((n - 1))); do
    worker "$w" "$((w % NGPU))" &
  done
  wait
  echo "POOL_DONE $(date +%F_%T)"
}

#!/bin/bash
# GDDL 2026 camera-ready: one job = ONE node (4 GPUs) draining ONE queue file.
# Limits (user, 2026-10-01): ONE round, at most TWO of these jobs at once (each on its own
# node, each on its own DISJOINT queue file), every job <= 20 h, no resubmits. Enforced below:
# >1 node per job, a third gddl-cr job, a running/queued calibration job, or a queue file
# already being drained by another live job -> refuse.
# Submit from $MP, one per node queue:
#   cd "$MP" && sbatch --export=ALL,MP="$MP",VENV="$VENV",WORKERS_PER_GPU=2,QUEUE=manifests/cr_nodeA.txt slurm/cr_node.sbatch
#   cd "$MP" && sbatch --export=ALL,MP="$MP",VENV="$VENV",WORKERS_PER_GPU=2,QUEUE=manifests/cr_nodeB.txt slurm/cr_node.sbatch
# There is no follow-up job: a train not finished at the walltime is lost; analyses not run
# can be done afterwards on CPU from the committed checkpoints. The job prints the
# unfinished and failed tasks at USR1 (15 min before the walltime) and at its end.
#SBATCH --job-name=gddl-cr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=workq
#SBATCH --time=20:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=logs/gddl-cr-%j.out
set -uo pipefail
umask 002

: "${MP:?pass MP via --export=ALL,MP=...}"
: "${VENV:?pass VENV via --export=ALL,VENV=...}"
cd "$MP" || { echo "bad MP=$MP"; exit 1; }
[ -x "$VENV/bin/python" ] || { echo "no python at $VENV/bin/python"; exit 1; }
export PATH="$VENV/bin:$PATH"
export PYTHONUNBUFFERED=1 WANDB_MODE=disabled MPLBACKEND=Agg

# FIRST: the venv must run on this compute node and see the GPUs (REQUIRE_CUDA=0 for CPU tests)
if [ "${REQUIRE_CUDA:-1}" = "1" ]; then
  python -c "import torch, torch_geometric; assert torch.cuda.is_available(), 'no CUDA'; \
print('cuda ok:', torch.cuda.device_count(), 'GPUs')" || { echo "venv/CUDA check FAILED"; exit 2; }
fi

# guards: one node per job; at most one OTHER main job; no calibration job alongside
if [ "${SLURM_JOB_NUM_NODES:-1}" != "1" ]; then echo "REFUSING: ${SLURM_JOB_NUM_NODES} nodes"; exit 3; fi
others=$(squeue -h -u "$USER" -n gddl-cr -t RUNNING,PENDING -o %i 2>/dev/null | grep -vx "${SLURM_JOB_ID:-x}" | wc -l)
if [ "$others" -gt 1 ]; then echo "REFUSING: $others other gddl-cr jobs (at most one allowed)"; exit 3; fi
calib=$(squeue -h -u "$USER" -n gddl-cr-calib -t RUNNING,PENDING -o %i 2>/dev/null | wc -l)
if [ "$calib" -ne 0 ]; then echo "REFUSING: a gddl-cr-calib job is still queued or running"; exit 3; fi
job_alive() { [ -n "$1" ] && squeue -h -j "$1" -o %i 2>/dev/null | grep -qx "$1"; }

source slurm/cr_pool.sh
# queue lock: two jobs must never drain the same queue file (claims are node-local)
mkdir -p "$MP/$STATE/queue_locks"
QLOCK="$MP/$STATE/queue_locks/$(basename "$QUEUE")"
if ! mkdir "$QLOCK" 2>/dev/null; then
  qowner=$(cat "$QLOCK/owner" 2>/dev/null)
  if job_alive "$qowner"; then echo "REFUSING: $QUEUE is being drained by live job $qowner"; exit 3; fi
  echo "taking over the stale lock on $QUEUE (job ${qowner:-?} is gone)"
fi
echo "${SLURM_JOB_ID:-local}" > "$QLOCK/owner"
setup_local
echo "TMPDIR=${TMPDIR:-} LOCALDIR=${LOCALDIR:-}"; df -h "$LOCAL" /tmp 2>/dev/null
echo "job ${SLURM_JOB_ID:-local} on $(hostname) $(date +%F_%T)"
echo "MP=$MP LOCAL=$LOCAL QUEUE=$QUEUE STATE=$STATE WORKERS_PER_GPU=$WORKERS_PER_GPU"
echo "HEAD $(git -C "$MP" rev-parse HEAD 2>/dev/null)"
python -c "import torch, torch_geometric as g; from src.model import GritMultiHeadAttention as G; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'pyg', g.__version__, 'GRIT_FAMILY', G.GRIT_FAMILY)"
nvidia-smi -L 2>/dev/null || true

# 1. PE caches (CPU, once): every config a train line names, plus the saved config of every
#    existing run an analysis line targets; built once, then only loaded.
CACHE_CONFIGS=$( { awk '$1 == "train" {print $3}' "$QUEUE"
                   awk '$1 != "train" && $1 !~ /^#/ && NF > 1 {print $2 "/config.yaml"}' "$QUEUE" |
                     while read -r c; do [ -f "$c" ] && echo "$c"; done; } | sort -u)
[ -n "$CACHE_CONFIGS" ] || { echo "no configs found in $QUEUE"; exit 4; }
# cache lock (Lustre mkdir is atomic): a second job starting at the same time waits, then finds
# the caches present instead of writing the same <cache>.pt.tmp
CLOCK="$MP/data/pe_cache/.build_lock"
mkdir -p "$MP/data/pe_cache"
got=0
for i in $(seq 1 180); do
  if mkdir "$CLOCK" 2>/dev/null; then echo "${SLURM_JOB_ID:-local}" > "$CLOCK/owner"; got=1; break; fi
  cowner=$(cat "$CLOCK/owner" 2>/dev/null)
  if [ -n "$cowner" ] && ! job_alive "$cowner"; then echo "removing stale cache lock of job $cowner"; rm -rf "$CLOCK"; continue; fi
  [ "$i" = 1 ] && echo "waiting for the cache lock held by job ${cowner:-?}"
  sleep 30
done
[ "$got" = 1 ] || { echo "could not get the cache lock in 90 min"; exit 4; }
# shellcheck disable=SC2086
if ! python -u scripts/build_caches.py $CACHE_CONFIGS > "$LOCAL/logs/build_caches.log" 2>&1; then
  rm -rf "$CLOCK"; sync_all; echo "cache build FAILED (see logs/cr_${SLURM_JOB_ID:-local}/build_caches.log)"; exit 4
fi
rm -rf "$CLOCK"
grep '^\[cache\]' "$LOCAL/logs/build_caches.log"
# shellcheck disable=SC2086
python scripts/build_caches.py --print-paths $CACHE_CONFIGS > "$LOCAL/cache_paths.txt"
stage_caches "$LOCAL/cache_paths.txt"

# 2. periodic copy local -> Lustre; stop claiming 15 min before the walltime; on TERM, copy and go
( while sleep $((SYNC_MIN * 60)); do sync_all; done ) &
SYNC_PID=$!
summary() {
  echo "=== $(date +%F_%T) status of $QUEUE"
  QUEUE="$QUEUE" STATE="$STATE" bash slurm/cr_status.sh | tail -1
  echo "--- unfinished (pending or running now; no follow-up job will run them):"
  QUEUE="$QUEUE" STATE="$STATE" bash slurm/cr_status.sh --pending
  echo "--- failed:"
  QUEUE="$QUEUE" STATE="$STATE" bash slurm/cr_status.sh --failed
}
on_usr1() { echo "USR1 $(date +%F_%T): no new tasks; syncing"; touch "$LOCAL/STOP"; sync_all; summary; }
on_term() { touch "$LOCAL/KILLED" "$LOCAL/STOP"; sync_all; }
trap on_usr1 USR1
trap on_term TERM

bg_caches
start_pool &
POOL_PID=$!
while kill -0 "$POOL_PID" 2>/dev/null; do wait "$POOL_PID"; done
kill "$SYNC_PID" 2>/dev/null
sync_all
chmod -R g+rwX "$MP/logs/cr_${SLURM_JOB_ID:-local}" "$MP/$STATE" 2>/dev/null || true   # outputs: umask 002
summary
rm -rf "$QLOCK"
echo "CR_NODE_DONE $(date +%F_%T)"

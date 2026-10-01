#!/bin/bash
# GDDL 2026 camera-ready: ONE node, 4 GPUs, drains manifests/camera_ready_q1.txt.
# HARD RULE: never more than one node, never two of these jobs at once (checked below).
# Submit from $MP:
#   cd "$MP" && sbatch --export=ALL,MP="$MP",VENV="$VENV"[,WORKERS_PER_GPU=2] slurm/cr_node.sbatch
# Resubmit (only after the previous job has ENDED) until slurm/cr_status.sh shows nothing
# pending; interrupted trains resume from their ckpt.pt. Optional env: QUEUE (default
# manifests/camera_ready_q1.txt), WORKERS_PER_GPU, BG_CACHE_CONFIGS (configs whose PE caches
# are built on the CPU in the background for a later job, e.g. configs/peptides_grit.yaml).
#SBATCH --job-name=gddl-cr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=workq
#SBATCH --time=24:00:00
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

# one-node / one-job guards
if [ "${SLURM_JOB_NUM_NODES:-1}" != "1" ]; then echo "REFUSING: ${SLURM_JOB_NUM_NODES} nodes"; exit 3; fi
others=$(squeue -h -u "$USER" -n gddl-cr,gddl-cr-calib -t RUNNING,PENDING -o %i 2>/dev/null | grep -vx "${SLURM_JOB_ID:-x}" | wc -l)
if [ "$others" -ne 0 ]; then echo "REFUSING: another gddl-cr job is running"; exit 3; fi

source slurm/cr_pool.sh
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
# shellcheck disable=SC2086
if ! python -u scripts/build_caches.py $CACHE_CONFIGS > "$LOCAL/logs/build_caches.log" 2>&1; then
  sync_all; echo "cache build FAILED (see logs/cr_${SLURM_JOB_ID:-local}/build_caches.log)"; exit 4
fi
grep '^\[cache\]' "$LOCAL/logs/build_caches.log"
# shellcheck disable=SC2086
python scripts/build_caches.py --print-paths $CACHE_CONFIGS > "$LOCAL/cache_paths.txt"
stage_caches "$LOCAL/cache_paths.txt"

# 2. periodic copy local -> Lustre; stop claiming 15 min before the walltime; on TERM, copy and go
( while sleep $((SYNC_MIN * 60)); do sync_all; done ) &
SYNC_PID=$!
on_usr1() { echo "USR1 $(date +%F_%T): no new tasks; syncing"; touch "$LOCAL/STOP"; sync_all; }
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
bash slurm/cr_status.sh | tail -1
echo "CR_NODE_DONE $(date +%F_%T)"

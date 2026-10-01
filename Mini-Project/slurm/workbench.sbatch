#!/bin/bash
# WORKBENCH node: full node (4 GPUs). Behaviour:
#  1. If manifests/CACHES_READY is absent: run the smoke payload (builds every PE cache,
#     validates every code path, evals existing ckpts). On success, touch the flag.
#  2. Start a 4-worker pool draining manifests/task_queue.txt until empty.
# While this job runs, interactive commands can be injected from the login node with:
#     srun --jobid <JOBID> --overlap --pty bash        (or non-pty single commands)
#SBATCH --job-name=gt-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=workq
#SBATCH --time=24:00:00
#SBATCH --output=logs/bench_%j.out
set -uo pipefail
umask 002

MP=/home/u6gb/alexbismuth.u6gb/AlphaTrade/experiments/miscellaneous/Graph_Transformers/Mini-Project
cd "$MP"
export PATH="$MP/../venv/bin:$PATH"
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp
export WANDB_MODE=disabled
export MPLBACKEND=Agg

nvidia-smi -L 2>/dev/null || true

if [ ! -f manifests/CACHES_READY ]; then
  echo "=== workbench: running smoke payload (caches + validation) ==="
  bash slurm/smoke_payload.sh
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "WORKBENCH: smoke payload FAILED rc=$rc — not starting workers"
    exit "$rc"
  fi
  touch manifests/CACHES_READY
  echo "=== workbench: CACHES_READY set ==="
else
  echo "=== workbench: caches already ready, skipping smoke payload ==="
fi

source slurm/worker_pool.sh
start_pool 4
chmod -R g+rwX "$MP/outputs" "$MP/logs" "$MP/manifests" "$MP/data" 2>/dev/null || true
echo "WORKBENCH_DONE"

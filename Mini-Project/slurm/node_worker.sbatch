#!/bin/bash
# Pure worker node: waits (<=90 min) for manifests/CACHES_READY, then drains the shared
# task queue with 4 workers (one per GPU) until empty.
#SBATCH --job-name=gt-worker
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=workq
#SBATCH --time=24:00:00
#SBATCH --output=logs/worker_%j.out
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

for i in $(seq 1 150); do
  [ -f manifests/CACHES_READY ] && break
  echo "waiting for CACHES_READY ($i/150 min)"
  sleep 60
done
if [ ! -f manifests/CACHES_READY ]; then
  echo "CACHES_READY never appeared — exiting (resubmit after smoke/workbench is green)"
  exit 3
fi

source slurm/worker_pool.sh
start_pool 4
chmod -R g+rwX "$MP/outputs" "$MP/logs" "$MP/manifests" 2>/dev/null || true
echo "WORKER_NODE_DONE"

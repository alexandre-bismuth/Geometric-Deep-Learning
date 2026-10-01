#!/bin/bash
# Restart pool workers on specific GPU indices of an ALREADY-RUNNING worker job, e.g.
#   srun --jobid 6038932 --overlap --ntasks=1 --gres=gpu:4 bash slurm/overlap_workers.sh 1 2
# Use when workers of an older pool exited early ("queue drained") and left GPUs idle:
# the replacements pick up the CURRENT worker_pool.sh, so they also gain the stale-claim
# reaper, the idle-poll, the not-ready guard and the #F-gated task types.
set -uo pipefail
umask 002

MP=/home/u6gb/alexbismuth.u6gb/AlphaTrade/experiments/miscellaneous/Graph_Transformers/Mini-Project
cd "$MP"
export PATH="$MP/../venv/bin:$PATH"
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp
export WANDB_MODE=disabled
export MPLBACKEND=Agg

source slurm/worker_pool.sh
echo "overlap pool on job ${SLURM_JOB_ID:-?} gpus: $* ($(date +%F_%T))"
for g in "$@"; do worker "$g" & done
wait
chmod -R g+rwX "$MP/outputs" "$MP/logs" "$MP/manifests" 2>/dev/null || true
echo "OVERLAP_POOL_DONE $(date +%F_%T)"

#!/bin/bash
# Eval-only suite over a list of run dirs (sequential in one job — each run is minutes).
#   RUNS   (required) space-separated run dirs relative to Mini-Project
#   EXTRA  (optional) extra args for run_eval_suite.py (e.g. "--randinit --seed 0" or "--max-graphs 200")
#SBATCH --job-name=gt-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=workq
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_%j.out
set -uo pipefail
umask 002

MP="${MP:-/home/u6gb/alexbismuth.u6gb/AlphaTrade/experiments/miscellaneous/Graph_Transformers/Mini-Project}"   # override with --export=ALL,MP=...
cd "$MP"
export PATH="${VENV:-$MP/../venv}/bin:$PATH"
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp
export WANDB_MODE=disabled
export MPLBACKEND=Agg

FAIL=0
for d in ${RUNS}; do
  echo "===================================================================="
  echo "[job ${SLURM_JOB_ID}] eval suite: $d  (extra: ${EXTRA:-none})"
  python -u scripts/run_eval_suite.py --run-dir "$d" ${EXTRA:-} \
    2>&1 | awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }'
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { echo "[job ${SLURM_JOB_ID}] FAIL rc=$rc for $d"; FAIL=1; }
done

chmod -R g+rwX "$MP/outputs" "$MP/logs" 2>/dev/null || true
echo "[job ${SLURM_JOB_ID}] overall fail=$FAIL"
exit "$FAIL"

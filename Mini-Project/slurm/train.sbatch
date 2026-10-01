#!/bin/bash
# Generic 1-GPU training job. Parameterised via --export:
#   CONFIG  (required) path to experiment YAML, relative to Mini-Project
#   SEED    (optional) training seed  -> output dir gets -s$SEED suffix
#   RUNNER  (optional) grit|gps       (default grit)
#   EXTRA   (optional) extra CLI overrides, e.g. "--experiment_id smoke-x --training.epochs 2"
#SBATCH --job-name=gt-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=workq
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out
set -uo pipefail
umask 002

MP="${MP:-/home/u6gb/alexbismuth.u6gb/AlphaTrade/experiments/miscellaneous/Graph_Transformers/Mini-Project}"   # override with --export=ALL,MP=...
cd "$MP" || { echo "bad MP=$MP"; exit 1; }
export PATH="${VENV:-$MP/../venv}/bin:$PATH"
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp                      # /local/user scratch-wipe mitigation
export WANDB_MODE=disabled
export MPLBACKEND=Agg

RUNNER="${RUNNER:-grit}"
SCRIPT=scripts/run_experiment.py
[ "$RUNNER" = "gps" ] && SCRIPT=scripts/run_experiment_graphgps.py

echo "[job ${SLURM_JOB_ID}] config=${CONFIG} seed=${SEED:-cfg} runner=${RUNNER} extra=${EXTRA:-none}"
nvidia-smi -L 2>/dev/null || true

python -u "$SCRIPT" --config "$CONFIG" --no-wandb \
  ${SEED:+--training.seed "$SEED"} ${EXTRA:-} \
  2>&1 | awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }'
RC=${PIPESTATUS[0]}

chmod -R g+rwX "$MP/outputs" "$MP/logs" 2>/dev/null || true
echo "[job ${SLURM_JOB_ID}] rc=$RC"
exit "$RC"

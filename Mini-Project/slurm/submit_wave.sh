#!/bin/bash
# Staggered wave submission (Isambard pacing rules: one-by-one, sleep 30 between).
# Usage: bash slurm/submit_wave.sh manifests/waveN.txt
# Manifest line format:   RUNNER CONFIG SEED WALL    (RUNNER in {grit,gps}; lines starting # skipped)
set -uo pipefail
cd "$(dirname "$0")/.."
MANIFEST="$1"
mkdir -p logs
N=0
while read -r RUNNER CONFIG SEED WALL _; do
  [ -z "${RUNNER:-}" ] && continue
  case "$RUNNER" in \#*) continue;; esac
  JID=$(sbatch --parsable --time="$WALL" \
        --export=ALL,CONFIG="$CONFIG",SEED="$SEED",RUNNER="$RUNNER" \
        slurm/train.sbatch) || { echo "SUBMIT FAILED: $RUNNER $CONFIG s$SEED — stopping wave"; exit 1; }
  N=$((N+1))
  echo "$(date +%F_%H:%M:%S) submitted $JID  $RUNNER $CONFIG seed=$SEED wall=$WALL" | tee -a logs/submissions.log
  sleep 30
done < "$MANIFEST"
echo "wave complete: $N jobs submitted from $MANIFEST"

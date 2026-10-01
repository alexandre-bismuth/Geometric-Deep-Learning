#!/bin/bash
# GDDL camera-ready: throughput calibration on ONE node (~20-30 min). Decides WORKERS_PER_GPU.
# GPU g runs (g+1) concurrent copies of a 10-layer ZINC GRIT training for $EPOCHS epochs, all
# GPUs at once; every training copy writes only to node-local disk (wiped at job end). The
# single-process cache step first builds the ZINC processed/ files and RRWP-21 cache in
# $MP/data (Lustre scratch) if missing; they are reused by the main jobs.
# Reported: wall time per copy and runs/hour/GPU = copies * 3600 / (wall * 2000/EPOCHS).
#   cd "$MP" && sbatch --export=ALL,MP="$MP",VENV="$VENV" slurm/cr_calibrate.sbatch
#SBATCH --job-name=gddl-cr-calib
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=workq
#SBATCH --time=00:45:00
#SBATCH --output=logs/gddl-cr-calib-%j.out
set -uo pipefail
: "${MP:?}"; : "${VENV:?}"; : "${EPOCHS:=15}"; : "${CONFIG:=configs/zinc_grit_dotlogit.yaml}"
cd "$MP" || exit 1
export PATH="$VENV/bin:$PATH" PYTHONUNBUFFERED=1 WANDB_MODE=disabled MPLBACKEND=Agg
python -c "import torch; assert torch.cuda.is_available(); print('cuda ok', torch.cuda.device_count())" || exit 2
if [ "${SLURM_JOB_NUM_NODES:-1}" != "1" ]; then echo "REFUSING: more than one node"; exit 3; fi
others=$(squeue -h -u "$USER" -n gddl-cr,gddl-cr-calib -t RUNNING,PENDING -o %i 2>/dev/null | grep -vx "${SLURM_JOB_ID:-x}" | wc -l)
if [ "$others" -ne 0 ]; then echo "REFUSING: another gddl-cr job is queued or running"; exit 3; fi
L="${LOCALDIR:-${TMPDIR:-/tmp}}/gddl_calib_${SLURM_JOB_ID:-local}"
mkdir -p "$L/work/outputs" "$L/work/data/pe_cache" "$L/logs"
for d in configs scripts src; do ln -s "$MP/$d" "$L/work/$d"; done
for d in ZINC LRGB; do [ -e "$MP/data/$d" ] && ln -s "$MP/data/$d" "$L/work/data/$d"; done
# single process, before any GPU work: builds the ZINC caches BOTH main jobs need (RRWP-21 for
# GRIT, RWSE-28 for GraphGPS), so the two main jobs start with them present
python scripts/build_caches.py "$CONFIG" configs/zinc_graphgps.yaml || exit 4
python scripts/build_caches.py --print-paths "$CONFIG" | while read -r p; do cp "$MP/$p" "$L/work/$p"; done
echo "TMPDIR=${TMPDIR:-} LOCALDIR=${LOCALDIR:-}"; df -h "$L"; nvidia-smi -L
cd "$L/work" || exit 1
for g in 0 1 2 3; do
  for c in $(seq 1 $((g + 1))); do
    ( s=$(date +%s)
      CUDA_VISIBLE_DEVICES=$g python -u scripts/run_experiment.py --config "$CONFIG" --no-wandb \
          --training.seed $((10 * g + c)) --training.epochs "$EPOCHS" \
          --training.ckpt_every_epochs 0 --diagnostics.enabled false \
          --logging.save_dir "$L/work/outputs/g${g}c${c}" > "$L/logs/g${g}c${c}.log" 2>&1
      echo "gpu $g copies $((g + 1)) copy $c rc=$? wall_s=$(( $(date +%s) - s ))" >> "$L/times.txt" ) &
  done
done
( while sleep 60; do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader >> "$L/util.txt"; done ) &
UPID=$!
wait $(jobs -p | grep -v "^$UPID$") 2>/dev/null
kill $UPID 2>/dev/null
sort "$L/times.txt"
echo "--- GPU utilisation samples (index, util, mem)"; sort "$L/util.txt" | uniq -c | sort -k2 | tail -20
python - "$L/times.txt" "$EPOCHS" <<'EOF'
import sys, collections
t = collections.defaultdict(list)
for line in open(sys.argv[1]):
    f = line.split(); t[int(f[3])].append(int(f[-1].split('=')[1]))
ep = int(sys.argv[2])
for k in sorted(t):
    w = max(t[k]); per_epoch = w / ep
    print(f'copies/GPU={k}: wall {w}s for {ep} epochs -> {per_epoch:.1f} s/epoch/run, '
          f'2000-epoch run ~{per_epoch * 2000 / 3600:.1f} h, throughput {k * 3600 / (per_epoch * 2000):.3f} runs/h/GPU')
EOF
grep -h "Error\|error" "$L"/logs/*.log | head -5
echo "CALIBRATION_DONE"

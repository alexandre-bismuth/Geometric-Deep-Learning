#!/bin/bash
# SMOKE (A5): proves every code path the waves will use, on one GPU:
#  0. dataset sizes + Peptides quartile train counts (feeds R9b configs)
#  1. PE precompute for all caches the waves need (zinc rrwp21, peptides rrwp17,
#     zinc rwse28, zinc lappe8, zinc rrwp21+vnodepe, zinc rwse28+vnodepe)
#     — triggered implicitly by the short trainings below
#  2. 2-epoch trainings through every new code path (GRIT grit/dot_bias/dot,
#     GPS gine/none/gatedgcn/lappe, vnode-perecomp both sides, entropy forcing, sink bias)
#  3. eval suite (v2 metrics incl. spectra/logits) on two EXISTING checkpoints + randinit
#SBATCH --job-name=gt-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=workq
#SBATCH --time=05:00:00
#SBATCH --output=logs/smoke_%j.out
set -uo pipefail
umask 002

MP=/home/u6gb/alexbismuth.u6gb/AlphaTrade/experiments/miscellaneous/Graph_Transformers/Mini-Project
cd "$MP"
export PATH="$MP/../venv/bin:$PATH"
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp
export WANDB_MODE=disabled
export MPLBACKEND=Agg

run () {  # run <label> <cmd...>
  echo "==================== SMOKE STEP: $1 ===================="
  shift
  "$@" 2>&1 | awk '{ print strftime("[%H:%M:%S]"), $0; fflush() }'
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then echo "SMOKE_FAIL step rc=$rc"; exit "$rc"; fi
}

nvidia-smi -L 2>/dev/null || true
python -c "import torch; print('cuda_available', torch.cuda.is_available(), torch.version.cuda)"

# --- 0. dataset stats (also triggers raw downloads if login prefetch missed) -------------
run "dataset stats" python -u - <<'PYEOF'
import sys; sys.path.insert(0, '.')
from torch_geometric.datasets import ZINC, LRGBDataset
z = {s: ZINC(root='data/ZINC', subset=True, split=s) for s in ('train','val','test')}
print('ZINC sizes:', {k: len(v) for k, v in z.items()})
p = {s: LRGBDataset(root='data/LRGB', name='Peptides-func', split=s) for s in ('train','val','test')}
print('Peptides sizes:', {k: len(v) for k, v in p.items()})
import numpy as np
ns = np.array([d.num_nodes for d in p['train']])
print('Peptides train node counts: min', ns.min(), 'max', ns.max(), 'mean', round(ns.mean(),1))
bands = [(0,91),(91,136),(136,204),(204,300)]
for lo,hi in bands:
    print(f'  quartile ({lo},{hi}]: train count =', int(((ns>lo)&(ns<=hi)).sum()))
nt = np.array([d.num_nodes for d in p['test']])
for lo,hi in bands:
    print(f'  quartile ({lo},{hi}]: test count =', int(((nt>lo)&(nt<=hi)).sum()))
PYEOF

# --- 2a. GRIT paths on ZINC (2 epochs each; separate smoke- ids, never touch real outputs)
run "grit-grit"    python -u scripts/run_experiment.py --config configs/zinc_grit.yaml --no-wandb --experiment_id smoke-grit --training.epochs 2 --training.warmup_epochs 1
run "grit-dotbias" python -u scripts/run_experiment.py --config configs/zinc_grit_dotbias.yaml --no-wandb --experiment_id smoke-dotbias --training.epochs 2 --training.warmup_epochs 1
run "grit-dot"     python -u scripts/run_experiment.py --config configs/zinc_grit_dot.yaml --no-wandb --experiment_id smoke-dot --training.epochs 2 --training.warmup_epochs 1
run "grit-vnode-perecomp" python -u scripts/run_experiment.py --config configs/zinc_grit_vnode_perecomp.yaml --no-wandb --experiment_id smoke-vnpe --training.epochs 2 --training.warmup_epochs 1
run "grit-forced-ent" python -u scripts/run_experiment.py --config configs/zinc_grit_forced_ent2.yaml --no-wandb --experiment_id smoke-ent --training.epochs 2 --training.warmup_epochs 1
run "grit-vnode-bias" python -u scripts/run_experiment.py --config configs/zinc_grit_vnode_bias.yaml --no-wandb --experiment_id smoke-vbias --training.epochs 2 --training.warmup_epochs 1

# --- 2b. GPS paths on ZINC --------------------------------------------------------------
run "gps-gine"     python -u scripts/run_experiment_graphgps.py --config configs/zinc_graphgps.yaml --no-wandb --experiment_id smoke-gps --training.epochs 2 --training.warmup_epochs 1
run "gps-none"     python -u scripts/run_experiment_graphgps.py --config configs/zinc_gps_none.yaml --no-wandb --experiment_id smoke-gpsnone --training.epochs 2 --training.warmup_epochs 1
run "gps-gatedgcn" python -u scripts/run_experiment_graphgps.py --config configs/zinc_gps_gatedgcn.yaml --no-wandb --experiment_id smoke-gpsggcn --training.epochs 2 --training.warmup_epochs 1
run "gps-lappe"    python -u scripts/run_experiment_graphgps.py --config configs/zinc_gps_lappe_none.yaml --no-wandb --experiment_id smoke-gpslap --training.epochs 2 --training.warmup_epochs 1
run "gps-vnode-perecomp" python -u scripts/run_experiment_graphgps.py --config configs/zinc_gps_vnode_perecomp.yaml --no-wandb --experiment_id smoke-gpsvnpe --training.epochs 2 --training.warmup_epochs 1

# --- 2c. Peptides path (triggers the big rrwp17 precompute; 1 epoch of Q1) --------------
run "pep-q1"       python -u scripts/run_experiment.py --config configs/peptides_grit_q1.yaml --no-wandb --experiment_id smoke-pep-q1 --training.epochs 1 --training.warmup_epochs 1

# --- 3. eval suite on EXISTING checkpoints (v2 metrics, capped for speed) ---------------
run "eval-zinc-grit"      python -u scripts/run_eval_suite.py --run-dir outputs/zinc-grit --max-graphs 50
run "eval-zinc-grit-rand" python -u scripts/run_eval_suite.py --run-dir outputs/zinc-grit --max-graphs 50 --randinit --seed 0
run "eval-gps"            python -u scripts/run_eval_suite.py --run-dir outputs/zinc-graphgps --max-graphs 50
run "eval-pep-q4"         python -u scripts/run_eval_suite.py --run-dir outputs/peptides-grit-Q4 --max-graphs 20

chmod -R g+rwX "$MP/outputs" "$MP/logs" "$MP/data" 2>/dev/null || true
echo "SMOKE_ALL_GREEN"

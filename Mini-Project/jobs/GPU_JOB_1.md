# GPU_JOB_1: GDDL 2026 camera-ready training programme

## R. ONE-ROUND PLAN (user, 2026-10-01). This supersedes §0 (node rule) and §2–§3 below.

- **Limits:** at most 2 nodes at once, every job ≤ 20 h, ONE round, no resubmits.
- **Scope:** P1 + P1b + P2 only: 14 trains plus their analyses. P0, P3–P8 and `BG_CACHE_CONFIGS` are dropped.
- **Job 1: calibration**
  - `slurm/cr_calibrate.sbatch`, 1 node, ≤ 45 min.
  - It also builds BOTH ZINC caches (RRWP-21 and RWSE-28), so the main jobs start with them present.
- **Decision rule.** From the calibration table, let T2 = 2000 × (s/epoch/run at copies/GPU = 2) in hours.
  - If T2 ≤ 17.5 h: `WORKERS_PER_GPU=2`, with queues `manifests/cr_nodeA.txt` and `manifests/cr_nodeB.txt`. That is 7 trains per node, all starting at t=0: 6 GRIT + 1 GraphGPS, seeds interleaved so a node failure still leaves every arm with seeds.
  - Else, if T1 = 2000 × (s/epoch at 1 copy) ≤ 17.5 h: `WORKERS_PER_GPU=1`, with `manifests/cr_nodeA_wpg1.txt` and `manifests/cr_nodeB_wpg1.txt`. That is 4 trains per node (P1 + P1b), and P2 is dropped.
  - Otherwise: stop and message me.
- **Jobs 2 and 3 (concurrently, 1 node each):**
  ```bash
  cd $MP && sbatch --export=ALL,MP=$MP,VENV=$VENV,WORKERS_PER_GPU=<k>,QUEUE=manifests/cr_nodeA<sfx>.txt slurm/cr_node.sbatch
  cd $MP && sbatch --export=ALL,MP=$MP,VENV=$VENV,WORKERS_PER_GPU=<k>,QUEUE=manifests/cr_nodeB<sfx>.txt slurm/cr_node.sbatch
  ```
  - `--time=20:00:00`, USR1 at 15 min before the end.
  - Guards refuse:
    - a 3rd gddl-cr job;
    - any queued or running gddl-cr-calib job;
    - a queue file already drained by a live job (queue lock);
    - more than 1 node per job.
  - The cache build is behind a Lustre mkdir lock, so the second job waits and then finds the caches present.
- **Analyses:** inside each node queue they are ordered eval, ablatenr, ablate, probe, perturb, cert.
  - Whatever is unfinished at the walltime, I run afterwards on CPU from the committed checkpoints.
  - Only the trains are time-critical.
- **At USR1 and at the end**, the job prints the unfinished and failed tasks explicitly. Nothing depends on a later job.
- **Pre-flight:**
  - `QUEUE=manifests/cr_nodeA.txt bash slurm/cr_status.sh | tail -1` → pending=49, and the same for B;
  - the wpg1 files → 28 each.
- **Commit (§4):** after both jobs end, with `QUEUE=<that node file>` for `--done-runs`. Also commit unfinished trains' partial dirs? No: commit only DONE runs.


From the GDDL-Camera-Ready planning session (Alexandre's laptop), for the Isambard session.
Branch `gddl-camera-ready`; this file lives at `Mini-Project/jobs/GPU_JOB_1.md`.

## 0. Rules (non-negotiable)

**Node and job limits**
- **One node, ever.** Every job is a single `--nodes=1` sbatch.
- **One job at a time.** First the calibration job (§2), then main jobs (§3) one after another.
- **Resubmit only after the previous job has ENDED.** `--dependency=afterany` only if Alexandre approves it.
- The scripts refuse to start on more than one node or next to another running `gddl-cr` job.

**Git**
- Run `git pull --rebase` only **between** jobs, never while one runs. Each task starts a fresh Python, so a pull mid-job changes the code for later tasks.
- If I need to push a code change while a job runs, I'll message first and say whether it is safe.

**What gets committed**
- Commit run outputs only (§4).
- Never commit `data/`, `logs/`, `manifests/cr1_state/`, `ckpt.pt`, `*.tmp`, or `eval_suite_cr.pkl`. They are gitignored, but check `git status` anyway.
- The answer document goes to me by SendMessage. Do not commit it.

**Code changes**
- Don't change code, configs or the queue without asking me.
- Fixing a path or environment issue is fine; tell me what you changed.

## 1. Paths and pre-flight (login node, light)

```bash
CR=/home/u6oz/alexbismuth.u6oz/gddl-camera-ready
MP=$CR/Mini-Project
VENV=$CR/venv
cd $CR && git pull --rebase origin gddl-camera-ready
git merge-base --is-ancestor 09e8c9b HEAD && echo "HEAD ok: $(git rev-parse --short HEAD)"
cd $MP && mkdir -p logs
$VENV/bin/python -c "from src.model import GritMultiHeadAttention as G; print(G.GRIT_FAMILY)"
#   expect ('grit', 'grit_dotlogit', 'grit_plusdot', 'grit_noclamp', 'grit_official')
QUEUE=manifests/camera_ready_q1_zinc.txt bash slurm/cr_status.sh | tail -1   # expect: TOTAL done=0 ... pending=343
bash slurm/cr_status.sh | tail -1                                           # full queue: pending=352
ls data/ZINC/raw data/LRGB/peptides-func/raw   # the raw files you downloaded
```

What the job scripts do (please check them against your checklist):
- **Checklist 3, 4, 7, 8.** Each task runs in `$LOCALDIR/gddl_cr_<jobid>/work`. That directory holds:
  - symlinks to `configs/`, `scripts/` and `src/`;
  - a real local `outputs/`;
  - a local `data/pe_cache` with the ZINC caches copied in once per job (~0.7 GB). The ~25 GB Peptides cache is symlinked and read from Lustre once per process.
- Checkpoints, eval files and logs are written locally. They are rsynced to `$MP/outputs/<run>` (home NFS):
  - at the end of each task;
  - every 60 min, for resumability;
  - on USR1 (15 min before the walltime) and on TERM.

  Logs go to `$MP/logs/cr_<jobid>/`.
- **Checklist 1.** Resume uses one fixed path, `outputs/<run>/ckpt.pt`. `latest_checkpoint.json` is written as a breadcrumb. Nothing is discovered by listing, and there is no `find` or `ls -R`.
- **Checklist 6.** A checkpoint every 50 epochs is ~15,600 optimiser steps on ZINC.
- **Checklist 8.** wandb is imported but disabled (`WANDB_MODE=disabled`, `--no-wandb`).
- **First command** in each job: `python -c "import torch; assert torch.cuda.is_available()"`, then `TMPDIR`/`LOCALDIR` and `df -h`.
- **Single-process prep:** `scripts/build_caches.py` processes the raw datasets and builds every PE cache the job's queue needs, before any GPU process starts. It writes to `$MP/data/pe_cache` (Lustre, a few big files), and writes are atomic (tmp + rename).
- **The first main jobs use the ZINC-only queue**, so the GPUs don't wait for the ~25 GB Peptides RRWP-17 cache. With `BG_CACHE_CONFIGS=configs/peptides_grit.yaml`, that cache is built by one niced background CPU process during the first job, ready for the later full-queue job.
  - Optional, only if your login-node rules allow ~a minute of CPU: prove the raw Peptides layout now with `python -c "from torch_geometric.datasets import LRGBDataset as D; [print(s, len(D(root='data/LRGB', name='Peptides-func', split=s))) for s in ('train','val','test')]"` from `$MP`. Expect 10873/2331/2331. This only does PyG's processing, no PE.
- **Checklist item 3.** Run dirs `$MP/outputs/<run>` are deliberately shared by *sequential* jobs, so a walltime-killed training can resume. They live on home NFS, not Lustre, and only one job ever runs. Everything a job writes while running is in its job-keyed `$LOCALDIR/gddl_cr_<jobid>` and `logs/cr_<jobid>/`.
- **Per-task state** goes to `manifests/cr1_state/<md5(line)>.{rc,attempts}` (home NFS).
  - A task killed by the walltime writes no `.rc`. The next job retries it, and a train resumes from its `ckpt.pt`. I tested this on CPU: the second job resumed at epoch 4/8, finished, and ran its eval.
  - A non-zero rc is not retried. That includes a diverged training, which now exits non-zero on a non-finite loss.

## 2. Job A: calibration (one node, at most 45 min)

```bash
cd $MP && sbatch --export=ALL,MP=$MP,VENV=$VENV slurm/cr_calibrate.sbatch
```

- GPU g runs g+1 concurrent copies of a 10-layer ZINC GRIT training for 15 epochs (GPU0 1 copy … GPU3 4 copies).
- Everything is written to `$LOCALDIR` and wiped afterwards. The ZINC caches get built on first use.
- Send me the final table it prints (`copies/GPU=k: … s/epoch/run … runs/h/GPU`), plus the GPU-utilisation samples.
- **Choose `WORKERS_PER_GPU` = the k with the highest runs/h/GPU.** On a tie within 10%, take the smaller k.
- If k=1 is best, use 1.

## 3. Main jobs (one at a time, until the queue is done or Alexandre/I say stop)

```bash
cd $CR && git pull --rebase origin gddl-camera-ready   # between jobs only
cd $MP
# first job(s): ZINC only, Peptides cache built in the background
sbatch --export=ALL,MP=$MP,VENV=$VENV,WORKERS_PER_GPU=<k>,QUEUE=manifests/camera_ready_q1_zinc.txt,BG_CACHE_CONFIGS=configs/peptides_grit.yaml slurm/cr_node.sbatch
# once all P1+P2 trains are DONE (or when I say so), and logs/cr_<jobid>/bg_caches.log of an
# earlier job ends with "bg cache build rc=0", switch to the full queue (adds P0 and P3):
sbatch --export=ALL,MP=$MP,VENV=$VENV,WORKERS_PER_GPU=<k> slurm/cr_node.sbatch
```

Both queue files contain identical lines, so a task finished under one counts as finished under the other. When using `cr_status.sh`, pass the same `QUEUE=...` as the job.

- **Monitor:** `squeue -u $USER`, `tail -n 40 logs/gddl-cr-<jobid>.out`, `bash slurm/cr_status.sh --failed`, and `bash slurm/cr_status.sh | tail -1`.
- **Queue:** `manifests/camera_ready_q1.txt`, highest priority first:

  | block | runs | analyses |
  |---|---|---|
  | **P0** | Peptides re-eval ×3 | |
  | **P1** | grit-dotlogit ×3, grit-blind ×3 | eval, ablate, ablatenr, cert, probe, perturb |
  | **P2** | grit-noclamp ×3, grit-official ×3 | same |
  | P3 | peptides-grit-dotlogit ×3 | eval |
  | P4 | seeds 3,4 of GRIT, GraphGPS, GRIT+VN | |
  | P5 | depth 20, seed 0 (GRIT, dotlogit) | |
  | P6 | plusdot ×3 | |
  | P7 | depth 20, seeds 1,2 | |
  | P8 | optional: seeds 3,4 of the rest | |

  Each block's analyses start automatically once its own training has finished.
- **Message me when:**
  - calibration is done;
  - **any P1/P2 task fails** (immediately, with the last 40 lines of its log `logs/cr_<jobid>/task_<…>.log`);
  - each main job ends (cr_status totals, failures, HEAD);
  - you see anything odd (NaN, OOM, a train much slower than the others, a cache build problem).

  A `grit-noclamp` or `grit-official` run *diverging* is a legitimate scientific outcome, not a bug. Report it (rc, epoch, last losses from `train_curve.json`) and do not retry it. Its analyses are then marked `FAILED rc=98` automatically (parent train failed), and the job carries on.

## 4. Commit procedure (after each main job ends)

```bash
cd $MP
bash slurm/cr_status.sh --done-runs                 # outputs/<run> dirs with finished tasks
bash slurm/cr_status.sh --done-runs | sed 's#^#Mini-Project/#' | (cd $CR && xargs git add --)
cd $CR && git diff --cached --name-only | grep -v '^Mini-Project/outputs/'   # must print nothing
git diff --cached --name-only | grep -c 'ckpt.pt\|\.tmp$\|_cr\.pkl$'         # must be 0
# (runs still training show up as untracked '??' dirs: leave them, they are committed when done)
git commit -m "GPU_JOB_1 outputs: job <jobid> (<n> tasks done)"
git pull --rebase origin gddl-camera-ready && git push origin gddl-camera-ready
```

Then send me the commit hash. I pull and do all the analysis myself, so there's no need to run the aggregation scripts.

## 5. Answer document (SendMessage, not committed)

1. Environment: HEAD per job, node name, torch/CUDA/PyG versions, `TMPDIR`/`LOCALDIR` and `df -h` from the job log.
2. Calibration table, and the `WORKERS_PER_GPU` you chose.
3. Per job: id, start/end time, how it ended (drained / walltime / failure), cr_status totals, failed tasks with log tails, and the commit hash.
4. Per arm trained: seconds per epoch. Take it from the tqdm rate in `logs/cr_<jobid>/task_train_*.log` (the last `it/s` value; 313 it/epoch on ZINC). Also the total wall time per run.
5. Provenance, a narrow check only:
   - does `/home/u6oz/alexbismuth.u6oz/AlphaTrade/experiments/miscellaneous/Graph_Transformers/tasks/` exist?
   - if so, list it (one level) and grep it for `0.657`, `11.69`, `2.74` and `logit_norm`.

   Do **not** search broadly.
6. Anything you changed (paths, env) and anything that surprised you.

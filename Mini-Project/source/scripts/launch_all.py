#!/usr/bin/env python3
"""Launch GRIT experiments in parallel across available GPUs.

Usage:
    python scripts/launch_all.py --quartiles     # Instance 1: 4 quartile experiments
    python scripts/launch_all.py --datasets       # Instance 2: 4 dataset experiments
    python scripts/launch_all.py --all            # All 8
    python scripts/launch_all.py --configs peptides_grit peptides_grit_q1
"""

import argparse
import subprocess
import os
import sys
import time
from pathlib import Path
from collections import deque

CONFIGS_DIR = Path("configs")

QUARTILE_CONFIGS = [
    "peptides_grit_q1",
    "peptides_grit_q2",
    "peptides_grit_q3",
    "peptides_grit_q4",
]

DATASET_CONFIGS = [
    "peptides_grit",
    "peptides_grit_vnode",
    "pascal_grit",
    "pascal_grit_vnode",
]

ALL_CONFIGS = QUARTILE_CONFIGS + DATASET_CONFIGS


def get_num_gpus():
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
            return len(lines)
    except FileNotFoundError:
        pass
    return 0


def launch_experiment(config_name, gpu_id, log_dir, extra_args=None):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        print(f"  [ERROR] Config not found: {config_path}")
        return None

    cmd = [sys.executable, "scripts/run_experiment.py", "--config", str(config_path)]
    if extra_args:
        cmd.extend(extra_args)

    log_path = os.path.join(log_dir, f"{config_name}.log")
    log_file = open(log_path, "w")

    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file


def main():
    parser = argparse.ArgumentParser(description="Launch GRIT experiments across GPUs")
    parser.add_argument("--all", action="store_true", help="Run all 8 experiments")
    parser.add_argument("--quartiles", action="store_true", help="Run 4 quartile experiments (instance 1)")
    parser.add_argument("--datasets", action="store_true", help="Run 4 dataset experiments (instance 2)")
    parser.add_argument("--configs", nargs="+", help="Specific config names to run")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--max-gpus", type=int, default=None, help="Limit GPUs")
    args = parser.parse_args()

    if args.configs:
        config_names = args.configs
    elif args.quartiles:
        config_names = QUARTILE_CONFIGS
    elif args.datasets:
        config_names = DATASET_CONFIGS
    elif args.all:
        config_names = ALL_CONFIGS
    else:
        parser.print_help()
        print("\nSpecify --quartiles (instance 1), --datasets (instance 2), --all, or --configs <names>")
        return

    num_gpus = get_num_gpus()
    if args.max_gpus:
        num_gpus = min(num_gpus, args.max_gpus)
    if num_gpus == 0:
        print("No GPUs detected. Running sequentially on CPU.")
        num_gpus = 1

    print(f"Launching {len(config_names)} experiments across {num_gpus} GPU(s)")
    print(f"Configs: {', '.join(config_names)}")

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    extra_args = ["--no-wandb"] if args.no_wandb else []

    queue = deque(config_names)
    gpu_procs = {}
    completed = []
    failed = []

    start_time = time.time()

    while queue or gpu_procs:
        for gpu_id in list(gpu_procs.keys()):
            proc, log_file, name = gpu_procs[gpu_id]
            if proc.poll() is not None:
                log_file.close()
                elapsed = time.time() - start_time
                if proc.returncode == 0:
                    print(f"  [GPU {gpu_id}] {name} DONE (exit 0, {elapsed/60:.1f}min elapsed)")
                    completed.append(name)
                else:
                    print(f"  [GPU {gpu_id}] {name} FAILED (exit {proc.returncode}). See logs/{name}.log")
                    failed.append(name)
                del gpu_procs[gpu_id]

        for gpu_id in range(num_gpus):
            if gpu_id not in gpu_procs and queue:
                config_name = queue.popleft()
                print(f"  [GPU {gpu_id}] Launching {config_name}...")
                result = launch_experiment(config_name, gpu_id, log_dir, extra_args)
                if result:
                    gpu_procs[gpu_id] = (*result, config_name)

        if gpu_procs:
            time.sleep(10)

    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    print(f"All experiments finished in {total_time:.1f} minutes")
    print(f"  Completed: {len(completed)} — {', '.join(completed)}")
    if failed:
        print(f"  Failed: {len(failed)} — {', '.join(failed)}")
    print(f"Logs in: {log_dir}/")
    print(f"Results in: outputs/")


if __name__ == "__main__":
    main()

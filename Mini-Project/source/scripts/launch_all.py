#!/usr/bin/env python3
"""Launch all experiments in parallel across available GPUs.

Usage:
    python scripts/launch_all.py --all
    python scripts/launch_all.py --zinc-only
    python scripts/launch_all.py --configs zinc_B1 zinc_A1
"""

import argparse
import subprocess
import os
import sys
import time
from pathlib import Path
from collections import deque

CONFIGS_DIR = Path("configs")

ZINC_CONFIGS = ["zinc_B1", "zinc_A1", "zinc_A2", "zinc_A3", "zinc_A4", "zinc_A5", "zinc_A6", "zinc_A7"]
PEPTIDES_CONFIGS = ["peptides_P1", "peptides_P2"]
PASCAL_CONFIGS = ["pascal_V1", "pascal_V2"]
ALL_CONFIGS = ZINC_CONFIGS + PEPTIDES_CONFIGS + PASCAL_CONFIGS


def get_num_gpus():
    """Detect number of available GPUs."""
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
            return len(lines)
    except FileNotFoundError:
        pass
    return 0


def launch_experiment(config_name, gpu_id, log_dir, extra_args=None):
    """Launch a single experiment on a specific GPU."""
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
    parser = argparse.ArgumentParser(description="Launch experiments in parallel across GPUs")
    parser.add_argument("--all", action="store_true", help="Run all 12 experiments")
    parser.add_argument("--zinc-only", action="store_true", help="Run only ZINC experiments (B1, A1-A7)")
    parser.add_argument("--peptides-only", action="store_true", help="Run only Peptides experiments")
    parser.add_argument("--pascal-only", action="store_true", help="Run only PascalVOC experiments")
    parser.add_argument("--configs", nargs="+", help="Specific config names to run")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb for all experiments")
    parser.add_argument("--max-gpus", type=int, default=None, help="Limit number of GPUs to use")
    args = parser.parse_args()

    # Determine which configs to run
    if args.configs:
        config_names = args.configs
    elif args.zinc_only:
        config_names = ZINC_CONFIGS
    elif args.peptides_only:
        config_names = PEPTIDES_CONFIGS
    elif args.pascal_only:
        config_names = PASCAL_CONFIGS
    elif args.all:
        config_names = ALL_CONFIGS
    else:
        parser.print_help()
        print("\nSpecify --all, --zinc-only, or --configs <names>")
        return

    num_gpus = get_num_gpus()
    if args.max_gpus:
        num_gpus = min(num_gpus, args.max_gpus)

    if num_gpus == 0:
        print("No GPUs detected. Running sequentially on CPU.")
        num_gpus = 1  # Use 1 "slot" for CPU

    print(f"Launching {len(config_names)} experiments across {num_gpus} GPU(s)")
    print(f"Configs: {', '.join(config_names)}")

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    extra_args = ["--no-wandb"] if args.no_wandb else []

    # Queue and GPU pool
    queue = deque(config_names)
    gpu_procs = {}  # gpu_id -> (proc, log_file, config_name)
    completed = []
    failed = []

    start_time = time.time()

    while queue or gpu_procs:
        # Check for completed processes
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

        # Launch new experiments on free GPUs
        for gpu_id in range(num_gpus):
            if gpu_id not in gpu_procs and queue:
                config_name = queue.popleft()
                print(f"  [GPU {gpu_id}] Launching {config_name}...")
                result = launch_experiment(config_name, gpu_id, log_dir, extra_args)
                if result:
                    gpu_procs[gpu_id] = (*result, config_name)

        if gpu_procs:
            time.sleep(10)  # Poll every 10 seconds

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

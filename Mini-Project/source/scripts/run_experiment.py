#!/usr/bin/env python3
"""Run a single experiment: train + evaluate + diagnose + log.

Usage:
    python scripts/run_experiment.py --config configs/zinc_B1.yaml
    python scripts/run_experiment.py --config configs/zinc_A1.yaml --training.epochs 100
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch
import yaml

# Add parent directory to path so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import InstrumentedGPS
from src.datasets import get_dataloaders, DATASET_INFO
from src.train import train_model, eval_epoch, build_criterion
from src.diagnostics import run_diagnostics, aggregate_metrics, save_diagnostics, print_summary


def deep_merge(base, override):
    """Deep merge override dict into base dict. Override takes precedence."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path, base_path=None):
    """Load experiment config, merged with base."""
    if base_path is None:
        base_path = os.path.join(os.path.dirname(config_path), 'base.yaml')

    with open(base_path, 'r') as f:
        base = yaml.safe_load(f)

    with open(config_path, 'r') as f:
        override = yaml.safe_load(f) or {}

    return deep_merge(base, override)


def apply_cli_overrides(config, overrides):
    """Apply dotted CLI overrides like --training.epochs 100."""
    for key, value in overrides.items():
        parts = key.split('.')
        d = config
        for part in parts[:-1]:
            d = d[part]
        # Try to cast to appropriate type
        old_val = d.get(parts[-1])
        if isinstance(old_val, bool):
            value = value.lower() in ('true', '1', 'yes')
        elif isinstance(old_val, int):
            value = int(value)
        elif isinstance(old_val, float):
            value = float(value)
        d[parts[-1]] = value
    return config


def main():
    parser = argparse.ArgumentParser(description='Run a single experiment')
    parser.add_argument('--config', required=True, help='Path to experiment YAML config')
    parser.add_argument('--base-config', default=None, help='Path to base YAML config')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--device', default=None, help='Device (cuda, cpu, cuda:0, etc)')

    args, unknown = parser.parse_known_args()

    # Parse dotted overrides from unknown args
    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith('--'):
                overrides[key] = unknown[i + 1]
                i += 2
            else:
                overrides[key] = 'true'
                i += 1
        else:
            i += 1

    # Load config
    config = load_config(args.config, args.base_config)
    if overrides:
        config = apply_cli_overrides(config, overrides)

    experiment_id = config['experiment_id']
    dataset_name = config['data']['dataset']
    task = config['architecture']['task']
    use_wandb = not args.no_wandb

    print(f"=" * 60)
    print(f"Experiment: {experiment_id}")
    print(f"Dataset: {dataset_name}, Task: {task}")
    print(f"VNode: {config['vnode']['enabled']}, PE: {config['pe']['type']}, "
          f"Layers: {config['architecture']['num_layers']}, MPNN: {config['architecture']['mpnn']}")
    print(f"=" * 60)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Save directory
    save_dir = os.path.join(config['logging']['save_dir'], experiment_id)
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=config['logging']['wandb_project'],
                name=experiment_id,
                config=config,
                tags=[dataset_name, f"vnode={'on' if config['vnode']['enabled'] else 'off'}",
                      f"pe={config['pe']['type']}", f"layers={config['architecture']['num_layers']}"],
            )
        except Exception as e:
            print(f"wandb init failed: {e}. Continuing without wandb.")
            use_wandb = False

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, dataset_info = get_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # Build model
    model = InstrumentedGPS(config, dataset_info).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Train
    print("\nTraining...")
    results = train_model(
        model, train_loader, val_loader, config, device, task,
        save_dir=save_dir, use_wandb=use_wandb,
    )
    print(f"\nBest val {results['metric_name']}: {results['best_val_metric']:.4f}")

    # Test evaluation
    criterion = build_criterion(task)
    test_loss, test_metric, metric_name = eval_epoch(model, test_loader, criterion, device, task)
    print(f"Test {metric_name}: {test_metric:.4f}")

    if use_wandb:
        try:
            import wandb
            wandb.log({f'test/{metric_name}': test_metric, 'test/loss': test_loss})
        except Exception:
            pass

    # Diagnostics
    if config['diagnostics']['enabled']:
        print("\nRunning diagnostics...")
        metrics = run_diagnostics(model, test_loader, device,
                                  max_graphs=config['diagnostics']['max_graphs'])

        diag_path = os.path.join(save_dir, 'diagnostics.pkl')
        save_diagnostics(metrics, diag_path)

        agg, layers = aggregate_metrics(metrics)
        print_summary(agg, layers)

        if use_wandb:
            try:
                import wandb
                # Log key diagnostic scalars
                for key in ['max_sink_score', 'overall_sink_rate', 'matrix_entropy',
                            'anisotropy', 'max_to_mean_ratio']:
                    if key in agg:
                        vals = agg[key]['mean']
                        valid = vals[~np.isnan(vals)] if hasattr(vals, '__len__') else []
                        if len(valid) > 0:
                            wandb.log({f'diag/{key}_max': float(np.nanmax(vals)),
                                       f'diag/{key}_final': float(vals[-1]) if not np.isnan(vals[-1]) else None})
            except Exception:
                pass

    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    print(f"\nExperiment {experiment_id} complete. Results saved to {save_dir}/")


if __name__ == '__main__':
    main()

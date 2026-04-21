import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def plot_layer_diagnostics(agg, layers, title_suffix='', save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Layer-wise Diagnostics{title_suffix}', fontsize=14)

    plot_configs = [
        ('max_sink_score', 'Max Sink Score', 'tab:red'),
        ('overall_sink_rate', 'Overall Sink Rate', 'tab:orange'),
        ('matrix_entropy', 'Matrix Entropy H(X)', 'tab:blue'),
        ('anisotropy', 'Anisotropy p_1', 'tab:green'),
        ('dirichlet_energy', 'Dirichlet Energy', 'tab:purple'),
        ('max_to_mean_ratio', 'Max/Mean Norm Ratio', 'tab:brown'),
    ]

    for ax, (metric_name, title, color) in zip(axes.flat, plot_configs):
        if metric_name in agg:
            mean = agg[metric_name]['mean']
            std = agg[metric_name]['std']
            valid = ~np.isnan(mean)
            valid_layers = np.array(layers)[valid]
            ax.plot(valid_layers, mean[valid], color=color, linewidth=2, marker='o', markersize=3)
            ax.fill_between(valid_layers, (mean - std)[valid], (mean + std)[valid],
                          alpha=0.2, color=color)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes, color='red')
        ax.set_xlabel('Layer')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_path}")
    plt.close()
    return fig


def plot_ablation_comparison(experiment_results, metric_name, title, save_path=None):
    ids = list(experiment_results.keys())
    values = [experiment_results[eid].get(metric_name, 0) for eid in ids]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(ids, values, color='steelblue', edgecolor='white')
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    return fig


def plot_vnode_comparison(agg_vnode, agg_no_vnode, layers, metric_name,
                          title, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    for agg, label, color in [(agg_vnode, 'With VNode', 'tab:red'),
                               (agg_no_vnode, 'Without VNode', 'tab:blue')]:
        if metric_name in agg:
            mean = agg[metric_name]['mean']
            std = agg[metric_name]['std']
            valid = ~np.isnan(mean)
            vl = np.array(layers)[valid]
            ax.plot(vl, mean[valid], color=color, linewidth=2, marker='o',
                   markersize=4, label=label)
            ax.fill_between(vl, (mean - std)[valid], (mean + std)[valid],
                          alpha=0.15, color=color)

    ax.set_xlabel('Layer')
    ax.set_ylabel(metric_name.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    return fig


def plot_spectral_gap_correlation(lambda2, sink_rates, save_path=None):
    from scipy.stats import spearmanr

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(lambda2, sink_rates, alpha=0.3, s=15, color='steelblue')

    r, p = spearmanr(lambda2, sink_rates)
    ax.set_xlabel('Spectral gap ($\\lambda_2$)')
    ax.set_ylabel('Per-graph sink rate')
    ax.set_title(f'Spectral Gap vs Sink Rate (Spearman r={r:.3f}, p={p:.2e})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    return fig


def plot_training_curves(train_losses, val_losses, title='Training Curves', save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label='Train', alpha=0.8)
    ax.plot(val_losses, label='Val', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    return fig

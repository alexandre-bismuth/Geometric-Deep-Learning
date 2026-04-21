import os
import pickle
import numpy as np
import torch
from tqdm import tqdm

from .metrics import compute_all_metrics


@torch.no_grad()
def run_diagnostics(model, loader, device, max_graphs=200):
    model.eval()
    model._register_attn_hooks()

    num_layers = model.num_layers
    all_metrics = {l: [] for l in range(num_layers + 1)}

    graphs_processed = 0

    for batch in tqdm(loader, desc='Diagnostics'):
        if graphs_processed >= max_graphs:
            break

        batch = batch.to(device)
        _ = model(batch, collect_diagnostics=True)

        batch_ids = model.layer_data[0]['batch']
        unique_graphs = batch_ids.unique()

        for g_idx_in_batch, g_id in enumerate(unique_graphs):
            if graphs_processed >= max_graphs:
                break

            graph_mask = (batch_ids == g_id)
            num_nodes_g = graph_mask.sum().item()

            for layer_idx in range(num_layers + 1):
                H_graph = model.layer_data[layer_idx]['h'][graph_mask]

                attn_g = None
                if layer_idx > 0 and model.attn_weights[layer_idx - 1] is not None:
                    attn_full = model.attn_weights[layer_idx - 1]
                    if g_idx_in_batch < attn_full.size(0):
                        attn_g = attn_full[g_idx_in_batch, :, :num_nodes_g, :num_nodes_g]

                metrics = compute_all_metrics(
                    H_graph, batch.edge_index.cpu(), num_nodes_g, attn_g
                )
                all_metrics[layer_idx].append(metrics)

            graphs_processed += 1

    model._remove_attn_hooks()
    print(f"\nProcessed {graphs_processed} graphs")

    return all_metrics


def aggregate_metrics(metrics_dict):
    layers = sorted(metrics_dict.keys())
    all_keys = set()
    for l in layers:
        for m in metrics_dict[l]:
            for k, v in m.items():
                if v is not None and not isinstance(v, (list, tuple, torch.Tensor)):
                    all_keys.add(k)
    metric_names = sorted(all_keys)

    result = {}
    for name in metric_names:
        means, stds = [], []
        for l in layers:
            values = [m[name] for m in metrics_dict[l] if m.get(name) is not None]
            if values:
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        result[name] = {'mean': np.array(means), 'std': np.array(stds)}

    return result, layers


def save_diagnostics(metrics_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(metrics_dict, f)
    print(f"Diagnostics saved to {save_path}")


def load_diagnostics(load_path):
    with open(load_path, 'rb') as f:
        return pickle.load(f)


def print_summary(agg, layers):
    print("=== Key Results ===\n")

    if 'max_sink_score' in agg:
        max_ss = np.nanmax(agg['max_sink_score']['mean'])
        max_ss_layer = layers[np.nanargmax(agg['max_sink_score']['mean'])]
        print(f"Attention sinks:")
        print(f"  Max sink score = {max_ss:.4f} at layer {max_ss_layer}")
        if max_ss > 3 * (1/23):
            print(f"  -> Some nodes receive {max_ss * 23:.1f}x more attention than uniform")
        else:
            print(f"  -> Attention is relatively diffuse")

    if 'overall_sink_rate' in agg:
        max_sr = np.nanmax(agg['overall_sink_rate']['mean'])
        print(f"  Max sink rate (eps=0.3) = {max_sr:.4f}")

    print()
    if 'max_to_mean_ratio' in agg:
        max_ratio = np.nanmax(agg['max_to_mean_ratio']['mean'])
        max_ratio_layer = layers[np.nanargmax(agg['max_to_mean_ratio']['mean'])]
        print(f"Norm concentration:")
        print(f"  Max norm^2/mean norm^2 ratio = {max_ratio:.2f} at layer {max_ratio_layer}")

    print()
    if 'matrix_entropy' in agg:
        min_ent = np.nanmin(agg['matrix_entropy']['mean'])
        min_ent_layer = layers[np.nanargmin(agg['matrix_entropy']['mean'])]
        max_aniso = np.nanmax(agg['anisotropy']['mean'])
        print(f"Compression:")
        print(f"  Min matrix entropy = {min_ent:.4f} at layer {min_ent_layer}")
        print(f"  Max anisotropy p_1 = {max_aniso:.4f}")

    print()
    if 'dirichlet_energy' in agg:
        de_first = agg['dirichlet_energy']['mean'][0]
        de_last = agg['dirichlet_energy']['mean'][-1]
        print(f"Over-smoothing:")
        print(f"  Dirichlet energy: layer 0 = {de_first:.1f}, layer {layers[-1]} = {de_last:.1f}")
        print(f"  Ratio last/first = {de_last/de_first:.4f}")

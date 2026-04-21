import torch
import numpy as np


def compute_sink_scores(attn_matrix, epsilon=0.3):
    H, N, _ = attn_matrix.shape
    per_head_sink = attn_matrix.mean(dim=1)
    sink_score_per_node = per_head_sink.mean(dim=0)
    sink_rate_per_node = (per_head_sink > epsilon).float().mean(dim=0)
    max_sink_score = sink_score_per_node.max().item()
    max_sink_node = sink_score_per_node.argmax().item()
    return {
        'sink_score_per_node': sink_score_per_node,
        'sink_rate_per_node': sink_rate_per_node,
        'max_sink_score': max_sink_score,
        'max_sink_node': max_sink_node,
        'overall_sink_rate': sink_rate_per_node.max().item(),
    }


def compute_norm_stats(H):
    norms = torch.norm(H, dim=-1)
    max_norm = norms.max().item()
    mean_norm = norms.mean().item()
    max_norm_node = norms.argmax().item()
    max_to_mean_ratio = (max_norm ** 2) / (mean_norm ** 2) if mean_norm > 1e-10 else 0.0
    return {
        'max_norm': max_norm,
        'mean_norm': mean_norm,
        'max_norm_node': max_norm_node,
        'max_to_mean_ratio': max_to_mean_ratio,
        'norm_std': norms.std().item(),
    }


def compute_matrix_entropy(H):
    S = torch.linalg.svdvals(H.float())
    S_sq = S.pow(2)
    total = S_sq.sum()
    if total < 1e-10:
        return 0.0
    p = S_sq / total
    p = p[p > 1e-10]
    entropy = -(p * torch.log(p)).sum().item()
    max_entropy = np.log(len(S))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_anisotropy(H):
    S = torch.linalg.svdvals(H.float())
    S_sq = S.pow(2)
    total = S_sq.sum()
    if total < 1e-10:
        return 1.0
    return (S_sq[0] / total).item()


def compute_dirichlet_energy(H, edge_index, num_nodes):
    src, dst = edge_index
    valid = (src < num_nodes) & (dst < num_nodes)
    src, dst = src[valid], dst[valid]
    if src.numel() == 0:
        return 0.0
    diff = H[src] - H[dst]
    return diff.pow(2).sum().item()


def compute_mixing_score(attn_matrix):
    H_heads, N, _ = attn_matrix.shape
    avg_attn = attn_matrix.mean(dim=0)
    log_attn = torch.log(avg_attn + 1e-10)
    entropy_per_node = -(avg_attn * log_attn).sum(dim=-1)
    return entropy_per_node.mean().item()


def compute_all_metrics(H, edge_index, num_nodes, attn_matrix=None):
    norm_stats = compute_norm_stats(H)
    result = {
        'matrix_entropy': compute_matrix_entropy(H),
        'anisotropy': compute_anisotropy(H),
        'dirichlet_energy': compute_dirichlet_energy(H, edge_index, num_nodes),
        'max_norm': norm_stats['max_norm'],
        'mean_norm': norm_stats['mean_norm'],
        'max_to_mean_ratio': norm_stats['max_to_mean_ratio'],
        'norm_std': norm_stats['norm_std'],
        'max_norm_node': norm_stats['max_norm_node'],
    }
    if attn_matrix is not None:
        sink_stats = compute_sink_scores(attn_matrix)
        result['max_sink_score'] = sink_stats['max_sink_score']
        result['overall_sink_rate'] = sink_stats['overall_sink_rate']
        result['max_sink_node'] = sink_stats['max_sink_node']
        result['mixing_score'] = compute_mixing_score(attn_matrix)
    return result

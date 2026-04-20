"""Diagnostic metrics for attention sink analysis in Graph Transformers.

Metrics from three papers:
- Sink score/rate (Barbero et al., 2025)
- Norm ratio, matrix entropy, anisotropy (Queipo-de-Llano et al., 2025)
- Dirichlet energy (Arroyo et al., 2025)
"""

import torch
import numpy as np


def compute_sink_scores(attn_matrix, epsilon=0.3):
    """Compute per-node sink scores from an attention matrix.

    Args:
        attn_matrix: (num_heads, seq_len, seq_len) attention weights for one graph.
                     attn_matrix[h, i, j] = how much node i attends to node j in head h.
        epsilon: threshold for sink detection (following Barbero et al.).

    Returns:
        dict with sink_score_per_node, sink_rate_per_node, max_sink_score,
        max_sink_node, overall_sink_rate
    """
    H, N, _ = attn_matrix.shape
    per_head_sink = attn_matrix.mean(dim=1)  # (H, N)
    sink_score_per_node = per_head_sink.mean(dim=0)  # (N,)
    sink_rate_per_node = (per_head_sink > epsilon).float().mean(dim=0)  # (N,)
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
    """Compute node norm statistics for a single graph.
    Args:
        H: (num_nodes, hidden_dim)
    """
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
    """Compute normalized matrix-based entropy H(X) in [0, 1]."""
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
    """Compute anisotropy p_1 = sigma_1^2 / ||X||_F^2."""
    S = torch.linalg.svdvals(H.float())
    S_sq = S.pow(2)
    total = S_sq.sum()
    if total < 1e-10:
        return 1.0
    return (S_sq[0] / total).item()


def compute_dirichlet_energy(H, edge_index, num_nodes):
    """Compute unnormalized Dirichlet energy over graph edges."""
    src, dst = edge_index
    valid = (src < num_nodes) & (dst < num_nodes)
    src, dst = src[valid], dst[valid]
    if src.numel() == 0:
        return 0.0
    diff = H[src] - H[dst]
    return diff.pow(2).sum().item()


def compute_mixing_score(attn_matrix):
    """Compute mixing score = average entropy of attention distributions.

    High mixing = diffuse attention. Low mixing = sharp/concentrated attention.
    """
    H_heads, N, _ = attn_matrix.shape
    # Average attention across heads
    avg_attn = attn_matrix.mean(dim=0)  # (N, N)
    # Shannon entropy per query node
    log_attn = torch.log(avg_attn + 1e-10)
    entropy_per_node = -(avg_attn * log_attn).sum(dim=-1)  # (N,)
    return entropy_per_node.mean().item()


def compute_all_metrics(H, edge_index, num_nodes, attn_matrix=None):
    """Compute all metrics for a single graph at a single layer."""
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

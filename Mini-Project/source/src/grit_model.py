"""GRIT (Graph Inductive Bias Transformer) implementation.

Follows Ma et al., ICML 2023: "Graph Inductive Biases in Transformers without
Message Passing." Pure Transformer architecture with RRWP positional encoding,
flexible edge-conditioned attention, and degree scalers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree


def signed_sqrt(x):
    return torch.sqrt(F.relu(x)) - torch.sqrt(F.relu(-x))


class GritAttentionHead(nn.Module):
    """Single GRIT attention head with edge-conditioned attention (Eq. 2)."""

    def __init__(self, hidden_dim, head_dim, attn_dropout=0.0):
        super().__init__()
        self.head_dim = head_dim
        self.W_Q = nn.Linear(hidden_dim, head_dim, bias=True)
        self.W_K = nn.Linear(hidden_dim, head_dim, bias=True)
        self.W_V = nn.Linear(hidden_dim, head_dim, bias=True)
        self.W_Ew = nn.Linear(hidden_dim, head_dim, bias=True)
        self.W_Eb = nn.Linear(hidden_dim, head_dim, bias=True)
        self.W_Ev = nn.Linear(head_dim, head_dim, bias=True)
        self.W_A = nn.Linear(head_dim, 1, bias=False)
        self.W_O = nn.Linear(head_dim, hidden_dim, bias=True)
        self.W_Eo = nn.Linear(head_dim, hidden_dim, bias=True)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, e, batch, mask):
        """
        Args:
            x: (N, hidden_dim) node features
            e: (N, N_max, hidden_dim) pairwise features (padded)
            batch: (N,) batch assignment
            mask: (B, N_max) boolean mask for valid nodes in the padded representation
        Returns:
            x_out: (N, hidden_dim) updated node features
            e_hat: (B, N_max, N_max, head_dim) updated pairwise features
        """
        B = batch.max().item() + 1
        N_max = mask.shape[1]

        q = self.W_Q(x)
        k = self.W_K(x)
        v = self.W_V(x)

        x_dense, node_mask = _to_dense(x, batch, N_max)
        q_dense, _ = _to_dense(q, batch, N_max)
        k_dense, _ = _to_dense(k, batch, N_max)
        v_dense, _ = _to_dense(v, batch, N_max)

        qk = q_dense.unsqueeze(2) + k_dense.unsqueeze(1)
        del q_dense, k_dense

        e_hat = F.relu(signed_sqrt(qk * self.W_Ew(e)) + self.W_Eb(e))
        del qk

        attn_logits = self.W_A(e_hat).squeeze(-1)
        attn_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        attn_logits = attn_logits.masked_fill(~attn_mask, float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=-1)
        del attn_logits
        attn_weights = attn_weights.masked_fill(~attn_mask, 0.0)
        attn_weights = self.attn_dropout(attn_weights)

        values = v_dense.unsqueeze(1) + self.W_Ev(e_hat)
        del v_dense
        x_attn = torch.matmul(attn_weights.unsqueeze(-2), values).squeeze(-2)
        del values

        x_attn_flat = x_attn[node_mask]
        x_out = self.W_O(x_attn_flat)
        e_out_dense = self.W_Eo(e_hat)
        del e_hat

        return x_out, e_out_dense, attn_weights


def _to_dense(x, batch, N_max):
    """Convert sparse node features to dense (B, N_max, D) with mask."""
    B = batch.max().item() + 1
    D = x.shape[1]
    dense = torch.zeros(B, N_max, D, device=x.device, dtype=x.dtype)
    mask = torch.zeros(B, N_max, dtype=torch.bool, device=x.device)

    _, counts = torch.unique(batch, return_counts=True)
    offset = 0
    for b in range(B):
        n = counts[b].item()
        dense[b, :n] = x[offset:offset + n]
        mask[b, :n] = True
        offset += n

    return dense, mask


class GritTransformerLayer(nn.Module):
    """One GRIT Transformer block: multi-head attention + degree scaler + FFN."""

    def __init__(self, hidden_dim, num_heads, attn_dropout=0.0, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.heads = nn.ModuleList([
            GritAttentionHead(hidden_dim, self.head_dim, attn_dropout)
            for _ in range(num_heads)
        ])

        self.deg_scaler_1 = nn.Parameter(torch.ones(hidden_dim))
        self.deg_scaler_2 = nn.Parameter(torch.zeros(hidden_dim))

        self.bn_node_attn = nn.BatchNorm1d(hidden_dim)
        self.bn_edge = nn.BatchNorm1d(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.bn_node_ffn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, e, batch, mask, deg):
        """
        Args:
            x: (N, hidden_dim)
            e: (B, N_max, N_max, hidden_dim) pairwise features
            batch: (N,)
            mask: (B, N_max) valid node mask
            deg: (N,) node degrees
        Returns:
            x_out, e_out
        """
        x_attn = torch.zeros_like(x)
        B_size, N_max = mask.shape
        e_attn = torch.zeros(B_size, N_max, N_max, self.hidden_dim, device=x.device)
        attn_all = []

        for head in self.heads:
            x_h, e_h, attn_h = head(x, e, batch, mask)
            x_attn = x_attn + x_h
            e_attn = e_attn + e_h
            attn_all.append(attn_h)
            del x_h, e_h

        log_deg = torch.log(1.0 + deg).unsqueeze(-1)
        x_attn = x_attn * self.deg_scaler_1 + log_deg * x_attn * self.deg_scaler_2

        x = x + self.bn_node_attn(x_attn)

        B, N_max = mask.shape
        e_flat = e_attn.reshape(-1, self.hidden_dim)
        e_mask_flat = (mask.unsqueeze(1) & mask.unsqueeze(2)).reshape(-1)
        valid_e = e_flat[e_mask_flat]
        if valid_e.shape[0] > 0:
            valid_e_normed = self.bn_edge(valid_e)
            e_normed = torch.zeros_like(e_flat)
            e_normed[e_mask_flat] = valid_e_normed
            e = e + e_normed.reshape(B, N_max, N_max, self.hidden_dim)
        else:
            e = e + e_attn

        x = x + self.bn_node_ffn(self.ffn(x))

        self._attn_weights = attn_all

        return x, e


class InstrumentedGRIT(nn.Module):
    """GRIT model instrumented for attention sink experiments.

    Args:
        config: dict with model, architecture, pe, data sections
        dataset_info: dict from datasets.py
    """

    def __init__(self, config, dataset_info):
        super().__init__()

        hidden_dim = config['model']['hidden_dim']
        num_heads = config['model']['num_heads']
        attn_dropout = config['model']['attn_dropout']
        dropout = config['model']['dropout']
        num_layers = config['architecture']['num_layers']
        task = config['architecture']['task']
        pe_dim = config['pe']['dim']

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.task = task
        self.pe_dim = pe_dim
        self.input_type = dataset_info['input_type']
        self.pool = config.get('model', {}).get('pool', 'mean')

        if self.input_type == 'categorical':
            num_node_types = dataset_info['num_node_types']
            num_edge_types = dataset_info['num_edge_types']
            self.node_emb = nn.Embedding(num_node_types, hidden_dim)
            self.edge_emb = nn.Embedding(num_edge_types, hidden_dim)
        else:
            node_feat_dim = dataset_info['node_feat_dim']
            edge_feat_dim = dataset_info['edge_feat_dim']
            self.node_emb = nn.Linear(node_feat_dim, hidden_dim)
            self.edge_emb = nn.Linear(edge_feat_dim, hidden_dim)

        self.pe_node_enc = nn.Linear(pe_dim, hidden_dim)
        self.pe_edge_enc = nn.Linear(pe_dim, hidden_dim)

        self.layers = nn.ModuleList([
            GritTransformerLayer(hidden_dim, num_heads, attn_dropout, dropout)
            for _ in range(num_layers)
        ])

        num_classes = dataset_info['num_classes']
        out_dim = 1 if task == 'regression' else num_classes
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

        self.layer_data = []
        self.attn_weights = []

    def forward(self, batch_data, collect_diagnostics=False):
        x = batch_data.x
        edge_index = batch_data.edge_index
        edge_attr = batch_data.edge_attr
        batch = batch_data.batch

        if self.input_type == 'categorical':
            x = self.node_emb(x.squeeze(-1))
        else:
            if x.dtype == torch.long:
                x = x.float()
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            x = self.node_emb(x)

        if hasattr(batch_data, 'rrwp_node') and batch_data.rrwp_node is not None:
            x = x + self.pe_node_enc(batch_data.rrwp_node)

        B = batch.max().item() + 1
        _, counts = torch.unique(batch, return_counts=True)
        N_max = counts.max().item()

        e_dense = torch.zeros(B, N_max, N_max, self.hidden_dim, device=x.device)

        if edge_attr is not None:
            if self.input_type == 'categorical':
                ea = self.edge_emb(edge_attr.squeeze(-1))
            else:
                if edge_attr.dtype == torch.long:
                    edge_attr = edge_attr.float()
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.unsqueeze(-1)
                ea = self.edge_emb(edge_attr)

            offset = 0
            for b_idx in range(B):
                n = counts[b_idx].item()
                graph_mask = (batch[edge_index[0]] == b_idx)
                src_local = edge_index[0, graph_mask] - offset
                dst_local = edge_index[1, graph_mask] - offset
                e_dense[b_idx, src_local, dst_local] = ea[graph_mask]
                offset += n

        if hasattr(batch_data, 'rrwp_edge') and batch_data.rrwp_edge is not None:
            rrwp_pe = self.pe_edge_enc(batch_data.rrwp_edge)
            offset = 0
            for b_idx in range(B):
                n = counts[b_idx].item()
                pe_graph = rrwp_pe[offset:offset + n * n].reshape(n, n, self.hidden_dim)
                e_dense[b_idx, :n, :n] = e_dense[b_idx, :n, :n] + pe_graph
                offset += n * n

        mask = torch.zeros(B, N_max, dtype=torch.bool, device=x.device)
        offset = 0
        for b_idx in range(B):
            n = counts[b_idx].item()
            mask[b_idx, :n] = True
            offset += n

        deg = degree(edge_index[0], num_nodes=x.size(0)).float()

        if collect_diagnostics:
            self.layer_data = [{'h': x.detach().cpu(), 'batch': batch.detach().cpu()}]
            self.attn_weights = []

        for layer_idx, layer in enumerate(self.layers):
            x, e_dense = layer(x, e_dense, batch, mask, deg)
            if collect_diagnostics:
                self.layer_data.append({'h': x.detach().cpu(), 'batch': batch.detach().cpu()})
                per_head_attn = torch.stack(layer._attn_weights, dim=1)
                self.attn_weights.append(per_head_attn.detach().cpu())

        if self.task == 'node_classification':
            return self.output_head(x)
        else:
            if self.pool == 'sum':
                graph_emb = global_add_pool(x, batch)
            else:
                graph_emb = global_mean_pool(x, batch)
            out = self.output_head(graph_emb)
            if self.task == 'regression':
                return out.squeeze(-1)
            return out

    def _register_attn_hooks(self):
        pass

    def _remove_attn_hooks(self):
        pass

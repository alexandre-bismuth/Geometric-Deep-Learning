"""GRIT (Graph Inductive Bias Transformer) implementation.

Follows Ma et al., ICML 2023: "Graph Inductive Biases in Transformers without
Message Passing." Pure Transformer architecture with RRWP positional encoding,
flexible edge-conditioned attention, and degree scalers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import degree, to_dense_batch


def signed_sqrt(x):
    return torch.sqrt(F.relu(x)) - torch.sqrt(F.relu(-x))


class GritMultiHeadAttention(nn.Module):
    """Batched multi-head GRIT attention (Eq. 2-3). Single forward pass for all heads."""

    def __init__(self, hidden_dim, num_heads, attn_dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        h = self.head_dim

        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_Ew = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_Eb = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_Ev = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_A = nn.Linear(h, 1, bias=False)
        self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_Eo = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x_dense, e, node_mask):
        """
        Args:
            x_dense: (B, N, D)
            e: (B, N, N, D)
            node_mask: (B, N)
        Returns:
            x_out: (B, N, D)
            e_out: (B, N, N, D)
            attn_weights: (B, H, N, N)
        """
        B, N, D = x_dense.shape
        H = self.num_heads
        h = self.head_dim

        # (B, N, D) -> (B, N, H, h) -> (B, H, N, h)
        q = self.W_Q(x_dense).reshape(B, N, H, h).permute(0, 2, 1, 3)
        k = self.W_K(x_dense).reshape(B, N, H, h).permute(0, 2, 1, 3)
        v = self.W_V(x_dense).reshape(B, N, H, h).permute(0, 2, 1, 3)

        # (B, H, N, 1, h) + (B, H, 1, N, h) -> (B, H, N, N, h)
        qk = q.unsqueeze(3) + k.unsqueeze(2)
        del q, k

        # Edge projections: (B, N, N, D) -> (B, N, N, H, h) -> (B, H, N, N, h)
        e_w = self.W_Ew(e).reshape(B, N, N, H, h).permute(0, 3, 1, 2, 4)
        e_b = self.W_Eb(e).reshape(B, N, N, H, h).permute(0, 3, 1, 2, 4)

        # e_hat: (B, H, N, N, h)
        e_hat = F.relu(signed_sqrt(qk * e_w) + e_b)
        del qk, e_w, e_b

        # Attention logits: (B, H, N, N, h) -> (B, H, N, N)
        attn_logits = self.W_A(e_hat).squeeze(-1)
        attn_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)  # (B, N, N)
        attn_logits = attn_logits.masked_fill(~attn_mask.unsqueeze(1), float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=-1)
        del attn_logits
        attn_weights = attn_weights.masked_fill(~attn_mask.unsqueeze(1), 0.0)
        attn_weights = self.attn_dropout(attn_weights)

        # Values: (B, H, 1, N, h) + W_Ev(e_hat) -> (B, H, N, N, h)
        e_v = self.W_Ev(e_hat)
        values = v.unsqueeze(2) + e_v
        del v, e_v

        # Weighted sum: (B, H, N, 1, N) @ (B, H, N, N, h) -> (B, H, N, h)
        x_attn = torch.matmul(attn_weights.unsqueeze(-2), values).squeeze(-2)
        del values

        # (B, H, N, h) -> (B, N, H, h) -> (B, N, D)
        x_out = self.W_O(x_attn.permute(0, 2, 1, 3).reshape(B, N, D))

        # e_hat: (B, H, N, N, h) -> (B, N, N, H, h) -> (B, N, N, D)
        e_out = self.W_Eo(e_hat.permute(0, 2, 3, 1, 4).reshape(B, N, N, D))
        del e_hat

        return x_out, e_out, attn_weights


class GritTransformerLayer(nn.Module):
    """One GRIT Transformer block: multi-head attention + degree scaler + FFN."""

    def __init__(self, hidden_dim, num_heads, attn_dropout=0.0, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.mha = GritMultiHeadAttention(hidden_dim, num_heads, attn_dropout)

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

    def forward(self, x_dense, e, node_mask, log_deg_dense):
        """
        Args:
            x_dense: (B, N, D)
            e: (B, N, N, D)
            node_mask: (B, N)
            log_deg_dense: (B, N, 1)
        """
        B, N, D = x_dense.shape

        x_attn, e_attn, attn_weights = self.mha(x_dense, e, node_mask)

        x_attn = x_attn * self.deg_scaler_1 + log_deg_dense * x_attn * self.deg_scaler_2
        x_dense = x_dense + self.bn_node_attn(x_attn.reshape(-1, D)).reshape(B, N, D)
        del x_attn

        e = e + self.bn_edge(e_attn.reshape(-1, D)).reshape(B, N, N, D)
        del e_attn

        x_ffn = self.ffn(x_dense.reshape(-1, D))
        x_dense = x_dense + self.bn_node_ffn(x_ffn).reshape(B, N, D)

        self._attn_weights = attn_weights

        return x_dense, e


class InstrumentedGRIT(nn.Module):
    """GRIT model instrumented for attention sink experiments."""

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

        x_dense, node_mask = to_dense_batch(x, batch)
        B, N_max, D = x_dense.shape

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

            src_global, dst_global = edge_index[0], edge_index[1]
            edge_batch = batch[src_global]
            _, counts = torch.unique_consecutive(batch, return_counts=True)
            offsets = torch.zeros(B, device=batch.device, dtype=torch.long)
            offsets[1:] = counts.cumsum(0)[:-1]
            src_local = src_global - offsets[edge_batch]
            dst_local = dst_global - offsets[edge_batch]
            e_dense[edge_batch, src_local, dst_local] = ea

        if hasattr(batch_data, 'rrwp_edge') and batch_data.rrwp_edge is not None:
            rrwp_pe = self.pe_edge_enc(batch_data.rrwp_edge)
            _, counts = torch.unique_consecutive(batch, return_counts=True)
            offset = 0
            for b_idx in range(B):
                n = counts[b_idx].item()
                e_dense[b_idx, :n, :n] = e_dense[b_idx, :n, :n] + rrwp_pe[offset:offset + n * n].reshape(n, n, self.hidden_dim)
                offset += n * n

        deg = degree(edge_index[0], num_nodes=x.size(0)).float()
        log_deg = torch.log(1.0 + deg)
        log_deg_dense, _ = to_dense_batch(log_deg.unsqueeze(-1), batch)

        if collect_diagnostics:
            self.layer_data = [{'h': x.detach().cpu(), 'batch': batch.detach().cpu()}]
            self.attn_weights = []

        for layer_idx, layer in enumerate(self.layers):
            x_dense, e_dense = layer(x_dense, e_dense, node_mask, log_deg_dense)
            if collect_diagnostics:
                x_flat = x_dense[node_mask]
                self.layer_data.append({'h': x_flat.detach().cpu(), 'batch': batch.detach().cpu()})
                self.attn_weights.append(layer._attn_weights.detach().cpu())

        x_flat = x_dense[node_mask]

        if self.task == 'node_classification':
            return self.output_head(x_flat)
        else:
            if self.pool == 'sum':
                graph_emb = global_add_pool(x_flat, batch)
            else:
                graph_emb = global_mean_pool(x_flat, batch)
            out = self.output_head(graph_emb)
            if self.task == 'regression':
                return out.squeeze(-1)
            return out

    def _register_attn_hooks(self):
        pass

    def _remove_attn_hooks(self):
        pass

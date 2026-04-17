"""Configurable InstrumentedGPS model for attention sink experiments.

Supports all ablation axes:
- VNode: via num_node_types / num_edge_types (28/4 without, 29/5 with)
- PE: RWSE, LapPE, or None
- MPNN: GINE or None (Transformer-only)
- Task: regression, graph_classification, multilabel_classification, node_classification
- Depth: configurable num_layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool, MessagePassing


class IdentityConv(MessagePassing):
    """No-op message passing that returns input unchanged.
    Used when mpnn='none' to create Transformer-only GPS layers.
    """
    def __init__(self, channels):
        super().__init__(aggr=None)
        self.channels = channels

    def forward(self, x, edge_index, **kwargs):
        return x


class GatedGCNLayer(MessagePassing):
    """GatedGCN layer matching the GraphGPS paper implementation.

    Messages are gated: msg = sigmoid(gate) * (A*x_j + B*e_ij)
    """
    def __init__(self, in_channels, out_channels, edge_dim=None):
        super().__init__(aggr='add')
        self.lin_src = nn.Linear(in_channels, out_channels)
        self.lin_dst = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_dim or in_channels, out_channels)
        self.lin_gate = nn.Linear(3 * out_channels, out_channels)
        self.bn_node = nn.BatchNorm1d(out_channels)
        self.bn_edge = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), x.size(1), device=x.device)

        h_src = self.lin_src(x)
        h_dst = self.lin_dst(x)
        e = self.lin_edge(edge_attr)

        # Store for message passing
        self._h_src = h_src
        self._e = e

        out = self.propagate(edge_index, x=h_dst, h_src=h_src, e=e)
        out = self.bn_node(out)
        out = self.act(out + x)  # residual
        return out

    def message(self, h_src_j, x_i, e):
        gate = torch.sigmoid(self.lin_gate(torch.cat([h_src_j, x_i, e], dim=-1)))
        return gate * h_src_j


class InstrumentedGPS(nn.Module):
    """GraphGPS model instrumented to extract attention weights and node representations.

    Args:
        config: dict with model, architecture, pe, vnode, data sections
        dataset_info: dict from datasets.py with task, num_classes, input_type, etc.
    """

    def __init__(self, config, dataset_info):
        super().__init__()

        hidden_dim = config['model']['hidden_dim']
        num_heads = config['model']['num_heads']
        attn_dropout = config['model']['attn_dropout']
        dropout = config['model']['dropout']
        num_layers = config['architecture']['num_layers']
        mpnn_type = config['architecture']['mpnn']
        task = config['architecture']['task']
        pe_type = config['pe']['type']
        pe_dim = config['pe']['dim'] if pe_type != 'none' else 0

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pe_dim = pe_dim
        self.pe_type = pe_type
        self.task = task
        self.input_type = dataset_info['input_type']

        feat_dim = hidden_dim - pe_dim

        # --- Input embedding ---
        if self.input_type == 'categorical':
            num_node_types = dataset_info['num_node_types']
            num_edge_types = dataset_info['num_edge_types']
            if config['vnode']['enabled']:
                num_node_types = config['vnode']['num_node_types']
                num_edge_types = config['vnode']['num_edge_types']
            self.node_emb = nn.Embedding(num_node_types, feat_dim)
            self.edge_emb = nn.Embedding(num_edge_types, hidden_dim)
        else:
            node_feat_dim = dataset_info['node_feat_dim']
            edge_feat_dim = dataset_info['edge_feat_dim']
            self.node_emb = nn.Linear(node_feat_dim, feat_dim)
            self.edge_emb = nn.Linear(edge_feat_dim, hidden_dim)

        # --- PE encoder ---
        if pe_dim > 0:
            self.pe_encoder = nn.Sequential(
                nn.BatchNorm1d(pe_dim),
                nn.Linear(pe_dim, pe_dim),
                nn.ReLU(),
                nn.Linear(pe_dim, pe_dim),
            )

        # --- GPS layers ---
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if mpnn_type == 'gine':
                gine_nn = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                local_model = GINEConv(gine_nn, edge_dim=hidden_dim)
            elif mpnn_type == 'gatedgcn':
                local_model = GatedGCNLayer(hidden_dim, hidden_dim, edge_dim=hidden_dim)
            else:
                local_model = IdentityConv(hidden_dim)

            gps_layer = GPSConv(
                channels=hidden_dim,
                conv=local_model,
                heads=num_heads,
                dropout=dropout,
                norm='batch_norm',
                attn_type='multihead',
                attn_kwargs={'dropout': attn_dropout},
            )
            self.layers.append(gps_layer)

        # --- Output head ---
        num_classes = dataset_info['num_classes']
        out_dim = 1 if task == 'regression' else num_classes
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

        # --- Diagnostics storage ---
        self.layer_data = []
        self.attn_weights = []
        self._attn_hooks = []

    def _register_attn_hooks(self):
        """Register forward hooks on MultiheadAttention inside each GPSConv layer."""
        self._remove_attn_hooks()
        self.attn_weights = [None] * self.num_layers

        for layer_idx, gps_layer in enumerate(self.layers):
            for name, module in gps_layer.named_modules():
                if isinstance(module, nn.MultiheadAttention):
                    # Monkey-patch forward to force weight extraction
                    original_forward = module.forward
                    def patched_forward(orig_fwd=original_forward):
                        def wrapper(*args, **kwargs):
                            kwargs['need_weights'] = True
                            kwargs['average_attn_weights'] = False
                            return orig_fwd(*args, **kwargs)
                        return wrapper
                    module.forward = patched_forward()

                    # Hook to capture attention weights
                    def make_hook(idx):
                        def hook_fn(module, args, output):
                            if isinstance(output, tuple) and len(output) == 2:
                                self.attn_weights[idx] = output[1].detach().cpu()
                        return hook_fn

                    handle = module.register_forward_hook(make_hook(layer_idx))
                    self._attn_hooks.append(handle)
                    break  # Only one MHA per GPS layer

    def _remove_attn_hooks(self):
        for h in self._attn_hooks:
            h.remove()
        self._attn_hooks = []
        self.attn_weights = []

    def forward(self, batch_data, collect_diagnostics=False):
        x = batch_data.x
        edge_index = batch_data.edge_index
        edge_attr = batch_data.edge_attr
        batch = batch_data.batch

        # --- Node embedding ---
        if self.input_type == 'categorical':
            x = self.node_emb(x.squeeze(-1))
        else:
            if x.dtype == torch.long:
                x = x.float()
            x = self.node_emb(x)

        # --- Add PE ---
        if self.pe_dim > 0:
            pe_attr = None
            if self.pe_type == 'rwse' and hasattr(batch_data, 'random_walk_pe'):
                pe_attr = batch_data.random_walk_pe
            elif self.pe_type == 'lappe' and hasattr(batch_data, 'laplacian_pe'):
                pe_attr = batch_data.laplacian_pe

            if pe_attr is not None:
                pe = self.pe_encoder(pe_attr)
                x = torch.cat([x, pe], dim=-1)
            else:
                # Pad if PE missing
                x = F.pad(x, (0, self.pe_dim))

        # --- Edge embedding ---
        if self.input_type == 'categorical':
            edge_attr = self.edge_emb(edge_attr.squeeze(-1))
        else:
            if edge_attr.dtype == torch.long:
                edge_attr = edge_attr.float()
            edge_attr = self.edge_emb(edge_attr)

        # --- Record input ---
        if collect_diagnostics:
            self.layer_data = [{'h': x.detach().cpu(), 'batch': batch.detach().cpu()}]

        # --- GPS layers ---
        for layer in self.layers:
            x = layer(x, edge_index, batch, edge_attr=edge_attr)
            if collect_diagnostics:
                self.layer_data.append({'h': x.detach().cpu(), 'batch': batch.detach().cpu()})

        # --- Output ---
        if self.task == 'node_classification':
            return self.output_head(x)
        else:
            graph_emb = global_add_pool(x, batch)
            out = self.output_head(graph_emb)
            if self.task == 'regression':
                return out.squeeze(-1)
            return out  # multilabel/graph_classification: (batch, num_classes)

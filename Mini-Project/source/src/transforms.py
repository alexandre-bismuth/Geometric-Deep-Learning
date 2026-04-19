"""Data transforms for attention sink experiments.

Key transform: AddVirtualNode — adds a virtual node connected to all real nodes,
so it participates in GPSConv's global attention (via to_dense_batch).

AddRRWP — computes Relative Random Walk Probabilities for GRIT (Ma et al., 2023).
"""

import torch
import numpy as np
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix, degree as pyg_degree


class AddVirtualNode(BaseTransform):
    """Add a virtual node connected to all real nodes in a graph.

    The VNode is appended as the LAST node (index = num_nodes).
    It gets a special node type and edge type to distinguish it from real nodes.

    IMPORTANT: Apply BEFORE PE transforms (RWSE, LapPE), so the VNode gets
    its own PE values reflecting its structural role (connected to all nodes).

    Args:
        vnode_node_type: Integer type for the virtual node (default: 28, after ZINC's 0-27).
        vnode_edge_type: Integer type for VNode edges (default: 4, after ZINC's 0-3).
    """

    def __init__(self, vnode_node_type=28, vnode_edge_type=4):
        self.vnode_node_type = vnode_node_type
        self.vnode_edge_type = vnode_edge_type

    def forward(self, data):
        num_nodes = data.num_nodes
        vnode_idx = num_nodes
        device = data.x.device if data.x is not None else 'cpu'

        # --- Pad precomputed PE tensors FIRST (before x changes num_nodes) ---
        for pe_attr in ['random_walk_pe', 'laplacian_pe']:
            if hasattr(data, pe_attr) and data[pe_attr] is not None:
                pe = data[pe_attr]
                if pe.dim() == 2 and pe.size(0) == num_nodes:
                    pad = torch.zeros(1, pe.size(1), device=pe.device, dtype=pe.dtype)
                    data[pe_attr] = torch.cat([pe, pad], dim=0)

        # --- Pad RRWP tensors for GRIT compatibility ---
        if hasattr(data, 'rrwp_node') and data.rrwp_node is not None:
            pe = data.rrwp_node
            if pe.size(0) == num_nodes:
                data.rrwp_node = torch.cat([pe, torch.zeros(1, pe.size(1), device=pe.device, dtype=pe.dtype)], dim=0)
        if hasattr(data, 'rrwp_edge') and data.rrwp_edge is not None:
            K = data.rrwp_node.size(1) if hasattr(data, 'rrwp_node') else data.rrwp_edge.size(1)
            n_new = num_nodes + 1
            old = data.rrwp_edge.reshape(num_nodes, num_nodes, K)
            new_rrwp = torch.zeros(n_new, n_new, K, device=old.device, dtype=old.dtype)
            new_rrwp[:num_nodes, :num_nodes, :] = old
            data.rrwp_edge = new_rrwp.reshape(n_new * n_new, K)

        # --- Node features ---
        if data.x is not None:
            if data.x.dtype in (torch.long, torch.int, torch.int32):
                if data.x.dim() == 1:
                    vnode_x = torch.tensor([self.vnode_node_type], device=device)
                else:
                    vnode_x = torch.full((1, data.x.size(1)), self.vnode_node_type,
                                         dtype=data.x.dtype, device=device)
            else:
                if data.x.dim() == 1:
                    vnode_x = torch.zeros(1, device=device, dtype=data.x.dtype)
                else:
                    vnode_x = torch.zeros(1, data.x.size(1), device=device, dtype=data.x.dtype)
            data.x = torch.cat([data.x, vnode_x], dim=0)

        # --- Edges: VNode <-> all real nodes (bidirectional) ---
        real_nodes = torch.arange(num_nodes, device=device)
        vnode_repeated = torch.full((num_nodes,), vnode_idx, dtype=torch.long, device=device)
        new_src = torch.cat([real_nodes, vnode_repeated])
        new_dst = torch.cat([vnode_repeated, real_nodes])
        new_edges = torch.stack([new_src, new_dst], dim=0)
        data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

        # --- Edge attributes for new edges ---
        if data.edge_attr is not None:
            num_new_edges = 2 * num_nodes
            if data.edge_attr.dim() == 1:
                new_edge_attr = torch.full((num_new_edges,), self.vnode_edge_type,
                                           dtype=data.edge_attr.dtype, device=device)
            else:
                new_edge_attr = torch.full((num_new_edges, data.edge_attr.size(1)),
                                           self.vnode_edge_type,
                                           dtype=data.edge_attr.dtype, device=device)
            data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)

        # --- Store VNode index for diagnostics ---
        data.vnode_idx = vnode_idx

        return data

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'vnode_node_type={self.vnode_node_type}, '
                f'vnode_edge_type={self.vnode_edge_type})')


class SafeLaplacianPE(BaseTransform):
    """LapPE that gracefully handles graphs with fewer nodes than k.

    If num_nodes <= k, clamps k to num_nodes - 1 and zero-pads the PE
    to the requested dimension.
    """

    def __init__(self, k=16, attr_name='laplacian_pe', is_undirected=True):
        from torch_geometric.transforms import AddLaplacianEigenvectorPE
        self.k = k
        self.attr_name = attr_name
        self.is_undirected = is_undirected
        self._base_class = AddLaplacianEigenvectorPE

    def forward(self, data):
        n = data.num_nodes
        actual_k = min(self.k, n - 1)

        if actual_k <= 0:
            data[self.attr_name] = torch.zeros(n, self.k)
            return data

        transform = self._base_class(
            k=actual_k,
            attr_name=self.attr_name,
            is_undirected=self.is_undirected,
        )
        data = transform(data)

        if actual_k < self.k:
            pe = data[self.attr_name]
            padding = torch.zeros(n, self.k - actual_k, device=pe.device, dtype=pe.dtype)
            data[self.attr_name] = torch.cat([pe, padding], dim=-1)

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(k={self.k})'


class AddRRWP(BaseTransform):
    """Compute Relative Random Walk Probabilities (Ma et al., ICML 2023).

    For each pair (i,j), computes P_{i,j} = [I, M, M^2, ..., M^{K-1}]_{i,j}
    where M = D^{-1}A is the random walk transition matrix.

    Stores:
        - data.rrwp_node: (n, K) diagonal entries (RWSE-like node PE)
        - data.rrwp_edge: (n*n, K) flattened pairwise entries for dense attention
    """

    def __init__(self, walk_length=17):
        self.walk_length = walk_length

    def forward(self, data):
        n = data.num_nodes
        K = self.walk_length
        device = data.edge_index.device

        if n == 0:
            data.rrwp_node = torch.zeros(0, K)
            data.rrwp_edge = torch.zeros(0, K)
            return data

        A = to_scipy_sparse_matrix(data.edge_index, num_nodes=n)
        deg = np.array(A.sum(axis=1)).flatten()
        deg_inv = np.zeros_like(deg)
        nonzero = deg > 0
        deg_inv[nonzero] = 1.0 / deg[nonzero]

        from scipy import sparse
        D_inv = sparse.diags(deg_inv)
        M = D_inv @ A

        rrwp_dense = np.zeros((n, n, K), dtype=np.float32)
        Mk = sparse.eye(n, format='csr')
        for k in range(K):
            Mk_dense = Mk.toarray() if sparse.issparse(Mk) else Mk
            rrwp_dense[:, :, k] = Mk_dense
            if k < K - 1:
                Mk = Mk @ M

        rrwp_node = torch.from_numpy(np.diagonal(rrwp_dense, axis1=0, axis2=1).T.copy())
        rrwp_edge = torch.from_numpy(rrwp_dense.reshape(n * n, K).copy())

        data.rrwp_node = rrwp_node.float()
        data.rrwp_edge = rrwp_edge.float()

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(walk_length={self.walk_length})'

"""Dataset loading with configurable transforms for attention sink experiments.

Supports:
- ZINC-12k (~23 nodes, graph regression)
- MNIST-SP (~75 nodes, graph classification)
- Peptides-func (~150 nodes, multilabel graph classification)
- PascalVOC-SP (~480 nodes, node classification)
"""

import os
import pickle
import numpy as np
import torch
from torch_geometric.datasets import ZINC, LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, AddRandomWalkPE
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, degree
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

from .transforms import AddVirtualNode, SafeLaplacianPE


# Dataset metadata
DATASET_INFO = {
    'zinc': {
        'task': 'regression',
        'metric': 'mae',
        'num_classes': 1,
        'num_node_types': 28,
        'num_edge_types': 4,
        'input_type': 'categorical',  # integer node types -> nn.Embedding
    },
    'peptides_func': {
        'task': 'multilabel_classification',
        'metric': 'ap',
        'num_classes': 10,
        'num_node_types': None,  # continuous features
        'num_edge_types': None,
        'input_type': 'continuous',
        'node_feat_dim': 9,
        'edge_feat_dim': 3,
    },
    'pascal_voc_sp': {
        'task': 'node_classification',
        'metric': 'f1',
        'num_classes': 21,
        'num_node_types': None,
        'num_edge_types': None,
        'input_type': 'continuous',
        'node_feat_dim': 14,
        'edge_feat_dim': 2,
    },
    'mnist_sp': {
        'task': 'graph_classification',
        'metric': 'accuracy',
        'num_classes': 10,
        'num_node_types': None,
        'num_edge_types': None,
        'input_type': 'continuous',
        'node_feat_dim': 3,    # x, y coordinates + pixel value
        'edge_feat_dim': 1,    # edge weight
    },
}


def build_transforms(config):
    """Build the transform pipeline from config.

    Order matters:
    1. AddVirtualNode (if enabled) — before PE so VNode gets its own PE
    2. PE transform (RWSE or LapPE)
    """
    transforms = []

    if config['vnode']['enabled']:
        transforms.append(AddVirtualNode(
            vnode_node_type=config['vnode']['vnode_node_type'],
            vnode_edge_type=config['vnode']['vnode_edge_type'],
        ))

    pe_type = config['pe']['type']
    pe_dim = config['pe']['dim']

    if pe_type == 'rwse' and pe_dim > 0:
        transforms.append(AddRandomWalkPE(
            walk_length=pe_dim,
            attr_name='random_walk_pe',
        ))
    elif pe_type == 'lappe' and pe_dim > 0:
        transforms.append(SafeLaplacianPE(
            k=pe_dim,
            attr_name='laplacian_pe',
            is_undirected=True,
        ))

    return Compose(transforms) if transforms else None


def get_datasets(config):
    """Load train/val/test datasets with transforms applied.

    Args:
        config: dict with keys 'data', 'vnode', 'pe'

    Returns:
        (train_dataset, val_dataset, test_dataset, info_dict)
    """
    dataset_name = config['data']['dataset']
    transform = build_transforms(config)
    data_root = config['data'].get('root', 'data')

    if dataset_name == 'zinc':
        train = ZINC(root=os.path.join(data_root, 'ZINC'), subset=True, split='train', transform=transform)
        val = ZINC(root=os.path.join(data_root, 'ZINC'), subset=True, split='val', transform=transform)
        test = ZINC(root=os.path.join(data_root, 'ZINC'), subset=True, split='test', transform=transform)

    elif dataset_name == 'peptides_func':
        train = LRGBDataset(root=os.path.join(data_root, 'LRGB'), name='Peptides-func', split='train', transform=transform)
        val = LRGBDataset(root=os.path.join(data_root, 'LRGB'), name='Peptides-func', split='val', transform=transform)
        test = LRGBDataset(root=os.path.join(data_root, 'LRGB'), name='Peptides-func', split='test', transform=transform)

    elif dataset_name == 'pascal_voc_sp':
        train = LRGBDataset(root=os.path.join(data_root, 'LRGB'), name='PascalVOC-SP', split='train', transform=transform)
        val = LRGBDataset(root=os.path.join(data_root, 'LRGB'), name='PascalVOC-SP', split='val', transform=transform)
        test = LRGBDataset(root=os.path.join(data_root, 'LRGB'), name='PascalVOC-SP', split='test', transform=transform)

    elif dataset_name == 'mnist_sp':
        from torch_geometric.datasets import GNNBenchmarkDataset
        train = GNNBenchmarkDataset(root=os.path.join(data_root, 'GNNBenchmark'), name='MNIST', split='train', transform=transform)
        val = GNNBenchmarkDataset(root=os.path.join(data_root, 'GNNBenchmark'), name='MNIST', split='val', transform=transform)
        test = GNNBenchmarkDataset(root=os.path.join(data_root, 'GNNBenchmark'), name='MNIST', split='test', transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    info = DATASET_INFO[dataset_name].copy()

    # Adjust type counts if VNode is enabled
    if config['vnode']['enabled'] and info['num_node_types'] is not None:
        info['num_node_types'] = config['vnode']['num_node_types']
        info['num_edge_types'] = config['vnode']['num_edge_types']

    return train, val, test, info


def get_dataloaders(config):
    """Load datasets and create DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader, info_dict)
    """
    train, val, test, info = get_datasets(config)

    batch_size = config['data']['batch_size']
    num_workers = config['data'].get('num_workers', 0)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, info


# --- Spectral gap computation (for Experiment 5) ---

def compute_spectral_properties(data):
    """Compute spectral properties of a single graph.

    Returns dict with lambda_2, spectral_radius, num_nodes, num_edges, avg_degree, max_degree.
    """
    n = data.num_nodes
    if n <= 1:
        return {
            'lambda_2': 0.0, 'spectral_radius': 0.0,
            'num_nodes': n, 'num_edges': data.num_edges,
            'avg_degree': 0.0, 'max_degree': 0,
        }

    edge_index, edge_weight = get_laplacian(
        data.edge_index, normalization='sym', num_nodes=n
    )
    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=n)
    deg = degree(data.edge_index[0], num_nodes=n)

    if n <= 64:
        eigenvalues = np.linalg.eigvalsh(L.toarray())
        eigenvalues = np.sort(eigenvalues)
    else:
        try:
            k = min(6, n - 1)
            eigenvalues = eigsh(L.tocsc(), k=k, sigma=1e-6, which='LM', return_eigenvectors=False)
            eigenvalues = np.sort(eigenvalues)
        except Exception:
            eigenvalues = np.linalg.eigvalsh(L.toarray())
            eigenvalues = np.sort(eigenvalues)

    nonzero_eigs = eigenvalues[eigenvalues > 1e-7]
    lambda_2 = float(nonzero_eigs[0]) if len(nonzero_eigs) > 0 else 0.0
    spectral_radius = float(1.0 - eigenvalues[0])

    return {
        'lambda_2': lambda_2,
        'spectral_radius': spectral_radius,
        'num_nodes': n,
        'num_edges': data.num_edges,
        'avg_degree': float(deg.mean().item()),
        'max_degree': int(deg.max().item()),
    }


def compute_and_cache_spectral(dataset, split_name, cache_dir='data/spectral_cache'):
    """Compute spectral properties for all graphs in a dataset split, with caching."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{split_name}_spectral.pkl')

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            props = pickle.load(f)
        print(f"  Loaded cached spectral properties for {split_name} ({len(props)} graphs)")
        return props

    print(f"  Computing spectral properties for {split_name} ({len(dataset)} graphs)...")
    props = []
    for data in tqdm(dataset, desc=f'  {split_name}'):
        props.append(compute_spectral_properties(data))

    with open(cache_path, 'wb') as f:
        pickle.dump(props, f)
    print(f"  Saved to {cache_path}")
    return props

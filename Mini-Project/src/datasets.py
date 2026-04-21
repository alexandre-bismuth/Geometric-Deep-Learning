import os
import pickle
import numpy as np
import torch
from torch_geometric.datasets import ZINC, LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, degree
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

from .transforms import AddVirtualNode, AddRRWP

DATASET_INFO = {
    'zinc': {
        'task': 'regression',
        'metric': 'mae',
        'num_classes': 1,
        'num_node_types': 28,
        'num_edge_types': 4,
        'input_type': 'categorical',
    },
    'peptides_func': {
        'task': 'multilabel_classification',
        'metric': 'ap',
        'num_classes': 10,
        'num_node_types': None,
        'num_edge_types': None,
        'input_type': 'continuous',
        'node_feat_dim': 9,
        'edge_feat_dim': 3,
    }
}


def _build_pe_transform(config):
    pe_type = config['pe']['type']
    pe_dim = config['pe']['dim']

    if pe_type == 'rrwp' and pe_dim > 0:
        return AddRRWP(walk_length=pe_dim)
    return None


def _build_vnode_transform(config):
    vnode_cfg = config.get('vnode', {})
    if vnode_cfg.get('enabled', False):
        return AddVirtualNode(
            vnode_node_type=vnode_cfg['vnode_node_type'],
            vnode_edge_type=vnode_cfg['vnode_edge_type'],
        )
    return None


def _pe_cache_path(data_root, dataset_name, split, pe_type, pe_dim):
    cache_dir = os.path.join(data_root, 'pe_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'{dataset_name}_{split}_{pe_type}{pe_dim}.pt')


def _load_raw_dataset(dataset_name, data_root, split):
    if dataset_name == 'zinc':
        return ZINC(root=os.path.join(data_root, 'ZINC'), subset=True, split=split)
    elif dataset_name == 'peptides_func':
        return LRGBDataset(root=os.path.join(data_root, 'LRGB'), name='Peptides-func', split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _precompute_pe(dataset_name, data_root, split, pe_transform, pe_type, pe_dim):
    cache_path = _pe_cache_path(data_root, dataset_name, split, pe_type, pe_dim)

    if os.path.exists(cache_path):
        print(f"  Loading cached PE for {dataset_name}/{split} from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    print(f"  Precomputing {pe_type}(dim={pe_dim}) for {dataset_name}/{split}...")
    raw_ds = _load_raw_dataset(dataset_name, data_root, split)

    cached_data = []
    for i in tqdm(range(len(raw_ds)), desc=f'  {split}'):
        data = raw_ds[i].clone()
        if pe_transform is not None:
            data = pe_transform(data)
        cached_data.append(data)

    torch.save(cached_data, cache_path)
    print(f"  Saved to {cache_path}")
    return cached_data


class PrecomputedDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx].clone()
        if self.transform is not None:
            data = self.transform(data)
        return data


def get_datasets(config):
    dataset_name = config['data']['dataset']
    data_root = config['data'].get('root', 'data')
    pe_type = config['pe']['type']
    pe_dim = config['pe']['dim']

    pe_transform = _build_pe_transform(config)
    vnode_transform = _build_vnode_transform(config)

    datasets = {}
    for split in ['train', 'val', 'test']:
        if pe_transform is not None:
            data_list = _precompute_pe(dataset_name, data_root, split, pe_transform, pe_type, pe_dim)
        else:
            raw_ds = _load_raw_dataset(dataset_name, data_root, split)
            data_list = [raw_ds[i] for i in range(len(raw_ds))]

        node_filter = config['data'].get('node_count_filter', None)
        if node_filter is not None:
            lo = node_filter.get('min_nodes', 0)
            hi = node_filter.get('max_nodes', float('inf'))
            before = len(data_list)
            data_list = [d for d in data_list if lo < d.num_nodes <= hi]
            print(f"  Filtered {split}: {before} -> {len(data_list)} graphs "
                  f"(nodes in ({lo}, {hi}])")

        datasets[split] = PrecomputedDataset(data_list, transform=vnode_transform)

    info = DATASET_INFO[dataset_name].copy()
    vnode_cfg = config.get('vnode', {})
    if vnode_cfg.get('enabled', False) and info.get('num_node_types') is not None:
        info['num_node_types'] = info['num_node_types'] + 1
        info['num_edge_types'] = info['num_edge_types'] + 1

    return datasets['train'], datasets['val'], datasets['test'], info


def get_dataloaders(config):
    train, val, test, info = get_datasets(config)

    batch_size = config['data']['batch_size']
    num_workers = config['data'].get('num_workers', 0)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, info


def compute_spectral_properties(data):
    n = data.num_nodes
    if n <= 1:
        return {
            'lambda_2': 0.0, 'spectral_radius': 0.0,
            'num_nodes': n, 'num_edges': data.num_edges,
            'avg_degree': 0.0, 'max_degree': 0,
        }

    edge_index, edge_weight = get_laplacian(data.edge_index, normalization='sym', num_nodes=n)
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

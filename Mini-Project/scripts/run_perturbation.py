"""R4: single-node input perturbation -> per-layer propagation vs graph distance.

Fixes vs the old notebook: (i) categorical resampling excludes the original type
(no ~1/28 silent no-ops); (ii) with a virtual node, BFS distances are computed on the
REAL-node subgraph (the vnode makes everything distance <=2), and the vnode is never
the perturbed node.

Usage: python scripts/run_perturbation.py --run-dir outputs/zinc-grit [--n-graphs 100]
"""
import argparse
import json
import os
import sys
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.seed import set_seed
from scripts.run_eval_suite import load_run_config, detect_arch, build_everything


def bfs_dist(n, edge_index, src, exclude_node=None):
    adj = [[] for _ in range(n)]
    for a, b in edge_index.t().tolist():
        if exclude_node is not None and (a == exclude_node or b == exclude_node):
            continue
        adj[a].append(b)
    dist = [-1] * n
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


@torch.no_grad()
def layer_states(model, data, device):
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([data]).to(device)
    model(batch, collect_diagnostics=True)
    return [ld['h'].clone() for ld in model.layer_data]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--ckpt', default='best_model.pt')
    p.add_argument('--n-graphs', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_run_config(args.run_dir)
    arch = detect_arch(config)
    has_vnode = config.get('vnode', {}).get('enabled', False)
    categorical = config['data']['dataset'] == 'zinc'
    set_seed(args.seed)

    model, test_loader, _, _ = build_everything(config, arch, device)
    state = torch.load(os.path.join(args.run_dir, args.ckpt), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model._register_attn_hooks()

    dataset = test_loader.dataset
    rng = np.random.RandomState(args.seed)
    n_graphs = min(args.n_graphs, len(dataset))

    buckets = ['d0', 'd1', 'd2', 'd3plus']
    num_layers = model.num_layers
    acc = {l: {b: [] for b in buckets} for l in range(num_layers + 1)}

    for gi in range(n_graphs):
        data = dataset[gi]
        n = data.num_nodes
        n_real = n - 1 if has_vnode else n
        if n_real < 4:
            continue
        u = int(rng.randint(0, n_real))          # never the vnode

        clean = layer_states(model, data, device)

        pert = data.clone()
        if categorical:
            old = int(pert.x[u].item()) if pert.x.dim() == 1 else int(pert.x[u, 0].item())
            choices = [t for t in range(28) if t != old]
            new = int(rng.choice(choices))
            if pert.x.dim() == 1:
                pert.x[u] = new
            else:
                pert.x[u, 0] = new
        else:
            std = pert.x.float().std().item() + 1e-8
            noise = torch.from_numpy(rng.randn(pert.x.size(1)).astype('float32')) * std
            pert.x = pert.x.clone().float()
            pert.x[u] += noise

        dirty = layer_states(model, pert, device)
        dist = bfs_dist(n, data.edge_index, u,
                        exclude_node=(n - 1) if has_vnode else None)

        for l in range(num_layers + 1):
            delta = (dirty[l] - clean[l]).norm(dim=-1)     # (n,)
            d0 = delta[u].item() + 1e-12
            for b in buckets:
                if b == 'd0':
                    vals = [delta[u].item()]
                else:
                    lo = {'d1': 1, 'd2': 2, 'd3plus': 3}[b]
                    hi = {'d1': 1, 'd2': 2, 'd3plus': 10 ** 9}[b]
                    idx = [i for i in range(n_real)
                           if i != u and 0 <= dist[i] and lo <= dist[i] <= hi]
                    if not idx:
                        continue
                    vals = [delta[idx].mean().item()]
                acc[l][b].append(vals[0] / d0 if b != 'd0' else vals[0])

    model._remove_attn_hooks()
    out = {'n_graphs': n_graphs, 'buckets': {}}
    for l in range(num_layers + 1):
        out['buckets'][l] = {b: {'mean': float(np.mean(v)) if v else None,
                                 'std': float(np.std(v)) if v else None,
                                 'n': len(v)}
                             for b, v in acc[l].items()}
    path = os.path.join(args.run_dir, f'perturbation_s{args.seed}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"saved {path}")
    fl = out['buckets'][num_layers]
    print('final layer relative propagation:',
          {b: (round(fl[b]['mean'], 4) if fl[b]['mean'] is not None else None) for b in buckets})


if __name__ == '__main__':
    main()

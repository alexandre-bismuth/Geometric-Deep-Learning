"""Camera-ready offline analysis (GDDL 2026). Reads only committed run outputs
(eval_summary.json, eval_suite.pkl, metrics.json, sink_ablation_s0.json,
theory_certificate.json); needs numpy, no GPU.

Sections:
  A  headline table, seeded runs only vs. with the legacy unseeded runs
  B  epsilon sweep and threshold-free peak share n * max_j s_j
  C  over-mixing proxies by layer (anisotropy, Dirichlet Rayleigh quotient)
  D  sink-node value norm relative to the graph mean (no-op test)
  E  per-seed sink ablation (sink / none, sink / random)
  F  virtual-node rank-1 frequency per seed
  G  sink rate vs. test score across arms (rank correlation)
  H  size-binned sink rate inside the full-data Peptides models
  I  logit-norm slope per arm
  J  no-op test conditioned on who the sink is (VN vs real node; sink present or not)

Usage: python scripts/camera_ready_analysis.py [--outputs outputs] [-o report.md]
"""
import argparse
import glob
import json
import math
import os
import pickle
import re
from collections import defaultdict

import numpy as np

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228}
def is_legacy(run):
    """Unseeded runs (no -s<seed> suffix): the April zinc-grit / zinc-grit-vnode / zinc-graphgps
    and the unmatched peptides-grit-Q1..Q4 quartiles. Excluded from every camera-ready number."""
    return re.search(r'-s\d+$', run) is None


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def ci(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return '--'
    if len(v) == 1:
        return f'{v[0]:.3f} (n=1)'
    h = T95.get(len(v) - 1, 1.96) * v.std(ddof=1) / math.sqrt(len(v))
    return f'{v.mean():.3f} ± {h:.3f} (n={len(v)})'


def nanmax(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return max(xs) if xs else None


def arm_of(run):
    return re.sub(r'-s\d+$', '', run)


def peak_layer(agg, key='overall_sink_rate'):
    m = agg.get(key, {}).get('mean', [])
    vals = [(v, i) for i, v in enumerate(m) if v is not None and np.isfinite(v)]
    return max(vals)[1] if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', default='outputs')
    ap.add_argument('-o', '--out', default=None)
    args = ap.parse_args()

    runs = {}
    for d in sorted(glob.glob(os.path.join(args.outputs, '*'))):
        name = os.path.basename(d)
        if name.startswith(('smoke-', 'randinit-')) or not os.path.isdir(d):
            continue
        e = load(os.path.join(d, 'eval_summary.json'))
        m = load(os.path.join(d, 'metrics.json'))
        if e is None:
            continue
        runs[name] = {'dir': d, 'agg': e['agg'], 'metrics': m or {},
                      'abl': load(os.path.join(d, 'sink_ablation_s0.json')),
                      'cert': load(os.path.join(d, 'theory_certificate.json'))}
    arms = defaultdict(list)
    for r in runs:
        arms[arm_of(r)].append(r)

    L = ['# Camera-ready offline analysis\n']

    def score(r):
        m = runs[r]['metrics']
        return m.get('test_mae', m.get('test_ap'))

    # A
    L.append('## A. Headline: seeded runs only vs. including legacy unseeded runs\n')
    L.append('| arm | sink (all) | sink (seeded only) | score (all) | score (seeded only) |')
    L.append('|---|---|---|---|---|')
    for a in sorted(arms):
        rs = arms[a]
        seeded = [r for r in rs if not is_legacy(r)]
        f = lambda rr: [nanmax(runs[r]['agg']['overall_sink_rate']['mean']) for r in rr]
        g = lambda rr: [score(r) for r in rr]
        L.append(f'| `{a}` | {ci(f(rs))} | {ci(f(seeded))} | {ci(g(rs))} | {ci(g(seeded))} |')

    # B
    L.append('\n## B. Epsilon sweep (max over layers of mean sink_rate_eps) and peak share n·max s_j\n')
    eps = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.8']
    L.append('| arm | ' + ' | '.join(f'ε={e}' for e in eps) + ' | n·max s_j (peak layer) |')
    L.append('|---|' + '---|' * (len(eps) + 1))
    for a in sorted(arms):
        rs = [r for r in arms[a] if not is_legacy(r)]
        cells = []
        for e in eps:
            vals = [nanmax(runs[r]['agg'].get(f'sink_rate_eps{e}', {}).get('mean', [])) for r in rs]
            v = [x for x in vals if x is not None]
            cells.append(f'{np.mean(v):.3f}' if v else '--')
        ns = [nanmax(runs[r]['agg'].get('norm_sink_score', {}).get('mean', [])) for r in rs]
        L.append(f'| `{a}` | ' + ' | '.join(cells) + f' | {ci(ns)} |')

    # C
    L.append('\n## C. Over-mixing proxies (seeded runs). Anisotropy p1 = σ1²/‖X‖² → 1 means '
             'token collapse; Dirichlet Rayleigh = Σ_edges‖x_i−x_j‖²/‖X‖² → 0 means smoothing.\n')
    L.append('| arm | anisotropy L0 | anisotropy last | anisotropy max | Rayleigh L0 | Rayleigh last | Rayleigh last/L0 |')
    L.append('|---|---|---|---|---|---|---|')
    for a in sorted(arms):
        rs = [r for r in arms[a] if not is_legacy(r)]
        an0 = [runs[r]['agg']['anisotropy']['mean'][0] for r in rs]
        anL = [runs[r]['agg']['anisotropy']['mean'][-1] for r in rs]
        anM = [nanmax(runs[r]['agg']['anisotropy']['mean']) for r in rs]
        ra0 = [runs[r]['agg']['dirichlet_rayleigh']['mean'][0] for r in rs]
        raL = [runs[r]['agg']['dirichlet_rayleigh']['mean'][-1] for r in rs]
        rat = [l / f if f else None for f, l in zip(ra0, raL)]
        L.append(f'| `{a}` | {ci(an0)} | {ci(anL)} | {ci(anM)} | {ci(ra0)} | {ci(raL)} | {ci(rat)} |')

    # D
    L.append('\n## D. Sink-node value norm / graph-mean value norm at the peak-sinking layer '
             '(≪1 would be an LLM-style no-op sink)\n')
    L.append('| arm | sink_value_ratio @ peak layer | min_value_ratio @ peak layer | peak layer(s) |')
    L.append('|---|---|---|---|')
    for a in sorted(arms):
        rs = [r for r in arms[a] if not is_legacy(r)]
        sv, mv, pl = [], [], []
        for r in rs:
            agg = runs[r]['agg']
            p = peak_layer(agg)
            if p is None or 'sink_value_ratio' not in agg:
                continue
            sv.append(agg['sink_value_ratio']['mean'][p])
            mv.append(agg['min_value_ratio']['mean'][p])
            pl.append(p)
        L.append(f'| `{a}` | {ci(sv)} | {ci(mv)} | {pl} |')

    # E
    L.append('\n## E. Sink ablation per seed (500 test graphs; column zeroed + renormalised in every layer)\n')
    L.append('| arm | per-seed sink/none | per-seed sink/random | mean sink/random |')
    L.append('|---|---|---|---|')
    for a in sorted(arms):
        sn, sr = [], []
        for r in arms[a]:
            ab = runs[r]['abl']
            if not ab:
                continue
            c = ab['conditions']
            sn.append(c['sink']['mae'] / c['none']['mae'])
            sr.append(c['sink']['mae'] / c['random']['mae'])
        if sn:
            L.append(f'| `{a}` | {[round(x, 2) for x in sn]} | {[round(x, 2) for x in sr]} | {ci(sr)} |')

    # F
    L.append('\n## F. Virtual node ranked first (vnode_is_sink), per seed, max over layers\n')
    L.append('| run | vnode_is_sink max over layers | layer | vnode_recv_score at that layer |')
    L.append('|---|---|---|---|')
    for r in sorted(runs):
        agg = runs[r]['agg']
        if 'vnode_is_sink' not in agg:
            continue
        m = agg['vnode_is_sink']['mean']
        vals = [(v, i) for i, v in enumerate(m) if v is not None and np.isfinite(v)]
        if not vals:
            continue
        v, i = max(vals)
        rec = agg.get('vnode_recv_score', {}).get('mean', [None] * len(m))[i]
        L.append(f'| `{r}` | {v:.3f} | {i} | {rec:.3f} |')

    # G
    L.append('\n## G. Sink rate vs. test MAE across ZINC arms (arm means, seeded runs)\n')
    xs, ys, names = [], [], []
    for a in sorted(arms):
        if not a.startswith('zinc'):
            continue
        rs = [r for r in arms[a] if not is_legacy(r)]
        s = [nanmax(runs[r]['agg']['overall_sink_rate']['mean']) for r in rs]
        m = [score(r) for r in rs]
        if s and m and all(v is not None for v in m):
            xs.append(np.mean(s)); ys.append(np.mean(m)); names.append(a)
    if len(xs) > 3:
        rx = np.argsort(np.argsort(xs)); ry = np.argsort(np.argsort(ys))
        rho = np.corrcoef(rx, ry)[0, 1]
        L.append(f'Spearman ρ(sink rate, MAE) over {len(xs)} ZINC arms = {rho:.2f}\n')
        for n_, x, y in sorted(zip(names, xs, ys), key=lambda t: t[1]):
            L.append(f'- `{n_}`: sink {x:.3f}, MAE {y:.3f}')

    # H
    L.append('\n## H. Size-binned sink rate within full-data Peptides models (one model, all sizes)\n')
    bins = [(0, 91), (91, 136), (136, 204), (204, 300)]
    L.append('| run | ' + ' | '.join(f'({lo},{hi}]' for lo, hi in bins) + ' |')
    L.append('|---|' + '---|' * len(bins))
    per_bin = {b: {'sr': [], 'ms': [], 'ns': []} for b in bins}
    for r in sorted(runs):
        if not re.fullmatch(r'peptides-grit-s\d+', r):
            continue
        with open(os.path.join(runs[r]['dir'], 'eval_suite.pkl'), 'rb') as f:
            suite = pickle.load(f)
        res = suite['results']
        layers = [l for l in sorted(res) if l > 0]
        cells = []
        for b in bins:
            lo, hi = b
            idx = [i for i, m in enumerate(res[layers[0]]) if lo < m['num_nodes'] <= hi]
            if not idx:
                cells.append('--'); continue
            sr = max(np.mean([res[l][i]['sink_rate_eps0.3'] for i in idx]) for l in layers)
            ms = max(np.mean([res[l][i]['max_sink_score'] for i in idx]) for l in layers)
            ns = max(np.mean([res[l][i]['norm_sink_score'] for i in idx]) for l in layers)
            per_bin[b]['sr'].append(sr); per_bin[b]['ms'].append(ms); per_bin[b]['ns'].append(ns)
            cells.append(f'SR {sr:.3f} / max s {ms:.3f} / n·s {ns:.2f} (G={len(idx)})')
        L.append(f'| `{r}` | ' + ' | '.join(cells) + ' |')
    L.append('| **mean ± CI** | ' + ' | '.join(
        f"SR {ci(per_bin[b]['sr'])}; max s {ci(per_bin[b]['ms'])}" for b in bins) + ' |')

    # I
    L.append('\n## I. Logit-norm slope (log|logit| vs log‖h‖), max-|.| layer and peak-sink layer\n')
    L.append('| arm | slope @ peak-sink layer | attn_norm_spearman @ peak-sink layer |')
    L.append('|---|---|---|')
    for a in sorted(arms):
        rs = [r for r in arms[a] if not is_legacy(r)]
        sl, sp = [], []
        for r in rs:
            agg = runs[r]['agg']
            p = peak_layer(agg)
            if p is None or 'logit_norm_slope' not in agg:
                continue
            sl.append(agg['logit_norm_slope']['mean'][p])
            sp.append(agg['attn_norm_spearman']['mean'][p])
        L.append(f'| `{a}` | {ci(sl)} | {ci(sp)} |')

    # J
    L.append('\n## J. No-op test conditioned on WHO the sink is (per-graph, from eval_suite.pkl)\n')
    L.append('Value ratio = ‖v‖ of the argmax-received node / mean ‖v‖ over the graph (head-mean). '
             '"sink present" = at least one head with s_j > 0.3 in that graph and layer. '
             'Medians over graphs; arm rows are mean ± CI over seeds of the per-seed medians.\n')
    L.append('### J1. Virtual-node arms: VN-argmax graphs vs real-node-argmax graphs, per layer\n')
    L.append('| run | layer | P(VN argmax) | ratio, VN argmax (med) | ratio, real-node argmax (med) '
             '| P(VN argmax ∧ sink present) | ratio, VN argmax ∧ sink present (med) |')
    L.append('|---|---|---|---|---|---|---|')
    j2 = defaultdict(lambda: {'vn': [], 'real': [], 'p': []})
    for r in sorted(runs):
        if is_legacy(r) or not r.startswith('zinc') or 'vnode' not in r:
            continue
        pk = os.path.join(runs[r]['dir'], 'eval_suite.pkl')
        if not os.path.exists(pk):
            continue
        with open(pk, 'rb') as f:
            res = pickle.load(f)['results']
        pl = peak_layer(runs[r]['agg'])
        for l in sorted(k for k in res if k > 0):
            g = [m for m in res[l] if 'vnode_is_sink' in m and 'sink_value_ratio' in m]
            if not g:
                continue
            vn = [m['sink_value_ratio'] for m in g if m['vnode_is_sink'] == 1.0]
            real = [m['sink_value_ratio'] for m in g if m['vnode_is_sink'] != 1.0]
            vns = [m['sink_value_ratio'] for m in g
                   if m['vnode_is_sink'] == 1.0 and m['overall_sink_rate'] > 0]
            med = lambda x: f'{np.median(x):.3f} (G={len(x)})' if x else '--'
            star = ' *' if l == pl else ''
            L.append(f'| `{r}` | {l}{star} | {len(vn) / len(g):.3f} | {med(vn)} | {med(real)} '
                     f'| {len(vns) / len(g):.3f} | {med(vns)} |')
            if l == pl:
                j2[arm_of(r)]['p'].append(len(vn) / len(g))
                if vn:
                    j2[arm_of(r)]['vn'].append(float(np.median(vn)))
                if real:
                    j2[arm_of(r)]['real'].append(float(np.median(real)))
    L.append('\n(* = peak-sinking layer of that run; agg index == layer index, layer 0 is the input.)\n')
    L.append('| arm (peak layer) | P(VN argmax) | ratio, VN argmax | ratio, real-node argmax |')
    L.append('|---|---|---|---|')
    for a in sorted(j2):
        L.append(f"| `{a}` | {ci(j2[a]['p'])} | {ci(j2[a]['vn'])} | {ci(j2[a]['real'])} |")

    L.append('\n### J2. All ZINC arms: value ratio of the argmax node at the peak layer, '
             'graphs with a sink present only\n')
    L.append('| arm | P(sink present) | ratio, sink present (med) | ratio, no sink (med) |')
    L.append('|---|---|---|---|')
    for a in sorted(arms):
        if not a.startswith('zinc'):
            continue
        ps, rp, rn = [], [], []
        for r in arms[a]:
            if is_legacy(r):
                continue
            pk = os.path.join(runs[r]['dir'], 'eval_suite.pkl')
            pl = peak_layer(runs[r]['agg'])
            if pl is None or not os.path.exists(pk):
                continue
            with open(pk, 'rb') as f:
                g = [m for m in pickle.load(f)['results'][pl] if 'sink_value_ratio' in m]
            if not g:
                continue
            yes = [m['sink_value_ratio'] for m in g if m['overall_sink_rate'] > 0]
            no = [m['sink_value_ratio'] for m in g if m['overall_sink_rate'] == 0]
            ps.append(len(yes) / len(g))
            if yes:
                rp.append(float(np.median(yes)))
            if no:
                rn.append(float(np.median(no)))
        if ps:
            L.append(f'| `{a}` | {ci(ps)} | {ci(rp)} | {ci(rn)} |')

    md = '\n'.join(L) + '\n'
    print(md)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(md)


if __name__ == '__main__':
    main()

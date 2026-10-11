"""Build the public code release (github.com/alexandre-bismuth/graph-transformer-sinks).

Copies, from this Mini-Project directory into --dest, only what the GDDL 2026 paper needs:
source, the scripts that produce or analyse its results, the configs of the arms it
reports, and the small result files from which every number regenerates
(scripts/paper_numbers.py). Checkpoints and per-graph pickles are left out, except the
few that paper_numbers.py reads (GRIT checkpoints for the score-spread bound, the full
Peptides-func eval pickles for the size bins, the GraphGPS + virtual node pickles for the
sink's value norm). README, LICENSE, requirements.txt and .gitignore come from release_files/.

Usage: python scripts/make_release.py --dest ../../release/graph-transformer-sinks
"""
import argparse
import glob
import os
import re
import shutil

# metrics_v2/diagnostics_v2 only: the v1 modules computed the Dirichlet energy on the wrong
# graph's edges and are no longer used by any released script.
SRC = ['src/__init__.py', 'src/datasets.py', 'src/transforms.py', 'src/model.py', 'src/train.py',
       'src/seed.py', 'src/metrics_v2.py', 'src/diagnostics_v2.py',
       'src/graphgps/__init__.py', 'src/graphgps/datasets.py', 'src/graphgps/transforms.py',
       'src/graphgps/model.py', 'src/graphgps/train.py']
SCRIPTS = ['run_experiment.py', 'run_experiment_graphgps.py', 'run_eval_suite.py',
           'run_sink_ablation.py', 'run_norm_intervention.py', 'run_symmetry_probe.py',
           'run_theory_certificate.py', 'lm_sink_reference.py', 'make_fig1_data.py', 'plot_fig1.py',
           'paper_numbers.py', 'aggregate_results.py', 'build_caches.py', 'cpu_gate.py']
CONFIGS = ['base.yaml', 'peptides_grit.yaml'] + [f'peptides_grit_q{q}m.yaml' for q in range(1, 5)] + [
    'zinc_grit.yaml', 'zinc_grit_dotlogit.yaml', 'zinc_grit_blind.yaml', 'zinc_grit_dot.yaml',
    'zinc_grit_dotbias.yaml', 'zinc_grit_noclamp.yaml', 'zinc_grit_official.yaml',
    'zinc_grit_forced_ent1.yaml', 'zinc_grit_forced_ent2.yaml', 'zinc_grit_vnode.yaml',
    'zinc_grit_vnode_bias.yaml', 'zinc_grit_vnode_perecomp.yaml', 'zinc_graphgps.yaml',
    'zinc_gps_gatedgcn.yaml', 'zinc_gps_none.yaml', 'zinc_gps_lappe_none.yaml', 'zinc_gps_vnode.yaml',
    'zinc_gps_vnode_perecomp.yaml']
# run families reported in the paper (seeded runs only: <prefix>-s<k>)
RUNS = ['zinc-grit', 'zinc-grit-dotlogit', 'zinc-grit-blind', 'zinc-grit-dot', 'zinc-grit-dotbias',
        'zinc-grit-noclamp', 'zinc-grit-official', 'zinc-grit-forced-ent1', 'zinc-grit-forced-ent2',
        'zinc-grit-vnode', 'zinc-grit-vnode-bias', 'zinc-grit-vnode-perecomp', 'zinc-graphgps',
        'zinc-gps-gatedgcn', 'zinc-gps-none', 'zinc-gps-lappe-none', 'zinc-gps-vnode',
        'zinc-gps-vnode-perecomp', 'peptides-grit'] + [f'peptides-grit-Q{q}m' for q in range(1, 5)]
EXTRA = {   # run family -> additional (non-JSON) files that paper_numbers.py reads
    'zinc-grit': ['best_model.pt'],
    'zinc-grit-vnode': ['best_model.pt'],
    'zinc-gps-vnode': ['eval_suite.pkl'],
    'peptides-grit': ['eval_suite.pkl'],
}
OTHER = ['outputs/lm_reference/lm_sink_reference.json', 'outputs/figures/fig1_data.json',
         'outputs/figures/fig1_measured.pdf']


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', required=True)
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dest = os.path.abspath(args.dest)
    if os.path.exists(dest) and os.listdir(dest):
        raise SystemExit(f'{dest} exists and is not empty')
    n = 0
    for f in SRC + [f'scripts/{s}' for s in SCRIPTS] + [f'configs/{c}' for c in CONFIGS] + OTHER:
        copy(os.path.join(root, f), os.path.join(dest, f))
        n += 1
    for prefix in RUNS:
        runs = [d for d in glob.glob(os.path.join(root, 'outputs', prefix + '-s*'))
                if re.fullmatch(re.escape(prefix) + r'-s\d', os.path.basename(d))]
        if not runs:
            raise SystemExit(f'no seeded runs for {prefix}')
        for d in sorted(runs):
            keep = glob.glob(os.path.join(d, '*.json')) + [os.path.join(d, 'config.yaml')]
            keep += [os.path.join(d, x) for x in EXTRA.get(prefix, [])]
            for f in keep:
                copy(f, os.path.join(dest, os.path.relpath(f, root)))
                n += 1
    for f in sorted(os.listdir(os.path.join(root, 'release_files'))):
        copy(os.path.join(root, 'release_files', f), os.path.join(dest, f))
        n += 1
    print(f'copied {n} files into {dest}')


if __name__ == '__main__':
    main()

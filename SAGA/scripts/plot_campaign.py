"""PNG summaries from exported tables; no MACE or geometry loading."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['red', 'green', 'blue', 'orange', '#d6b700', 'pink', 'purple', 'violet']


def load(path):
    return list(csv.DictReader(path.open())) if path.exists() else []


def plot(tables, out):
    out.mkdir(parents=True, exist_ok=True)
    history = load(tables / 'generation_history.csv')
    genomes = load(tables / 'genome_summary.csv')
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), constrained_layout=True)
        for i, seed in enumerate(sorted({r['seed'] for r in history}, key=int)):
            rows = sorted((r for r in history if r['seed'] == seed), key=lambda r: int(r['gen']))
            x = [int(r['gen']) for r in rows]
            color = COLORS[i % len(COLORS)]
            axes[0].plot(x, [float(r['best_composite']) for r in rows], color=color, label=seed)
            axes[1].plot(x, [int(r['n_unique']) for r in rows], color=color)
        axes[0].set(ylabel='Current archive leader fitness', ylim=(0, 1.02))
        axes[1].set(ylabel='Unique evaluated genomes')
        axes[0].legend(title='Seed', loc='lower right', frameon=False)
        for ax in axes:
            ax.set(xlabel='Generation'); ax.set_box_aspect(1)
        fig.savefig(out / 'search_progress.png', dpi=180)
        plt.close(fig)
    validation = [r for r in genomes if r['phase'] == 'validation']
    if validation:
        species = [k[:-len('_mean_eV')] for k in validation[0] if k.endswith('_mean_eV')]
        fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
        seeds = sorted({r['seed'] for r in validation}, key=int)
        for i, seed in enumerate(seeds):
            counts = [sum(r[a+'_inside_window'] == 'True' for a in species)
                      for r in validation if r['seed'] == seed and r['eligible'] == 'True']
            ax.scatter([i] * len(counts), counts, color=COLORS[i % len(COLORS)], alpha=.35)
            if counts:
                ax.scatter(i, np.mean(counts), color='black', marker='_', s=200)
        ax.set(xticks=range(len(seeds)), xticklabels=seeds, xlabel='Seed',
               ylabel='Windows hit by eligible validation finalists', yticks=range(len(species)+1))
        ax.set_ylim(-.25, len(species)+.25)
        fig.savefig(out / 'validation_hits.png', dpi=180)
        plt.close(fig)
    if not history and not validation:
        raise ValueError('No generation history or validation rows found; export a checkpoint first')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('tables', type=Path)
    p.add_argument('--out', type=Path, default=Path('analysis_output'))
    a = p.parse_args()
    plot(a.tables, a.out)

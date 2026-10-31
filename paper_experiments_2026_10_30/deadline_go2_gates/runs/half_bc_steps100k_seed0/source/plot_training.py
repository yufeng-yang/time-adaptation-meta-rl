"""Plot recorded optimization loss and actual closed-loop validation results."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/time_adapt_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    progress = json.loads((args.run/'progress.json').read_text())
    bc = [r for r in progress if r['phase'] == 'bc']
    evaluations = [r for r in progress if 'validation' in r]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    axes[0].plot([r['update'] for r in bc], [r['loss'] for r in bc], color='#4c72b0', lw=1.2)
    axes[0].set(xlabel='BC update', ylabel='Action MSE + 0.05 value MSE', title='200-update warmstart')
    for deadline, color in [('18.0', '#55a868'), ('48.0', '#4c72b0'), ('100.0', '#c44e52')]:
        points = [(0 if r['phase'] == 'bc_validation' else r['steps']/1000,
                   r['validation'][deadline]['episodes'][2]['score']) for r in evaluations]
        if points:
            axes[1].plot(*zip(*points), marker='o', color=color, label=f'{float(deadline):g} s')
    axes[1].set(xlabel='Training high-level steps (thousands; 0 = after BC)', ylabel='Episode 3 delivered value',
                title='Closed-loop validation: all 4 gate patterns', ylim=(-.1, 4.2))
    axes[1].legend(title='Deadline', frameon=False)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(args.run/'training_curve.png', dpi=180)
    fig.savefig(args.run/'training_curve.svg')


if __name__ == '__main__':
    main()

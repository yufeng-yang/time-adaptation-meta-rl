"""Plot recorded BC losses and PPO learning curves without rerunning training."""
import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config = json.loads((run / 'config.json').read_text())
    no_bc = config.get('no_bc', False)
    initial_name = 'initial_validation.json' if no_bc else 'after_bc_validation.json'
    results = json.loads((run / 'test_results.json').read_text())
    with (run / 'training.csv').open() as stream:
        training = [{k: float(v) if v not in ('', None) else float('nan') for k, v in row.items()}
                    for row in csv.DictReader(stream)]
    by_update = {int(row['update']): row for row in training}
    validation = [(0, json.loads((run / initial_name).read_text()))]
    validation += [(int(path.stem.split('_')[-1]), json.loads(path.read_text()))
                   for path in sorted(run.glob('validation_*.json'))]
    log_path = run.parent / f'{run.name}.log'
    bc_losses = [(int(update), float(loss)) for update, loss in re.findall(
        r'^BC (\d+)/\d+: loss=([\d.eE+-]+)', log_path.read_text(), re.MULTILINE)]
    steps = np.array([row['low_steps'] for row in training]) / 1000
    returns = np.array([row['train_mean_trial_score'] for row in training])
    eval_steps = np.array([0 if update == 0 else by_update[update]['low_steps']
                           for update, _ in validation]) / 1000
    selected = results['selected_checkpoint']
    best_step = selected['low_steps'] / 1000
    anneal_update = max(1, int(np.ceil(config['ppo_updates'] * .5)))
    anneal_step = by_update[anneal_update]['low_steps'] / 1000
    colors = ['#286090', '#D47A25', '#268169']
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.5))

    def style_ppo(ax):
        ax.set_xlabel('PPO lower-layer steps (thousands)')
        ax.set_xlim(-12, steps[-1] + 12)
        ax.grid(axis='y', alpha=.17)
        ax.axvline(best_step, color='.3', linestyle=':', lw=1.2)
        if not no_bc:
            ax.axvline(anneal_step, color='.55', linestyle='--', lw=1.1)

    ax = axs[0, 0]
    ax.plot(steps, returns, color='.7', lw=.9, alpha=.7, label='Training: raw batch mean')
    window = 10
    rolling = np.convolve(returns, np.ones(window) / window, mode='valid')
    ax.plot(steps[window - 1:], rolling, color=colors[0], lw=2,
            label=f'Training: trailing {window}-update mean')
    means = [np.mean([v['mean_trial_score'] for v in summary.values()])
             for _, summary in validation]
    ax.plot(eval_steps, means, color=colors[1], marker='o', lw=1.8,
            label='Validation: equal mean over 3 deadlines')
    ax.set_title('A  Pure PPO from random initialization' if no_bc else 'A  PPO return after BC warmstart')
    ax.set_ylabel('Mean raw score per 3-episode trial (max 12)')
    style_ppo(ax)
    ax.legend(fontsize=8.5, loc='lower right')

    ax = axs[0, 1]
    for color, deadline in zip(colors, ('30', '65', '150')):
        values = [summary[deadline]['mean_trial_score'] for _, summary in validation]
        ax.plot(eval_steps, values, color=color, marker='o', lw=1.7,
                label=f'Deadline {deadline} steps')
    ax.set_title('B  Held-out validation return by deadline')
    ax.set_ylabel('Mean raw score per 3-episode trial (max 12)')
    ax.set_ylim(-.2 if no_bc else 2.4, 12.6)
    style_ppo(ax)
    ax.legend(fontsize=9, loc='center right')

    ax = axs[1, 0]
    for index, (color, marker) in enumerate(zip(colors, ('o', 's', '^')), 1):
        values = [summary['65']['episodes'][str(index)]['score'] for _, summary in validation]
        ax.plot(eval_steps, values, color=color, marker=marker, lw=1.7, label=f'Episode {index}')
    ax.set_title('C  Deadline 65: validation score across episodes')
    ax.set_ylabel('Mean raw score per episode (max 4)')
    ax.set_ylim(-.1 if no_bc else 1.7, 4.2)
    style_ppo(ax)
    ax.legend(fontsize=9, loc='center left')

    ax = axs[1, 1]
    if no_bc:
        for index, (color, marker) in enumerate(zip(colors, ('o', 's', '^')), 1):
            values = [summary['150']['episodes'][str(index)]['score'] for _, summary in validation]
            ax.plot(eval_steps, values, color=color, marker=marker, lw=1.7, label=f'Episode {index}')
        ax.set_title('D  Deadline 150: validation score across episodes')
        ax.set_ylabel('Mean raw score per episode (max 4)')
        ax.set_ylim(-.1, 4.2)
        style_ppo(ax)
        ax.legend(fontsize=9, loc='upper left')
    elif bc_losses:
        updates, losses = zip(*bc_losses)
        ax.plot(updates, losses, color=colors[0], marker='o', markersize=3, lw=1.8)
    if not no_bc:
        ax.set_title('D  BC warmstart: recorded minibatch loss')
        ax.set_xlabel('BC gradient updates (separate from PPO steps)')
        ax.set_ylabel('Cross-entropy + 0.1 x value MSE')
        ax.grid(axis='y', alpha=.17)
        ax.text(.98, .97, 'Logged every 50 updates\nNo BC reward curve was recorded',
                transform=ax.transAxes, ha='right', va='top', fontsize=9, color='.35')

    fig.suptitle('Pick-Place | ' + ('V2 pure PPO | ' if no_bc else '') +
                 f'fixed outer scores {{0, 1, 2, 4}} | seed {config["seed"]}',
                 fontsize=16, fontweight='bold', y=.98)
    fig.text(.5, .935,
             f'{config["bc_updates"]} BC updates, then {config["ppo_updates"]} PPO updates '
             f'({int(steps[-1] * 1000):,} lower-layer steps); ideal lower controller',
             ha='center', fontsize=10, color='.3')
    fig.text(.06, .048,
             f'Dotted line: validation-selected checkpoint ({selected["phase"]} update {selected["update"]}, '
             f'{selected["low_steps"]:,} steps).  ' + ('No BC warmstart or auxiliary loss.' if no_bc else
             f'Dashed line: BC auxiliary reaches zero (update {anneal_update}).'),
             fontsize=9, color='.3')
    fig.text(.06, .026,
             f'Source: {run.name}/training.csv, validation_*.json, {initial_name}. '
             f'Validation: {config["eval_trials"]} trials/deadline; single training seed; no confidence bands.',
             fontsize=8.5, color='.35')
    fig.subplots_adjust(top=.88, bottom=.13, left=.075, right=.975, hspace=.38, wspace=.25)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(run / f'training_curves.{extension}', dpi=180, facecolor='white')
    plt.close(fig)
    diagnostics = [row for row in training if np.isfinite(row.get('rollout_action_entropy', float('nan')))]
    if diagnostics:
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        dx = np.array([row['low_steps'] / 1000 for row in diagnostics])
        for ax, key, scale, title, ylabel in (
            (axes[0], 'rollout_action_entropy', 1, 'Policy entropy during training', 'Mean action entropy (nats)'),
            (axes[1], 'inspect_action_fraction', 100, 'Inspection actions during training', 'Inspection / all upper actions (%)'),
        ):
            dy = np.array([row[key] * scale for row in diagnostics])
            ax.plot(dx, dy, color='.7', lw=.8, label='Raw update mean')
            if len(dy) >= window:
                smooth = np.convolve(dy, np.ones(window) / window, mode='valid')
                ax.plot(dx[window - 1:], smooth, color=colors[0], lw=1.8,
                        label=f'Trailing {window}-update mean')
            ax.set_title(title)
            ax.set_xlabel('PPO lower-layer steps (thousands)')
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
            ax.grid(axis='y', alpha=.17)
            ax.legend(fontsize=8)
        figure.suptitle(f'Pure PPO diagnostics | seed {config["seed"]}', fontsize=14, fontweight='bold')
        figure.text(.07, .025,
                    f'Source: {run.name}/training.csv. Recorded updates '
                    f'{int(diagnostics[0]["update"])}-{int(diagnostics[-1]["update"])}; '
                    'earlier missing diagnostics are omitted.', fontsize=9, color='.35')
        figure.subplots_adjust(top=.83, bottom=.19, left=.075, right=.98, wspace=.25)
        for extension in ('png', 'pdf'):
            figure.savefig(run / f'training_diagnostics.{extension}', dpi=180, facecolor='white')
        plt.close(figure)
    rows = []
    for update, summary in validation:
        for deadline, values in summary.items():
            row = dict(ppo_update=update, ppo_low_steps=0 if update == 0 else int(by_update[update]['low_steps']),
                       deadline=deadline, mean_trial_score=values['mean_trial_score'])
            row.update({f'episode_{index}_score': values['episodes'][str(index)]['score'] for index in (1, 2, 3)})
            rows.append(row)
    with (run / 'validation_curves.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if bc_losses:
        with (run / 'bc_loss.csv').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['bc_update', 'cross_entropy_plus_0.1_value_mse'])
            writer.writerows(bc_losses)
    print(run / 'training_curves.png')


if __name__ == '__main__':
    main()

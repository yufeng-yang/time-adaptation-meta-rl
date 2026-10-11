"""Portable entry points for the curated 0--64.5w results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENDPOINTS = {'episode': 645145, 'step': 643351}
STARTS = {'episode': 551179, 'step': 550096}


def data():
    return json.loads((HERE / 'curves.json').read_text())


def validate_files():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for relative, expected in manifest['sha256'].items():
        path = (HERE / relative).resolve()
        if not path.is_relative_to(HERE):
            raise AssertionError('Manifest path leaves result directory')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f'Checksum mismatch: {relative}')
    value = data()
    for mode, points in value['curves'].items():
        assert points[0]['steps'] == 0
        assert points[-1]['steps'] == ENDPOINTS[mode]
        assert all(a['steps'] < b['steps'] for a, b in zip(points, points[1:]))
        for p in points:
            assert p['steps'] <= 645145
            source = HERE / p['source']
            records = json.loads(source.read_text())['episodes']
            rows = [r for r in records if r['episode'] == 3]
            cases = {(r['gate_state'], r['deadline']) for r in rows}
            assert len(cases) == 6 and len(rows) == 18
            means = []
            for gate, deadline in cases:
                subset = [r for r in rows if r['gate_state'] == gate and r['deadline'] == deadline]
                assert len(subset) == 3
                means.append(sum(r['return'] for r in subset) / 3)
            assert abs(sum(means) / 6 - p['return_ep3']) < 1e-7
    return manifest


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    validate_files()
    output = HERE / 'generated'
    output.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    styles = {'episode': ('#2864a4', 'o', 'Episode-level h (closed D25 after 55w)'),
              'step': ('#c36526', 's', 'Step-level h (closed D19 throughout)')}
    for mode, points in data()['curves'].items():
        color, marker, label = styles[mode]
        ax.plot([p['steps'] / 10000 for p in points], [p['return_ep3'] for p in points],
                color=color, marker=marker, markersize=4, linewidth=1.8, label=label)
    ax.axhline(.4816666667, color='.45', linestyle=':', linewidth=1,
               label='Optimal six-case mean: 0.482')
    ax.axvline(55.1179, color='.55', linestyle=':', linewidth=1)
    ax.axvline(60, color='.55', linestyle='--', linewidth=1)
    ax.text(54.6179, -1.93, 'Resume / task change', rotation=90, va='bottom', ha='right', fontsize=8)
    ax.text(60.35, -1.93, 'BC = 0', rotation=90, va='bottom', fontsize=8)
    ax.set(title='Last episode of each trial: episode-level vs step-level memory',
           xlabel='Total environment steps (x 10,000)', ylabel='Episode-3 return (six-case mean)',
           xlim=(0, 64.7645), ylim=(-2, .78))
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 64.5])
    ax.grid(alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper left', fontsize=9)
    fig.text(.08, .025,
             'Source: bundled saved-checkpoint evaluations; only episode 3; six cases equally weighted.\n'
             'Three evaluation seeds per case; deterministic actions; no smoothing or extrapolation.\n'
             'After 55w: episode-level closed D25; step-level closed D19.\n'
             'Zero-step networks reconstructed using original seed 0; all other markers are saved checkpoints.', fontsize=8)
    fig.tight_layout(rect=(0, .14, 1, 1))
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'third_episode_return.{extension}', dpi=180)
    plt.close(fig)
    print('Plot:', output / 'third_episode_return.png')


def verify_models():
    import torch
    from tidyup_plan2.baseline3_gate_meta_rl.continuation_h_comparison import train

    validate_files()
    torch.set_num_threads(1)
    for mode, steps in ENDPOINTS.items():
        path = HERE / 'models' / f'{mode}_{steps}.pt'
        _, args, model, _, loaded_steps, _, _ = train.load_runtime(path, mode)
        assert loaded_steps == steps
        args.eval_seed, args.eval_trials_per_gate = 10000, 3
        actual = train.evaluate_preserving_rng(model, args, mode)
        saved = json.loads((HERE / 'evaluations' / f'{mode}_{steps}.json').read_text())
        assert actual['episodes'] == saved['episodes'], f'{mode} episode records differ'
        assert actual['aggregate'] == saved['aggregate'], f'{mode} aggregate differs'
        print(f'{mode}: all 54 episode records match at {steps} steps')


def continue_training(mode, output):
    from tidyup_plan2.baseline3_gate_meta_rl.continuation_h_comparison import train

    validate_files()
    options = argparse.Namespace(
        mode=mode, checkpoint=HERE / 'models' / f'{mode}_{STARTS[mode]}.pt',
        total_steps=645000, resume_seed=20261011, eval_interval_steps=5000,
        checkpoint_interval=20, log_interval=50,
        closed_long_deadline=25 if mode == 'episode' else 19,
        output_dir=output or HERE / 'runs' / mode,
    )
    train.run(options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plot', 'verify', 'train'))
    parser.add_argument('--mode', choices=('episode', 'step'), default='episode')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.command == 'plot':
        plot()
    elif args.command == 'verify':
        verify_models()
    else:
        continue_training(args.mode, args.output_dir)


if __name__ == '__main__':
    main()

"""Replay experiment 3 using only archived policies and archived runtime code."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runtime_source' / 'high_level'))

import torch
from env import TaskConfig
from model import TransformerMetaPolicy
from train import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('half_bc', 'pure_ppo'), default='half_bc')
    parser.add_argument('--checkpoint', choices=('best', 'final'), default='best')
    parser.add_argument('--trials', type=int, default=128, help='Trials per deadline')
    parser.add_argument('--seed', type=int, default=300000)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--render', action='store_true', help='Render one trial instead of evaluating a batch')
    parser.add_argument('--deadline', type=int, default=150)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = ROOT / args.variant / f'{args.checkpoint}.pt'
    if args.render:
        destination = args.output or ROOT / 'replays' / f'{args.variant}_{args.checkpoint}_D{args.deadline}_seed{args.seed}'
        subprocess.run([sys.executable, str(ROOT / 'runtime_source' / 'high_level' / 'render_trial.py'),
                        '--checkpoint', str(checkpoint), '--deadline', str(args.deadline),
                        '--seed', str(args.seed), '--output-dir', str(destination)], check=True)
        return
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = TransformerMetaPolicy().eval()
    model.load_state_dict(saved['model'])
    config = TaskConfig(**saved['config']['task']['task'])
    summary, _ = evaluate(model, config, args.trials, args.seed)
    verified = None
    if args.trials == 128 and args.seed == 300000:
        original = json.loads((ROOT / args.variant / 'test_results.json').read_text())
        expected = original['policy'] if args.checkpoint == 'best' else original['stage_tests']['final']
        verified = summary == expected
        if not verified:
            raise RuntimeError('Archived replay differs from the stored test results.')
    result = dict(variant=args.variant, checkpoint=args.checkpoint, phase=saved['phase'],
                  update=saved['update'], seed=args.seed, trials_per_deadline=args.trials,
                  matches_stored_test_results=verified, summary=summary)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

"""Evaluate a saved upper policy without rendering or MuJoCo."""
import argparse
import json

import torch

from env import TaskConfig
from model import TransformerMetaPolicy
from train import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--trials', type=int, default=32, help='Trials per deadline')
    parser.add_argument('--seed', type=int, default=400000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = TransformerMetaPolicy().eval()
    model.load_state_dict(saved['model'])
    task = dict(saved['config']['task']['task'])
    # Old checkpoints predate the fixed-multiset protocol.
    task.setdefault('score_sampling', 'independent')
    config = TaskConfig(**task)
    summary, trials = evaluate(model, config, args.trials, args.seed)
    print(json.dumps(summary, indent=2))
    for deadline in config.deadlines:
        example = next(t for t in trials if t['deadline'] == deadline)
        print(f"\nDeadline={deadline}; scores S/A/B/C/D={example['scores']}")
        for record in example['records']:
            print(f"  Episode {record['episode']}: {record['actions']}; "
                  f"score={record['score']}; steps={record['low_steps']}; "
                  f"delivered={record['delivered']}")


if __name__ == '__main__':
    main()

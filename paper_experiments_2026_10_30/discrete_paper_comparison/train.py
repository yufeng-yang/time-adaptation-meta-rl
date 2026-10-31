"""Matched input ablations of the retained discrete Transformer trial PPO."""
import argparse
from pathlib import Path
import torch
from tidyup_plan2.baseline3_gate_meta_rl.episode_transformer_bc import train as engine


class AblationPolicy(engine.TransformerTrialActorCritic):
    def __init__(self, hidden_dim=32, transformer_layers=2, attention_heads=4, mode='full'):
        super().__init__(hidden_dim, transformer_layers, attention_heads)
        if mode not in ('state', 'time', 'context', 'full'):
            raise ValueError(mode)
        self.mode = mode

    def mask_observation(self, observations):
        if self.mode in ('state', 'context'):
            observations = observations.clone()
            observations[..., -1] = 0.
        return observations

    def forward(self, observations, hidden):
        if self.mode in ('state', 'time'):
            hidden = torch.zeros_like(hidden)
        return super().forward(self.mask_observation(observations), hidden)

    def update_hidden(self, hidden, observations, actions, rewards, next_observations, dones):
        if self.mode in ('state', 'time'):
            return torch.zeros_like(hidden)
        return super().update_hidden(hidden, self.mask_observation(observations), actions,
                                     rewards, self.mask_observation(next_observations), dones)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('state','time','context'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=800000)
    options = parser.parse_args()
    args = engine.parse_args(['--total-steps', str(options.steps), '--output-dir', options.output,
                              '--eval-interval-steps', '50000', '--checkpoint-interval-steps', '200000'])
    args.policy_mode = options.mode
    args.training_source_paths = [str(Path(__file__).resolve()),
                                 str(Path(engine.ROOT)/'tidyup_plan2/envs/base_env_V4.py'),
                                 str(Path(engine.ROOT)/'tidyup_plan2/envs/base_env_V3.py')]
    engine.train(args, model_factory=lambda *dims: AblationPolicy(*dims, mode=options.mode))


if __name__ == '__main__':
    main()

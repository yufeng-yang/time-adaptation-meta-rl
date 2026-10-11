"""Regression checks for fixed task parameters and inherited trial memory."""
from argparse import Namespace
import unittest
from unittest.mock import patch

import gymnasium as gym
import torch

from tidyup_plan2.baseline3_gate_meta_rl import train_decay_bc as train


class RecordingEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.starts = []

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        base = self.unwrapped
        self.starts.append((base.gate_state, base.deadline, base.remaining_steps))
        return result


class MemorySpy:
    def __init__(self):
        self.initializations = 0
        self.updates = 0
        self.controller_memory = []

    def eval(self):
        return self

    def initial_hidden(self, device):
        self.initializations += 1
        return torch.zeros(2, device=device)

    def __call__(self, observation, hidden):
        self.controller_memory.append(float(hidden[0]))
        # Deterministic forward action, so the environment supplies real endings.
        return torch.tensor([0., 0., 10.]), torch.tensor(0.)

    def update_hidden(self, hidden, *transitions):
        self.updates += 1
        return hidden + 1


class TrialProtocolTests(unittest.TestCase):
    def setUp(self):
        self.args = Namespace(
            step_penalty=.01, time_cap=20., gamma=.99, curriculum_steps=500_000,
            total_steps=1_000_000, eval_trials_per_gate=1, eval_seed=10_000,
        )
        self.device = torch.device('cpu')

    def test_sample_once_inherit_memory_and_reset_only_on_new_trial(self):
        for module in (train,):
            for gate in ('open', 'closed'):
                with self.subTest(module=module.__name__, gate=gate):
                    env = RecordingEnv(module.make_env(self.args))
                    model = MemorySpy()
                    try:
                        # A trial crosses the curriculum boundary, but must not
                        # resample its D between episodes.
                        with patch.object(module, 'curriculum_deadline', side_effect=[19, 15, 17]) as sample:
                            trial = module.collect_trial(
                                model, env, gate, 0, 499_999, self.args, self.device,
                                deterministic=True,
                            )
                            sample.assert_called_once_with(499_999, 500_000)
                        self.assertEqual([e.deadline for e in trial.episodes], [19]*3)
                        self.assertEqual(env.starts, [(gate, 19, 19)]*3)
                        self.assertEqual(model.initializations, 1)
                        self.assertEqual(model.updates, 2)
                        self.assertEqual(model.controller_memory, [0.]*19 + [1.]*19 + [2.]*19)

                        other_gate = 'closed' if gate == 'open' else 'open'
                        module.collect_trial(
                            model, env, other_gate, 1, 1_000_000, self.args,
                            self.device, deterministic=True, trial_deadline=17,
                        )
                        self.assertEqual(env.starts[3:], [(other_gate, 17, 17)]*3)
                        self.assertEqual(model.initializations, 2)
                        self.assertEqual(model.controller_memory[57:], [0.]*17 + [1.]*17 + [2.]*17)
                    finally:
                        env.close()

    def test_evaluation_covers_only_six_fixed_tasks(self):
        for module in (train,):
            with self.subTest(module=module.__name__):
                result = module.evaluate(MemorySpy(), self.args, self.device)
                self.assertEqual(len(result['episodes']), 6*3)
                for row in result['episodes']:
                    self.assertEqual(row['deadline_sequence'], [row['deadline']]*3)
                self.assertEqual(set(result['aggregate']), {'open', 'closed'})
                for gate in result['aggregate']:
                    self.assertEqual(set(result['aggregate'][gate]), {'15', '17', '19'})


if __name__ == '__main__':
    unittest.main()

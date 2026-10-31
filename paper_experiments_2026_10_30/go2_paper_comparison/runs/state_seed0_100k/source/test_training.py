"""Check causal replay and episode memory before using physical rollouts."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from environment import GateTrial
from layout import Config
from model import MetaPolicy, log_probability
from test_environment import FakeBackend
from train import collect_batch, replay, finish_returns, bc_coefficient


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        self.model = MetaPolicy()

    def rollout(self):
        class TexturedBackend(FakeBackend):
            def local_map(self):
                grid = np.zeros((3, 64, 64), np.float32)
                grid[:, 20:30, 20:30] = 1.
                return {'grid': grid}
        envs = [GateTrial(TexturedBackend(), Config(), seed=i) for i in range(2)]
        def advance(envs, actions, active):
            return [e.step(a) if run else None for e, a, run in zip(envs, actions, active)]
        with patch('train.advance_batch', advance):
            return collect_batch(self.model, envs, [1., 2.], patterns=[(True, False), (False, True)])

    def test_replay_matches_behavior_density_and_values_across_resets(self):
        trials = self.rollout()
        out = replay(self.model, trials)
        lp = log_probability(out['mean'], self.model.log_std, out['raw_actions'])
        torch.testing.assert_close(lp, out['log_probs'], atol=2e-5, rtol=2e-5)
        expected = [value for ep in range(3) for t in trials for value in t['episodes'][ep]['values']]
        torch.testing.assert_close(out['value'], torch.tensor(expected), atol=2e-5, rtol=2e-5)
        for trial in trials:
            self.assertEqual(len(trial['records']), 3)
            for ep in trial['episodes']:
                self.assertEqual(ep['obs'][0]['state'][18], 0.)
                self.assertEqual(ep['obs'][-1]['state'][18], 1.)

    def test_later_actions_backpropagate_through_episode_gru(self):
        out = replay(self.model, self.rollout())
        out['mean'][-1].square().sum().backward()
        self.assertGreater(float(self.model.gru.weight_ih.grad.abs().sum()), 0.)
        self.assertGreater(float(self.model.cnn[0].weight.grad.abs().sum()), 0.)

    def test_transformer_does_not_look_at_future_frames(self):
        encoded = torch.randn(2, 5, 64)
        hidden = self.model.initial_hidden(2)
        before = self.model.forward_features(encoded, torch.tensor([5, 5]), hidden)[0]
        changed = encoded.clone()
        changed[:, 3:] += 20.
        after = self.model.forward_features(changed, torch.tensor([5, 5]), hidden)[0]
        torch.testing.assert_close(before[:, :3], after[:, :3])

    def test_credit_crosses_episode_boundary_without_discount(self):
        trial = {'episodes': [dict(rewards=[0., x], values=[0., 0.]) for x in (.25, 1., 1.)]}
        finish_returns([trial])
        np.testing.assert_allclose(trial['episodes'][0]['returns'], [2.25, 2.25])
        np.testing.assert_allclose(trial['episodes'][1]['returns'], [2., 2.])
        np.testing.assert_allclose(trial['episodes'][2]['returns'], [1., 1.])

    def test_bc_starts_at_one_and_reaches_zero(self):
        self.assertEqual(bc_coefficient(0, 80000), 1.)
        self.assertEqual(bc_coefficient(40000, 80000), .5)
        self.assertEqual(bc_coefficient(80000, 80000), 0.)
        self.assertEqual(bc_coefficient(100000, 80000), 0.)
        weights = [bc_coefficient(i, 80000) for i in range(0, 100001, 1000)]
        self.assertTrue(all(a >= b for a, b in zip(weights, weights[1:])))


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
import torch

from env import PickPlaceTrial, TaskConfig
from model import TransformerMetaPolicy


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.env = PickPlaceTrial(seed=0)
        self.env.reset(deadline=65, scores=[1, 0, 1, 2, 4])

    def test_unseen_scores_do_not_change_observation(self):
        a = self.env.observation()
        self.env._scores[1:] = [4, 2, 0, 1]
        np.testing.assert_array_equal(a, self.env.observation())

    def test_every_trial_has_the_complete_outer_score_set(self):
        permutations = set()
        for _ in range(100):
            self.env.reset()
            self.assertEqual(self.env._scores[0], 1.)
            self.assertEqual(sorted(self.env._scores[1:]), [0., 1., 2., 4.])
            permutations.add(tuple(self.env._scores[1:]))
        self.assertGreater(len(permutations), 1)

    def test_paired_deadlines_share_layout_and_score_assignment(self):
        trials = [PickPlaceTrial(seed=123) for _ in range(3)]
        for deadline, env in zip((30, 65, 150), trials):
            env.reset(deadline=deadline)
        for env in trials[1:]:
            np.testing.assert_array_equal(env._scores, trials[0]._scores)
            np.testing.assert_array_equal(env.initial_balls, trials[0].initial_balls)

    def test_exact_deadline_delivery_succeeds_once(self):
        self.env.reset(deadline=25)
        obs, reward, done, info = self.env.step(4)
        self.assertTrue(info['episode_done'])
        self.assertTrue(info['episode_record']['delivered'])
        self.assertEqual(info['episode_record']['ball'], 'S')
        self.assertEqual(len(self.env.records), 1)
        self.assertEqual(self.env.episode, 1)
        self.assertEqual(reward, .25)

    def test_timeout_does_not_count_as_delivery(self):
        self.env.reset(deadline=24)
        obs, reward, done, info = self.env.step(4)
        self.assertTrue(info['episode_done'])
        self.assertFalse(info['episode_record']['delivered'])
        self.assertEqual(reward, 0)

    def test_boundary_keeps_terminal_event_only_in_previous_episode(self):
        obs, reward, done, info = self.env.step(8)
        terminal = info['terminal_observation']
        self.assertEqual(reward, 1.)
        self.assertEqual(terminal[28], 1.)
        self.assertEqual(terminal[29], 1.)
        self.assertEqual(terminal[39], 1.)
        self.assertEqual(terminal[41], 1.)
        np.testing.assert_array_equal(obs[24:43], np.zeros(19))
        np.testing.assert_allclose(obs[:3], self.env.home / .5)
        self.assertEqual(obs[23], 1.)

    def test_collector_summarizes_terminal_event_and_starts_clean(self):
        from train import collect_trial
        trial = collect_trial(None, PickPlaceTrial(seed=1), 30, baseline='safe')
        self.assertEqual(trial['episodes'][0]['obs'][-1][24], 1.)
        self.assertEqual(trial['episodes'][0]['obs'][-1][39], .25)
        np.testing.assert_array_equal(trial['episodes'][1]['obs'][0][24:43], np.zeros(19))

    def test_three_deliveries_are_three_episodes_not_one(self):
        positions = self.env.initial_balls.copy()
        scores = self.env._scores.copy()
        for index in range(3):
            obs, reward, done, info = self.env.step(4)
            self.assertEqual(done, index == 2)
            np.testing.assert_array_equal(positions, self.env.initial_balls)
            np.testing.assert_array_equal(scores, self.env._scores)
        with self.assertRaises(RuntimeError):
            self.env.step(4)

    def test_zero_value_has_valid_contact_marker(self):
        obs, reward, done, info = self.env.step(0)
        self.assertEqual(obs[25], 1)
        self.assertEqual(obs[29], 0)
        self.assertEqual(obs[24:29].sum(), 1)

    def test_feedback_is_not_a_persistent_value_table(self):
        self.env.step(0)
        obs, *_ = self.env.step(1)
        self.assertEqual(obs[24:29].sum(), 1)
        self.assertEqual(obs[25], 0)
        self.assertEqual(obs[26], 1)

    def test_partial_move_does_not_teleport_to_ball(self):
        self.env.reset(deadline=1)
        start = self.env.home.copy()
        target = self.env.initial_balls[1].copy()
        self.env.step(0)
        # Episode resets automatically, but no contact was revealed.
        self.assertEqual(self.env.feedback_id, -1)
        self.assertFalse(self.env.records[0]['delivered'])
        self.assertEqual(self.env.records[0]['low_steps'], 1)

    def test_duration_matches_execution(self):
        for action in range(9):
            self.env.reset(deadline=150)
            expected = self.env.duration(action)
            obs, reward, done, info = self.env.step(action)
            self.assertEqual(info['duration'], expected)

    def test_transformer_actor_cannot_read_future_tokens(self):
        torch.manual_seed(0)
        policy = TransformerMetaPolicy().eval()
        observations = torch.randn(1, 7, 49)
        altered = observations.clone()
        altered[:, 4:] += 10
        with torch.no_grad():
            a, _, _ = policy.forward_episode(observations, torch.tensor([7]), policy.initial_hidden())
            b, _, _ = policy.forward_episode(altered, torch.tensor([7]), policy.initial_hidden())
        torch.testing.assert_close(a[:, :4], b[:, :4])

    def test_replay_matches_rollout_probabilities(self):
        from train import collect_trial, replay
        torch.manual_seed(5)
        model = TransformerMetaPolicy()
        trial = collect_trial(model, PickPlaceTrial(seed=3), deadline=30)
        with torch.no_grad():
            output = replay(model, [trial])
            distribution = torch.distributions.Categorical(logits=output['logits'])
            replayed = distribution.log_prob(output['actions'])
        torch.testing.assert_close(replayed, output['old_log_probs'], atol=2e-5, rtol=2e-5)

    def test_no_bc_mode_never_uses_training_teacher_or_cloning_loss(self):
        import contextlib
        import io
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        import train
        original_collect = train.collect_trial
        trained = False

        def guarded_collect(model, env, *args, **kwargs):
            if not trained and kwargs.get('teacher'):
                raise AssertionError('Training requested a teacher in no-BC mode')
            return original_collect(model, env, *args, **kwargs)

        original_evaluate = train.evaluate

        def evaluation(model, config, count, seed, reset_meta=False):
            nonlocal trained
            if seed == 300000:
                trained = True
            return original_evaluate(model, config, count, seed, reset_meta)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(train, 'collect_trial', side_effect=guarded_collect), \
                 patch.object(train, 'evaluate', side_effect=evaluation), \
                 patch.object(train, 'calibration', return_value={}), \
                 patch.object(train.F, 'cross_entropy', side_effect=AssertionError('BC loss called')), \
                 contextlib.redirect_stdout(io.StringIO()):
                train.main(['--no-bc', '--output-dir', directory, '--ppo-updates', '1',
                            '--batch-trials', '2', '--eval-trials', '1', '--eval-every', '1',
                            '--bc-updates', '1000', '--expert-trials', '768'])
            path = Path(directory)
            config = json.loads((path / 'config.json').read_text())
            self.assertEqual(config['bc_updates'], 0)
            self.assertEqual(config['expert_trials'], 0)
            self.assertEqual(config['bc_aux_initial_weight'], 0)
            self.assertFalse((path / 'bc.pt').exists())
            self.assertTrue((path / 'initial.pt').exists())
            rows = json.loads((path / 'progress.json').read_text())
            self.assertEqual(rows[0]['bc_weight'], 0.)

    def test_saved_rng_resume_matches_uninterrupted_pure_ppo(self):
        import contextlib
        import io
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        import train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = ['--no-bc', '--batch-trials', '2', '--eval-trials', '1', '--eval-every', '1']
            with patch.object(train, 'calibration', return_value={}), contextlib.redirect_stdout(io.StringIO()):
                train.main(options + ['--output-dir', str(root / 'full'), '--ppo-updates', '2'])
                train.main(options + ['--output-dir', str(root / 'first'), '--ppo-updates', '1'])
                train.main(options + ['--output-dir', str(root / 'resumed'), '--ppo-updates', '2',
                                      '--resume-from', str(root / 'first' / 'final.pt')])
            full = torch.load(root / 'full' / 'final.pt', weights_only=False)
            resumed = torch.load(root / 'resumed' / 'final.pt', weights_only=False)
            self.assertEqual(full['low_steps'], resumed['low_steps'])
            for key, tensor in full['model'].items():
                torch.testing.assert_close(tensor, resumed['model'][key], rtol=0, atol=0)
            config = json.loads((root / 'resumed' / 'config.json').read_text())
            self.assertTrue(config['resume_rng_restored'])
            self.assertEqual(config['resume_start_update'], 1)
            progress = json.loads((root / 'resumed' / 'progress.json').read_text())
            self.assertEqual([row['update'] for row in progress], [1, 2])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

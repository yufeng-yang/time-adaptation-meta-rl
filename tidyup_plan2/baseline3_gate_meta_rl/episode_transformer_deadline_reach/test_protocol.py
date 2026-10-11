"""Use real environment paths to check gamma compensation at all endings."""
import unittest

import numpy as np
import torch

from tidyup_plan2.baseline3_gate_meta_rl.episode_transformer_bc import train
from . import train as train_deadline_reach
from .reward import DiscountCompensatedTerminalRefund


class TerminalRefundTests(unittest.TestCase):
    def setUp(self):
        self.old = train.parse_args([])
        self.new = train_deadline_reach.parse_args([])
        torch.set_num_threads(1)

    def rollout(self, args, script, deadline=25, gate="open"):
        env = train.make_env(args)
        rewards, infos = [], []
        try:
            env.reset(seed=0, options=dict(new_trial=True, gate_state=gate, deadline=deadline))
            for action in script:
                _, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                infos.append(info)
                if terminated or truncated:
                    break
            self.assertTrue(terminated or truncated)
            self.assertAlmostEqual(sum(rewards), info["episode_return"], places=10)
            return rewards, infos
        finally:
            env.close()

    def test_safe_success_equal_at_14_18_22_24_steps(self):
        direct = train.base.ThreeRouteHiddenGateEnvV4.DIRECT_SAFE_ACTIONS
        detour = train.base.ThreeRouteHiddenGateEnvV4.DETOUR_SAFE_ACTIONS
        # Pad with paired turns, not collisions: this adds only step costs.
        scripts = [direct, detour, (0, 1)*4 + direct, (0, 1)*5 + direct]
        lengths = []
        for script in scripts:
            rewards, infos = self.rollout(self.new, script)
            length = len(rewards)
            lengths.append(length)
            self.assertTrue(infos[-1]["success"])
            self.assertAlmostEqual(sum(self.new.gamma**i*r for i, r in enumerate(rewards)), 1., places=10)
            self.assertTrue(all(i["discounted_step_refund"] == 0 for i in infos[:-1]))
            self.assertTrue(all(abs(r+.01) < 1e-10 for r in rewards[:-1]))
        self.assertEqual(lengths, [14, 18, 22, 24])
        # The real first-episode probe + safe detour is also 22 steps.
        env = train.make_env(self.new)
        try:
            trial = train.base.collect_expert_trial(env, "closed", 25, 0, self.new)
        finally:
            env.close()
        self.assertEqual([e.steps for e in trial.episodes], [22, 18, 18])
        np.testing.assert_allclose([e.discounted_return for e in trial.episodes], [1, 1, 1], atol=1e-6)

    def test_danger_collision_and_timeout_unchanged_except_terminal_compensation(self):
        world = train.base.ThreeRouteHiddenGateEnvV4
        cases = [
            (world.MEDIUM_PATH_ACTIONS, 17, "closed"),
            (world.MEDIUM_PATH_ACTIONS, 15, "closed"),
            ((0,)*15, 15, "closed"),
            ((2,)*15, 15, "closed"),
        ]
        for script, deadline, gate in cases:
            with self.subTest(deadline=deadline, script=script):
                old, old_info = self.rollout(self.old, script, deadline, gate)
                new, new_info = self.rollout(self.new, script, deadline, gate)
                self.assertEqual(len(old), len(new))
                np.testing.assert_allclose(new[:-1], old[:-1], atol=0)
                n = len(new)
                discounts = self.new.gamma**np.arange(n)
                expected_refund = .01*discounts.sum()/discounts[-1]
                expected_goal = (1/discounts[-1]-1) if new_info[-1]["success"] else 0.
                self.assertAlmostEqual(new[-1]-old[-1], expected_refund+expected_goal, places=10)
                target = sum(discounts*(np.array(old)+.01))
                if old_info[-1]["success"]:
                    target += 1-discounts[-1]
                self.assertAlmostEqual(sum(discounts*np.array(new)), target, places=10)
                for before, after in zip(old_info, new_info):
                    for key in ("collision", "timeout", "success", "danger_penalty_applied"):
                        self.assertEqual(before[key], after[key])

    def test_gamma_one_and_refund_resets_each_episode(self):
        args = train_deadline_reach.parse_args(["--gamma", "1"])
        env = train.make_env(args)
        try:
            for index in range(3):
                env.reset(seed=0 if index == 0 else None,
                          options=dict(new_trial=index == 0, gate_state="open", deadline=25))
                rewards = []
                for action in train.base.ThreeRouteHiddenGateEnvV4.DIRECT_SAFE_ACTIONS:
                    _, reward, terminated, _, info = env.step(action)
                    rewards.append(reward)
                self.assertTrue(terminated)
                self.assertAlmostEqual(info["discounted_step_refund"], .14)
                self.assertEqual(info["goal_discount_compensation"], 0.)
                self.assertAlmostEqual(sum(rewards), 1.)
        finally:
            env.close()

    def test_separate_entry_and_old_defaults_preserved(self):
        self.assertEqual(self.old.gamma, .99)
        self.assertEqual(self.new.gamma, 1.)
        self.assertEqual(train_deadline_reach.parse_args(["--gamma", ".99"]).gamma, .99)
        self.assertEqual(train_deadline_reach.parse_args(["--gamma=.95"]).gamma, .95)
        self.assertFalse(getattr(self.old, "discount_compensated_terminal_refund", False))
        self.assertTrue(self.new.report_discounted_return)
        self.assertTrue(self.new.discount_compensated_terminal_refund)
        self.assertNotEqual(self.old.output_dir, self.new.output_dir)
        self.assertEqual(self.new.output_dir, str(train_deadline_reach.HERE / "runs/seed0"))
        self.assertTrue(all(train_deadline_reach.HERE == train_deadline_reach.Path(p).parent
                            for p in self.new.training_source_paths))
        for invalid in ("0", "-1", "1.1"):
            with self.assertRaises(ValueError):
                train_deadline_reach.parse_args(["--gamma", invalid])

    def test_default_gamma_one_returns_for_all_six_expert_trials(self):
        env = train.make_env(self.new)
        try:
            for gate in ("open", "closed"):
                for deadline in (15, 17, 25):
                    with self.subTest(gate=gate, deadline=deadline):
                        trial = train.base.collect_expert_trial(env, gate, deadline, 0, self.new)
                        expected = ([1., 1., 1.] if gate == "open" or deadline == 25 else
                                    [-1., .8, .8] if deadline == 17 else [-1., -1., -1.])
                        np.testing.assert_allclose(
                            [e.discounted_return for e in trial.episodes], expected, atol=1e-6)
                        np.testing.assert_allclose(
                            [e.episode_return for e in trial.episodes], expected, atol=1e-6)
        finally:
            env.close()

    def test_refunded_trials_replay_through_ppo_with_finite_losses(self):
        torch.manual_seed(0)
        args = self.new
        args.ppo_epochs = 1
        model = train.TransformerTrialActorCritic()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        env = train.make_env(args)
        try:
            experts = train.build_experts(env, args)
            trial = train.base.collect_trial(model, env, "closed", 0, 0, args,
                                             torch.device("cpu"), trial_deadline=25)
        finally:
            env.close()
        targets, advantages = train.base.build_meta_targets([trial], args.gamma, args.beta_trial)
        metrics = train.base.ppo_update(model, optimizer, [trial], targets, advantages, args,
                                       torch.device("cpu"), experts, 100)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertTrue(all(np.isclose(train.reported_return(e, args), e.discounted_return)
                            for e in trial.episodes))

    def test_arbitrary_19_22_25_step_safe_arrivals_have_same_discounted_return(self):
        # Test arbitrary arrival lengths independently of grid-route parity.
        class ArrivalEnv(train.base.gym.Env):
            step_penalty = .01

            def __init__(self, arrival):
                self.arrival = arrival
                self.observation_space = train.base.gym.spaces.Box(0., 1., shape=(1,), dtype=np.float32)
                self.action_space = train.base.gym.spaces.Discrete(1)

            def _reward(self):
                return 1.

            def reset(self, **kwargs):
                self.steps, self.episode_return = 0, 0.
                return np.zeros(1, np.float32), {}

            def step(self, action):
                self.steps += 1
                success = self.steps == self.arrival
                reward = float(success)-self.step_penalty
                self.episode_return += reward
                return np.zeros(1, np.float32), reward, success, False, {"success": success}

        for length in (19, 22, 25):
            env = DiscountCompensatedTerminalRefund(ArrivalEnv(length), gamma=.99)
            env.reset()
            total = 0.
            for t in range(length):
                _, reward, done, _, _ = env.step(0)
                total += .99**t*reward
            self.assertTrue(done)
            self.assertAlmostEqual(total, 1., places=10)
            env.close()


if __name__ == "__main__":
    unittest.main()

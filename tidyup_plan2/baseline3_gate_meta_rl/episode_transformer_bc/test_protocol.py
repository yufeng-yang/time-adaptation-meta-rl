"""Check time coordinates, causal trial use, Transformer and PPO replay."""
import random
import unittest
import csv
import hashlib
import json
from argparse import Namespace

import numpy as np
import torch
from torch.distributions import Categorical

from . import train
from ..test_trial_protocol import MemorySpy, RecordingEnv


class TransformerProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        self.args = train.parse_args([])
        self.device = torch.device("cpu")

    def test_defaults_and_all_six_expert_trials(self):
        args = self.args
        self.assertEqual(args.total_steps, 1_000_000)
        self.assertEqual(args.gamma, .99)
        self.assertFalse(getattr(args, "discount_compensated_terminal_refund", False))
        self.assertEqual(args.curriculum_steps, 0)
        self.assertEqual(args.time_scale, 25.)
        self.assertFalse(args.positional_encoding)
        self.assertFalse(args.causal_mask)
        self.assertTrue(args.from_scratch)
        env = train.make_env(args)
        try:
            experts = train.build_experts(env, args)
        finally:
            env.close()
        self.assertEqual({(t.gate_state, t.episodes[0].deadline) for t in experts},
                         {(g, d) for g in ("open", "closed") for d in (15, 17, 25)})
        for trial in experts:
            deadline = trial.episodes[0].deadline
            if trial.gate_state == "open":
                self.assertEqual([e.steps for e in trial.episodes], [14]*3)
                self.assertTrue(all(e.success for e in trial.episodes))
            elif deadline == 25:
                self.assertEqual([e.steps for e in trial.episodes], [22, 18, 18])
                self.assertTrue(all(e.success for e in trial.episodes))
                np.testing.assert_allclose([e.episode_return for e in trial.episodes], [.78, .82, .82], atol=1e-6)
            elif deadline == 17:
                self.assertEqual([e.steps for e in trial.episodes], [17, 16, 16])
                self.assertEqual([e.route for e in trial.episodes], ["detour", "medium", "medium"])
                self.assertEqual([e.success for e in trial.episodes], [False, True, True])
            else:
                self.assertTrue(all(not e.success for e in trial.episodes))
            self.assertTrue(trial.episodes[0].probed_gate)

    def test_time_is_fixed_scale_not_clipped_or_divided_by_own_deadline(self):
        env = train.make_env(self.args)
        try:
            for deadline in (15, 17, 25):
                obs, _ = env.reset(seed=0, options={"new_trial": True, "deadline": deadline, "gate_state": "closed"})
                self.assertAlmostEqual(float(obs[-1]), deadline/25, places=6)
                nxt, *_ = env.step(0)
                self.assertAlmostEqual(float(nxt[-1]), (deadline-1)/25, places=6)
                self.assertTrue(env.observation_space.contains(obs))
            raw, _ = env.env.reset(seed=0, options={"new_trial": True, "deadline": 25, "gate_state": "closed"})
            raw["time"] = np.array([30], dtype=np.float32)
            encoded = env.observation(raw)
            self.assertAlmostEqual(float(encoded[-1]), 1.2, places=6)
            self.assertTrue(env.observation_space.contains(encoded))
        finally:
            env.close()

    def test_no_curriculum_and_episode_memory_inheritance(self):
        np.random.seed(0)
        self.assertEqual({train.sample_deadline() for _ in range(1000)}, {15, 17, 25})
        env = RecordingEnv(train.make_env(self.args))
        model = MemorySpy()
        try:
            train.base.collect_trial(model, env, "closed", 0, 0, self.args, self.device,
                                     deterministic=True, trial_deadline=25)
            self.assertEqual(env.starts, [("closed", 25, 25)]*3)
            self.assertEqual(model.initializations, 1)
            self.assertEqual(model.updates, 2)
            self.assertEqual(model.controller_memory, [0.]*25 + [1.]*25 + [2.]*25)
        finally:
            env.close()

    def test_no_position_transformer_pool_is_permutation_invariant(self):
        model = train.TransformerTrialActorCritic()
        size = 13
        data = (torch.randn(size, train.base.OBS_DIM), torch.randint(0, 3, (size,)),
                torch.randn(size), torch.randn(size, train.base.OBS_DIM), torch.zeros(size))
        data[-1][-1] = 1.
        hidden = model.initial_hidden(self.device)
        order = torch.randperm(size)
        with torch.no_grad():
            original = model.update_hidden(hidden, *data)
            permuted = model.update_hidden(hidden, *(x[order] for x in data))
        torch.testing.assert_close(original, permuted, atol=2e-6, rtol=2e-6)
        self.assertTrue(all(m.p == 0 for m in model.modules() if isinstance(m, torch.nn.Dropout)))

    def test_ppo_replay_matches_rollout_and_gradients_reach_transformer(self):
        model = train.TransformerTrialActorCritic()
        env = train.make_env(self.args)
        try:
            trial = train.base.collect_trial(model, env, "closed", 0, 0, self.args,
                                             self.device, trial_deadline=25)
        finally:
            env.close()
        hidden = model.initial_hidden(self.device)
        last_loss = None
        for index, episode in enumerate(trial.episodes):
            data = train.base.episode_tensors(episode, self.device)
            logits, values = model(data[0], hidden)
            actual = Categorical(logits=logits).log_prob(data[1])
            torch.testing.assert_close(actual, torch.from_numpy(episode.old_log_probs), atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(values, torch.from_numpy(episode.old_values), atol=2e-6, rtol=2e-6)
            last_loss = logits.square().mean()
            if index < 2:
                hidden = model.update_hidden(hidden, *data)
        last_loss.backward()
        grad = model.trajectory_transformer.layers[0].self_attn.in_proj_weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.)

    def test_evaluation_preserves_rng_and_train_mode(self):
        model = train.TransformerTrialActorCritic()
        random.seed(4)
        np.random.seed(4)
        torch.manual_seed(4)
        before = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
        result = train.evaluate(model, self.args, 0)
        self.assertEqual(len(result["episodes"]), 54)
        self.assertTrue(model.training)
        self.assertEqual(random.getstate(), before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
        self.assertEqual(np.random.get_state()[2:], before[1][2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before[2]))
        self.assertTrue(all(r["deadline_sequence"] == [r["deadline"]]*3 for r in result["episodes"]))

    def test_selected_80w_release_matches_model_and_contains_no_later_records(self):
        release = train.HERE / "release_0_80w"
        manifest_path = release / "manifest.json"
        if not manifest_path.exists():
            self.skipTest("Selected result has not been exported")
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["selected_steps"], 800058)
        for relative, expected in manifest["sha256"].items():
            self.assertEqual(hashlib.sha256((release / relative).read_bytes()).hexdigest(), expected)
        for name in ("train_trials.csv", "evaluation_curve.csv"):
            with (release / name).open() as handle:
                steps = [int(r["total_steps"]) for r in csv.DictReader(handle)]
            self.assertEqual(max(steps), 800058)
            self.assertTrue(all(s <= 800058 for s in steps))
        self.assertTrue(all(int(p.stem.split("_")[1]) <= 800058
                            for p in (release / "evaluations").glob("*.json")))
        checkpoint = torch.load(release / "final_model.pt", map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["total_steps"], 800058)
        args = Namespace(**checkpoint["config"])
        model = train.TransformerTrialActorCritic(args.hidden_dim, args.transformer_layers, args.attention_heads)
        model.load_state_dict(checkpoint["model_state_dict"])
        result = train.evaluate(model, args, 800058)
        saved = json.loads((release / "latest_evaluation.json").read_text())
        self.assertEqual(result["episodes"], saved["episodes"])


if __name__ == "__main__":
    unittest.main()

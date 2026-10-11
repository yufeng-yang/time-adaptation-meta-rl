"""The extra metric changes neither training rewards nor the policy's memory."""
import unittest
from . import train
from .evaluate_deadline_score import deadline_score, score_rewards, load_curve


class DeadlineScoreTests(unittest.TestCase):
    def test_safe_success_independent_of_length_and_gamma(self):
        for length in (14, 18, 19, 22, 25):
            rewards = [-.01]*length
            rewards[-1] += 1
            for gamma in (.99, 1):
                result = score_rewards(rewards, gamma)
                self.assertAlmostEqual(result["deadline_score"], 1.)
                self.assertAlmostEqual(deadline_score(sum(rewards), length), 1.)

    def test_fixed_danger_collision_timeout_penalties_survive(self):
        for danger_step in (0, 9, 14):
            rewards = [-.01]*17
            rewards[danger_step] -= .2
            rewards[2] -= .05
            rewards[-1] -= 1
            self.assertAlmostEqual(score_rewards(rewards)["deadline_score"], -1.25)
        rewards = [-.01]*16
        rewards[9] -= .2
        rewards[-1] += 1
        self.assertAlmostEqual(score_rewards(rewards)["deadline_score"], .8)

    def test_scoring_does_not_mutate_reward_sequence(self):
        rewards = [-.01, -.21, .99]
        before = rewards.copy()
        score_rewards(rewards)
        self.assertEqual(rewards, before)
        args = train.parse_args([])
        self.assertEqual(args.gamma, .99)
        self.assertFalse(getattr(args, "discount_compensated_terminal_refund", False))

    def test_all_saved_curve_points_and_endpoint(self):
        points = load_curve(train.HERE / "release_0_80w", .01)
        self.assertEqual(points[-1]["total_steps"], 800058)
        self.assertEqual(len(points), 161)
        self.assertAlmostEqual(points[-1]["deadline_score"], .6, places=6)
        self.assertAlmostEqual(points[-1]["case_scores"]["closed_D15"], -1.2, places=6)

    def test_three_methods_keep_their_actual_endpoints(self):
        from .compare_deadline_scores import load_series
        series = load_series(include_continuations=False)
        self.assertEqual(series["episode"][-1]["total_steps"], 645145)
        self.assertEqual(series["step"][-1]["total_steps"], 643351)
        self.assertEqual(series["transformer"][-1]["total_steps"], 800058)
        self.assertAlmostEqual(series["episode"][-1]["deadline_score"], .6333333333, places=6)
        self.assertAlmostEqual(series["transformer"][-1]["deadline_score"], .6, places=6)
        for points in series.values():
            self.assertEqual(points[0]["total_steps"], 0)
            self.assertTrue(all(a["total_steps"] < b["total_steps"]
                                for a, b in zip(points, points[1:])))

    def test_completed_continuations_extend_without_replacing_history(self):
        from .compare_deadline_scores import load_series
        retained = load_series(include_continuations=False)
        extended = load_series()
        for method, endpoint, score in (("episode", 800027, -.35),
                                        ("step", 800003, .6)):
            self.assertEqual(extended[method][:len(retained[method])], retained[method])
            self.assertEqual(extended[method][-1]["total_steps"], endpoint)
            self.assertAlmostEqual(extended[method][-1]["deadline_score"], score, places=6)
            self.assertTrue(all(a["total_steps"] < b["total_steps"]
                                for a, b in zip(extended[method], extended[method][1:])))
        self.assertEqual(extended["transformer"], retained["transformer"])


if __name__ == "__main__":
    unittest.main()

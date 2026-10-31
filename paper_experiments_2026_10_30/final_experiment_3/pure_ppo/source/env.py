"""Timed Cartesian options for testing the high-level policy without MuJoCo.

This is an ideal kinematic substitute for a future frozen low-level policy.
Movement consumes integer low-level steps. It does not simulate robot joints,
collisions, physical grasping, or failures of a real low-level controller.
"""
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radial_layout import RadialLayout

LABELS = ("S", "A", "B", "C", "D")
ACTIONS = tuple(f"inspect_{x}" for x in LABELS[1:]) + tuple(f"deliver_{x}" for x in LABELS)
N_ACTIONS = len(ACTIONS)
OBS_DIM = 49


@dataclass(frozen=True)
class TaskConfig:
    deadlines: tuple = (30, 65, 150)
    episodes_per_trial: int = 3
    movement_m_per_step: float = 0.03
    inspect_steps: int = 2
    grasp_steps: int = 2
    release_steps: int = 2
    safe_score: float = 1.0
    unknown_scores: tuple = (0.0, 1.0, 2.0, 4.0)
    score_sampling: str = "permutation"
    score_scale: float = 4.0
    time_scale: float = 100.0


class PickPlaceTrial:
    def __init__(self, config=None, seed=None):
        self.config = config or TaskConfig()
        self.layout = RadialLayout()
        self.rng = np.random.default_rng(seed)
        self.finished = True

    def reset(self, deadline=None, *, offsets=None, scores=None):
        self.deadline = int(deadline if deadline is not None else self.rng.choice(self.config.deadlines))
        sampled = offsets if offsets is not None else self.layout.sample_ball_offsets(self.rng)
        self.initial_balls = np.array([
            [*sampled[label], self.layout.ball_radius + 0.001] for label in LABELS
        ], dtype=np.float64)
        self.balls = self.initial_balls.copy()
        self.plate = np.array([0.0, 0.0, 0.006 + self.layout.ball_radius])
        self.home = np.array([0.0, 0.0, self.layout.home_height_above_table])
        if scores is None:
            if self.config.score_sampling == "permutation":
                if len(self.config.unknown_scores) != 4:
                    raise ValueError("Permutation sampling requires four outer scores.")
                outer = self.rng.permutation(self.config.unknown_scores)
            elif self.config.score_sampling == "independent":
                outer = self.rng.choice(self.config.unknown_scores, size=4)
            else:
                raise ValueError(f"Unknown score sampling: {self.config.score_sampling}")
            scores = [self.config.safe_score, *outer]
        self._scores = np.array(scores, dtype=np.float64)
        self.episode = 0
        self.total_low_steps = 0
        self.finished = False
        self.records = []
        self.last_action = -1
        self.last_reward = 0.0
        self.outcome = -1
        self.feedback_id = -1
        self.feedback_score = 0.0
        self._reset_episode()
        return self.observation()

    def _reset_episode(self):
        self.eef = self.home.copy()
        self.balls[:] = self.initial_balls
        self.remaining = self.deadline
        self.boundary = True
        self.holding = -1
        self.episode_inspections = []
        self.episode_actions = []

    def observation(self):
        # Values are a one-event contact feedback, never a persistent answer table.
        feedback = np.zeros(5, np.float32)
        if self.feedback_id >= 0:
            feedback[self.feedback_id] = 1
        previous = np.zeros(N_ACTIONS, np.float32)
        if self.last_action >= 0:
            previous[self.last_action] = 1
        outcome = np.zeros(3, np.float32)
        if self.outcome >= 0:
            outcome[self.outcome] = 1
        holding = np.zeros(6, np.float32)
        holding[self.holding + 1] = 1
        obs = np.concatenate([
            self.eef / 0.5, self.plate / 0.5, self.balls.ravel() / 0.5,
            [self.remaining / self.config.time_scale,
             (self.config.episodes_per_trial - self.episode) / self.config.episodes_per_trial,
             float(self.boundary)],
            feedback, [self.feedback_score / self.config.score_scale],
            previous, [self.last_reward / self.config.score_scale], outcome, holding,
        ]).astype(np.float32)
        assert obs.shape == (OBS_DIM,)
        return obs

    def _travel(self, start, ball_index):
        target = self.initial_balls[ball_index].copy()
        target[2] += 0.005
        height = self.layout.home_height_above_table
        return [
            ("move", np.array([*start[:2], height])),
            ("move", np.array([*target[:2], height])),
            ("move", target),
            ("contact", ball_index),
        ]

    def plan(self, action, start=None):
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"Action must be in [0, {N_ACTIONS}).")
        start = self.eef if start is None else np.asarray(start)
        ball = action + 1 if action < 4 else action - 4
        segments = self._travel(start, ball)
        if action < 4:
            segments.append(("wait", self.config.inspect_steps))
        else:
            height = self.layout.home_height_above_table
            target = self.initial_balls[ball]
            segments.extend([
                ("wait", self.config.grasp_steps), ("grasp", ball),
                ("move", np.array([*target[:2], height])),
                ("move", np.array([*self.plate[:2], height])),
                ("move", self.plate + [0, 0, 0.005]),
                ("wait", self.config.release_steps), ("release", ball),
            ])
        return segments

    def duration(self, action, start=None):
        pos = self.eef.copy() if start is None else np.array(start, dtype=float)
        total = 0
        for kind, value in self.plan(action, pos):
            if kind == "move":
                total += int(np.ceil(max(0.0, np.linalg.norm(value - pos) - 1e-10) / self.config.movement_m_per_step))
                pos = value
            elif kind == "wait":
                total += value
        return total

    def step(self, action):
        if self.finished:
            raise RuntimeError("Trial finished; call reset() before taking another action.")
        action = int(action)
        self.boundary = False
        self.feedback_id = -1
        self.feedback_score = 0.0
        self.last_action = action
        self.last_reward = 0.0
        self.outcome = -1
        before = self.remaining
        completed = False
        self.episode_actions.append(ACTIONS[action])
        for kind, value in self.plan(action):
            if kind in ("move", "wait"):
                if kind == "move":
                    distance = np.linalg.norm(value - self.eef)
                    needed = int(np.ceil(max(0.0, distance - 1e-10) / self.config.movement_m_per_step))
                else:
                    needed = value
                used = min(needed, self.remaining)
                if kind == "move" and needed:
                    ratio = min(1.0, used * self.config.movement_m_per_step / max(distance, 1e-12))
                    self.eef += ratio * (value - self.eef)
                    if self.holding >= 0:
                        self.balls[self.holding] = self.eef - [0, 0, 0.005]
                self.remaining -= used
                if used < needed:
                    break
            elif kind == "contact":
                self.feedback_id = value
                self.feedback_score = float(self._scores[value])
            elif kind == "grasp":
                self.holding = value
            elif kind == "release":
                self.balls[value] = self.plate
                self.holding = -1
                self.last_reward = float(self._scores[value])
                completed = True
        spent = before - self.remaining
        self.total_low_steps += spent
        episode_done = completed or self.remaining == 0
        if action < 4:
            self.episode_inspections.append(LABELS[action + 1])
        self.outcome = 1 if completed else (2 if self.remaining == 0 else 0)
        info = dict(duration=spent, episode_done=episode_done)
        reward = self.last_reward / self.config.score_scale
        if episode_done:
            # Encode this terminal event before resetting the physical state.
            info["terminal_observation"] = self.observation().copy()
            record = dict(
                episode=self.episode + 1, deadline=self.deadline,
                score=self.last_reward, delivered=completed,
                ball=LABELS[action - 4] if completed else None,
                low_steps=self.deadline - self.remaining,
                inspections=list(self.episode_inspections), actions=list(self.episode_actions),
            )
            self.records.append(record)
            info["episode_record"] = record
            self.episode += 1
            self.finished = self.episode == self.config.episodes_per_trial
            if not self.finished:
                self._reset_episode()
                # Cross-episode information must travel through the meta memory.
                self.feedback_id = -1
                self.feedback_score = 0.0
                self.last_action = -1
                self.last_reward = 0.0
                self.outcome = -1
        return self.observation(), reward, self.finished, info

    def config_dict(self):
        return dict(task=asdict(self.config), layout=asdict(self.layout))


def calibration(samples=100, seed=123):
    env = PickPlaceTrial(seed=seed)
    near, far, full = [], [], []
    for _ in range(samples):
        env.reset()
        near.append(env.duration(4, env.home))
        far.extend(env.duration(4 + i, env.home) for i in range(1, 5))
        # A feasible fixed inspection order; worst case is fetching whichever
        # ball turns out best after the four inspections.
        best_worst = float("inf")
        for order in permutations(range(1, 5)):
            pos = env.home.copy()
            cost = 0
            for index in order:
                cost += env.duration(index - 1, pos)
                pos = env.initial_balls[index] + [0, 0, 0.005]
            worst = cost + max(env.duration(4 + i, pos) for i in range(5))
            best_worst = min(best_worst, worst)
        full.append(best_worst)
    return dict(
        samples=samples,
        direct_safe_steps=[min(near), max(near)],
        direct_outer_steps=[min(far), max(far)],
        inspect_all_then_deliver_worst_target_steps=[int(min(full)), int(max(full))],
    )

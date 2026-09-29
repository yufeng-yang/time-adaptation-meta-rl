"""Belief-state trial wrapper for ``ThreeRouteHiddenGateEnvV3``.

The hidden gate is sampled once per trial and never exposed in the policy
observation.  A deterministic Bayesian filter reads only the same local image
that the policy receives:

* before a door is visible: P(open) stays at its prior;
* after an open door is visible: P(open) = 1;
* after a closed door is visible: P(open) = 0.

Several base-environment episodes are joined into one Gymnasium episode.  This
is important: PPO can then assign value to information gathered in an early
episode because later episodes remain part of the same discounted return.
"""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Semantic local-map encoding used by the existing feed-forward baseline.
CELL_EMPTY = 0
CELL_WALL = 1
CELL_GOAL = 2
CELL_DOOR_OPEN = 3
CELL_DOOR_CLOSED = 4
CELL_AGENT = 5
N_CELL_TYPES = 6

# MiniGrid symbolic encoding.
OBJECT_WALL = 2
OBJECT_DOOR = 4
OBJECT_GOAL = 8
DOOR_OPEN = 0


class ExactBeliefFilter:
    """Exact filter for the deterministic open/closed door observation."""

    def __init__(self, prior_open: float = 0.5) -> None:
        if not 0.0 <= prior_open <= 1.0:
            raise ValueError("prior_open must be in [0, 1]")
        self.prior_open = float(prior_open)
        self.probability_open = float(prior_open)

    def reset(self) -> float:
        self.probability_open = self.prior_open
        return self.probability_open

    def update(self, observation: dict[str, Any]) -> float:
        """Update from the local image only; never inspect ``info``."""
        image = np.asarray(observation["image"], dtype=np.uint8)
        door_mask = image[..., 0] == OBJECT_DOOR
        if not np.any(door_mask):
            return self.probability_open

        door_states = image[..., 2][door_mask]
        observed_open = door_states == DOOR_OPEN
        if np.all(observed_open):
            self.probability_open = 1.0
        elif np.all(~observed_open):
            self.probability_open = 0.0
        else:
            raise RuntimeError("local observation contains inconsistent door states")
        return self.probability_open


class ExactBeliefTrialEnv(gym.Env[np.ndarray, int]):
    """Turn several hidden-gate episodes into one Bayes-adaptive trial.

    Policy observation layout::

        25 x 6 semantic cells
        + 4 direction one-hot
        + raw remaining time (D-t)
        + P(gate=open)
        + episode-boundary flag
        + number of episodes remaining, including the current one

    ``gate_state`` may appear in ``info`` for logging and evaluation, but the
    belief filter and policy observation never read it.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        base_env: gym.Env,
        *,
        deadlines: Sequence[int] = (15, 17, 19),
        episodes_per_trial: int = 5,
        prior_open: float = 0.5,
        fixed_gate_state: str | None = None,
    ) -> None:
        super().__init__()
        deadlines = tuple(int(value) for value in deadlines)
        if not deadlines or any(value <= 0 for value in deadlines):
            raise ValueError("deadlines must contain positive integers")
        if episodes_per_trial <= 0:
            raise ValueError("episodes_per_trial must be positive")
        if fixed_gate_state not in (None, "open", "closed"):
            raise ValueError("fixed_gate_state must be None, 'open', or 'closed'")

        self.base_env = base_env
        self.deadlines = deadlines
        self.episodes_per_trial = int(episodes_per_trial)
        self.prior_open = float(prior_open)
        # Curriculum control only.  This changes which task the environment
        # samples, but it is never copied into the policy observation/filter.
        self.fixed_gate_state = fixed_gate_state
        self.filter = ExactBeliefFilter(prior_open)
        self.action_space = base_env.action_space

        sensor_dim = 25 * N_CELL_TYPES
        observation_dim = sensor_dim + 4 + 4
        low = np.zeros(observation_dim, dtype=np.float32)
        high = np.ones(observation_dim, dtype=np.float32)
        # raw D-t and raw number of remaining episodes are not normalized.
        high[sensor_dim + 4] = np.inf
        high[-1] = float(self.episodes_per_trial)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._rng = np.random.default_rng()
        self._episode_index = 0
        self._episode_boundary = True
        self._fixed_deadline_sequence: tuple[int, ...] | None = None
        self._current_deadline = self.deadlines[0]
        self._trial_return = 0.0
        self._subepisode_return = 0.0
        self._subepisode_summaries: list[dict[str, Any]] = []

    @property
    def unwrapped(self):
        return self.base_env.unwrapped

    def _sample_deadline(self, episode_index: int) -> int:
        if self._fixed_deadline_sequence is not None:
            return self._fixed_deadline_sequence[episode_index]
        index = int(self._rng.integers(0, len(self.deadlines)))
        return self.deadlines[index]

    def _semantic_observation(self, observation: dict[str, Any]) -> np.ndarray:
        image = np.asarray(observation["image"], dtype=np.uint8)
        if image.shape[:2] != (5, 5):
            raise ValueError(f"expected a 5x5 local image, got {image.shape}")
        objects = image[..., 0]
        states = image[..., 2]

        cells = np.full((5, 5), CELL_EMPTY, dtype=np.int64)
        cells[objects == OBJECT_WALL] = CELL_WALL
        cells[objects == OBJECT_GOAL] = CELL_GOAL
        door_mask = objects == OBJECT_DOOR
        cells[door_mask & (states == DOOR_OPEN)] = CELL_DOOR_OPEN
        cells[door_mask & (states != DOOR_OPEN)] = CELL_DOOR_CLOSED
        cells[2, 2] = CELL_AGENT
        local_sensor = np.eye(N_CELL_TYPES, dtype=np.float32)[cells].reshape(-1)

        direction = np.zeros(4, dtype=np.float32)
        direction[int(observation["direction"])] = 1.0
        remaining_time = float(observation["time"][0])
        episodes_remaining = self.episodes_per_trial - self._episode_index
        auxiliary = np.asarray(
            [
                remaining_time,
                self.filter.probability_open,
                float(self._episode_boundary),
                float(episodes_remaining),
            ],
            dtype=np.float32,
        )
        return np.concatenate([local_sensor, direction, auxiliary]).astype(np.float32)

    def _augment_info(self, info: dict[str, Any], *, subepisode_done: bool) -> dict:
        result = dict(info)
        result.update(
            {
                "belief_open": float(self.filter.probability_open),
                "subepisode_done": bool(subepisode_done),
                "subepisode_index": int(self._episode_index),
                "episodes_per_trial": self.episodes_per_trial,
                "trial_return": float(self._trial_return),
            }
        )
        return result

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = dict(options or {})
        requested_sequence = options.pop("deadline_sequence", None)
        if requested_sequence is not None:
            sequence = tuple(int(value) for value in requested_sequence)
            if len(sequence) != self.episodes_per_trial:
                raise ValueError(
                    "deadline_sequence length must equal episodes_per_trial"
                )
            if any(value <= 0 for value in sequence):
                raise ValueError("deadline_sequence values must be positive")
            self._fixed_deadline_sequence = sequence
        elif "deadline" in options:
            deadline = int(options.pop("deadline"))
            self._fixed_deadline_sequence = (deadline,) * self.episodes_per_trial
        else:
            self._fixed_deadline_sequence = None

        self._episode_index = 0
        self._episode_boundary = True
        self._trial_return = 0.0
        self._subepisode_return = 0.0
        self._subepisode_summaries = []
        self.filter.reset()
        self._current_deadline = self._sample_deadline(self._episode_index)

        base_options: dict[str, Any] = {
            "new_trial": True,
            "deadline": self._current_deadline,
        }
        # Explicit gate_state is supported for evaluation.  A fixed stage gate
        # controls curriculum sampling without changing the b_0=prior input.
        requested_gate = options.get("gate_state", self.fixed_gate_state)
        if requested_gate is not None:
            base_options["gate_state"] = requested_gate
        observation, info = self.base_env.reset(seed=seed, options=base_options)
        self.filter.update(observation)
        return self._semantic_observation(observation), self._augment_info(
            info, subepisode_done=False
        )

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.base_env.step(
            int(action)
        )
        reward = float(reward)
        self._trial_return += reward
        self._subepisode_return += reward
        self._episode_boundary = False
        self.filter.update(observation)

        base_done = bool(terminated or truncated)
        if not base_done:
            return (
                self._semantic_observation(observation),
                reward,
                False,
                False,
                self._augment_info(info, subepisode_done=False),
            )

        summary = {
            "episode_index": int(self._episode_index),
            "deadline": int(self._current_deadline),
            "return": float(self._subepisode_return),
            "success": bool(info.get("success", False)),
            "timeout": bool(info.get("timeout", False)),
            "belief_open": float(self.filter.probability_open),
        }
        self._subepisode_summaries.append(summary)

        if self._episode_index + 1 >= self.episodes_per_trial:
            final_info = self._augment_info(info, subepisode_done=True)
            final_info["subepisode_return"] = float(self._subepisode_return)
            final_info["subepisodes"] = list(self._subepisode_summaries)
            return (
                self._semantic_observation(observation),
                reward,
                True,
                False,
                final_info,
            )

        self._episode_index += 1
        self._current_deadline = self._sample_deadline(self._episode_index)
        self._subepisode_return = 0.0
        self._episode_boundary = True
        next_observation, reset_info = self.base_env.reset(
            options={"new_trial": False, "deadline": self._current_deadline}
        )
        # Belief deliberately survives this reset and can also consume the new
        # legal observation if the door were ever visible from the start.
        self.filter.update(next_observation)
        next_info = self._augment_info(reset_info, subepisode_done=True)
        next_info["completed_subepisode"] = summary
        return self._semantic_observation(next_observation), reward, False, False, next_info

    def render(self):
        return self.base_env.render()

    def close(self) -> None:
        self.base_env.close()

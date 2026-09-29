"""Continuous baseline environment with local, ordinary gate lidar.

Purple walls and the yellow gate both use Safety-Gymnasium's regular
16-bin pseudo lidar.  They remain separate observation channels because they
are separate geom types.  Purple-wall lidar has the normal 3 metre range;
yellow-gate lidar has a 1 metre range.  No gate-state label is exposed.

The environment has a fixed internal episode limit of 1000 simulator steps.
Deadline and remaining time are deliberately omitted from the observation.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from continuous_pointnav_adaptation.envs.point_gate_goal import (
    PointThreeRouteGateEnv,
    PointThreeRouteGateTask,
)


FIXED_EPISODE_LIMIT = 1000
WALL_LIDAR_RANGE = 3.0
GATE_LIDAR_RANGE = 1.0
GATE_GEOM_GROUP = 3


class LocalGateLidarTask(PointThreeRouteGateTask):
    """Use a shorter range for the yellow gate's normal lidar channel."""

    def __init__(self, config: dict) -> None:
        super().__init__(config=config)
        # A distinct MuJoCo group lets _obs_lidar identify the gate channel.
        # It does not change whether the gate is collidable.
        self.yellow_gates.group = GATE_GEOM_GROUP

    def _obs_lidar(self, positions: np.ndarray | list, group: int) -> np.ndarray:
        if group != GATE_GEOM_GROUP:
            return super()._obs_lidar(positions, group)

        # Reuse exactly Safety-Gymnasium's ordinary pseudo-lidar calculation,
        # changing only its maximum distance for the yellow-gate channel.
        original_range = self.lidar_conf.max_dist
        self.lidar_conf.max_dist = GATE_LIDAR_RANGE
        try:
            return super()._obs_lidar(positions, group)
        finally:
            self.lidar_conf.max_dist = original_range


class NaiveGateEnv(PointThreeRouteGateEnv):
    """Fixed-1000-step task without deadline values in the observation."""

    def __init__(
        self,
        render_mode: str | None = None,
        *,
        gate_open_probability: float = 0.5,
        width: int = 512,
        height: int = 512,
        camera_name: str | None = "fixedfar",
    ) -> None:
        super().__init__(
            render_mode=render_mode,
            deadlines=(FIXED_EPISODE_LIMIT,),
            gate_open_probability=gate_open_probability,
            width=width,
            height=height,
            camera_name=camera_name,
        )

        # PointThreeRouteGateEnv normally appends remaining_steps and deadline.
        # This baseline intentionally exposes neither one.
        base_space = self.task.observation_space
        if not isinstance(base_space, spaces.Box):
            raise TypeError("expected a flattened Box observation")
        self._observation_space = spaces.Box(
            low=np.asarray(base_space.low, dtype=np.float64),
            high=np.asarray(base_space.high, dtype=np.float64),
            dtype=np.float64,
        )

    def _get_task(self) -> LocalGateLidarTask:
        task = LocalGateLidarTask(config=self.config)
        task.build_observation_space()
        return task

    def _augment_observation(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(observation, dtype=np.float64).reshape(-1)


class RewardOnlyGymWrapper(gym.Wrapper):
    """Convert Safety-Gymnasium's six-value API to Gymnasium's five values."""

    def step(self, action: np.ndarray):
        observation, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["safety_cost"] = float(cost)
        return observation, reward, terminated, truncated, info


def make_naive_env(**kwargs: Any) -> RewardOnlyGymWrapper:
    """Construct the SB3-compatible naive continuous environment."""

    return RewardOnlyGymWrapper(NaiveGateEnv(**kwargs))

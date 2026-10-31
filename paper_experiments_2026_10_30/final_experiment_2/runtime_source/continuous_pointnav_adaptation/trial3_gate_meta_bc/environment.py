"""Local ordinary gate lidar and explicit time for continuous trial Meta-RL."""
from __future__ import annotations

from typing import Any

import numpy as np

from continuous_pointnav_adaptation.envs.point_gate_goal import (
    GATE_LOCATIONS, PointThreeRouteGateEnv,
)
from continuous_pointnav_adaptation.naive_training.environment import (
    GATE_GEOM_GROUP, GATE_LIDAR_RANGE, LocalGateLidarTask,
)
from continuous_pointnav_adaptation.options import WaypointOptionController


DEADLINES = (195, 255, 275)
EPISODES_PER_TRIAL = 3


class ContinuousMetaGateEnv(PointThreeRouteGateEnv):
    """Keep the existing three-route geometry, time, and rendering support."""

    def _get_task(self) -> LocalGateLidarTask:
        task = LocalGateLidarTask(config=self.config)
        task.build_observation_space()
        return task


def make_env(*, render_mode: str | None = None, **kwargs: Any) -> ContinuousMetaGateEnv:
    return ContinuousMetaGateEnv(
        render_mode=render_mode, deadlines=DEADLINES,
        collision_penalty=0.1, **kwargs,
    )


def gate_in_sensing_range(env: ContinuousMetaGateEnv) -> bool:
    """Geometry-based diagnostic/expert trigger; never a policy input."""
    xy = np.asarray(env.task.agent.pos[:2], dtype=np.float64)
    distances = np.linalg.norm(np.asarray(GATE_LOCATIONS) - xy, axis=1)
    return bool(np.min(distances) < GATE_LIDAR_RANGE - 0.02)


def local_gate_reading(env: ContinuousMetaGateEnv) -> np.ndarray:
    """The same ordinary local lidar channel supplied to the policy."""
    return np.asarray(env.task._obs_lidar(env.task.yellow_gates.pos, GATE_GEOM_GROUP))


class DemonstrationController:
    """Waypoint teacher, used only to create full-trial training data.

    Before the first local reading, open and closed tasks use exactly the same
    direct controller. After sensing, an open gate keeps the direct route and
    a closed gate switches to the safe detour. Later episodes use the locally
    observed gate memory, rather than reading env.gate_state.
    """

    def __init__(self, deadline: int, remembered_gate: str | None = None):
        self.deadline = deadline
        self.observed_gate = remembered_gate
        self.controller = WaypointOptionController(self.route(remembered_gate))
        self.brake_remaining = 0

    def route(self, gate: str | None) -> str:
        if gate != "closed":
            return "direct"
        return "upper" if self.deadline == DEADLINES[1] else "lower"

    def action(self, env: ContinuousMetaGateEnv) -> np.ndarray:
        if self.observed_gate is None and gate_in_sensing_range(env):
            self.observed_gate = "closed" if np.any(local_gate_reading(env) > 0) else "open"
            # After investing in sensing, both detours miss the target budget;
            # choose the collision-free safe prefix instead of repeatedly
            # charging the closed gate or adding a danger penalty.
            route = "lower" if self.observed_gate == "closed" else "direct"
            self.controller = WaypointOptionController(route)
            if self.observed_gate == "closed":
                # The Point has inertia: cancel forward velocity before turning
                # toward the detour instead of drifting into the gate.
                self.brake_remaining = 8
                self.controller.waypoints = ((-2.0, 0.0),) + self.controller.waypoints
        if self.brake_remaining:
            self.brake_remaining -= 1
            return np.asarray([-1.0, 0.0], dtype=np.float64)
        return self.controller.action(env)

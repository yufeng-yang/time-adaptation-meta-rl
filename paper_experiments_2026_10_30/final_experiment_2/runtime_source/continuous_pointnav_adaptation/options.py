"""Frozen waypoint options for the first hierarchical continuous baseline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


ROUTE_NAMES = ("direct", "upper", "lower")
ROUTE_WAYPOINTS: dict[str, tuple[tuple[float, float], ...]] = {
    "direct": ((2.5, 0.0),),
    "upper": ((-0.75, 1.60), (0.75, 1.60), (2.5, 0.0)),
    "lower": ((-0.75, -1.90), (0.75, -1.90), (2.5, 0.0)),
}


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class WaypointOptionController:
    """Deterministic Point controller used as a frozen low-level option."""

    route: str
    waypoint_radius: float = 0.28
    turn_gain: float = 2.5

    def __post_init__(self) -> None:
        if self.route not in ROUTE_WAYPOINTS:
            raise ValueError(f"unknown route {self.route!r}")
        self.waypoints = ROUTE_WAYPOINTS[self.route]
        self.index = 0

    def action(self, env: Any) -> np.ndarray:
        position = np.asarray(env.task.agent.pos[:2], dtype=np.float64)
        while self.index + 1 < len(self.waypoints):
            waypoint = np.asarray(self.waypoints[self.index], dtype=np.float64)
            if np.linalg.norm(waypoint - position) > self.waypoint_radius:
                break
            self.index += 1

        target = np.asarray(self.waypoints[self.index], dtype=np.float64)
        desired_heading = math.atan2(
            float(target[1] - position[1]),
            float(target[0] - position[0]),
        )
        heading = float(env.task.agent.engine.data.qpos[2])
        error = _wrap_angle(desired_heading - heading)
        turn = float(np.clip(self.turn_gain * error, -1.0, 1.0))
        forward = float(np.clip(math.cos(error), 0.0, 1.0))
        return np.asarray([forward, turn], dtype=np.float64)


def execute_option(env: Any, route: str) -> dict[str, Any]:
    """Execute one complete option until success or the environment deadline."""

    controller = WaypointOptionController(route)
    total_reward = 0.0
    total_cost = 0.0
    actions: list[list[float]] = []
    positions = [np.asarray(env.task.agent.pos[:2], dtype=float).tolist()]
    final_info: dict[str, Any] = {}
    collision_events = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = controller.action(env)
        _, reward, cost, terminated, truncated, final_info = env.step(action)
        actions.append(action.tolist())
        positions.append(np.asarray(env.task.agent.pos[:2], dtype=float).tolist())
        total_reward += float(reward)
        total_cost += float(cost)
        collision_events += int(final_info.get("collision_started", False))
    return {
        "route": route,
        "success": bool(final_info.get("success", False)),
        "timeout": bool(final_info.get("timeout", False)),
        "return": total_reward,
        "cost": total_cost,
        "steps": len(actions),
        "danger_visits": int(final_info.get("danger_visits", 0)),
        "collision_events": collision_events,
        "actions": actions,
        "positions": positions,
        "final_info": final_info,
    }

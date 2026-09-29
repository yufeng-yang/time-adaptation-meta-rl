"""TimeGoalV2: a continuous two-route time-versus-danger environment.

The direct route is short but crosses a visible red danger strip.  The lower
route is longer and safe.  The former upper route is completely blocked by a
purple divider.  There is no gate and no hidden gate state.

Environmental perception contains exactly three ordinary lidar channels:
``goal_lidar``, ``walls_lidar`` and ``danger_lidar``.  Standard Point-agent
proprioceptive sensors are retained.  The flattened observation is augmented
with ``[remaining_steps, deadline]`` for time-aware experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
from safety_gymnasium.bases.base_object import Geom
from safety_gymnasium.builder import Builder
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0

from .point_gate_goal import BoxBlocks, PURPLE, RED


WALL_GROUP = 2
DANGER_GROUP = 3
DANGER_HALF_SIZE = 0.26
DANGER_LOCATIONS = ((0.0, -0.5), (0.0, 0.0), (0.0, 0.5))
START_POSITION = (-2.5, 0.0)
GOAL_POSITION = (2.5, 0.0)
START_GOAL_DISTANCE = 5.0


def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    count = int(round((stop - start) / step))
    for index in range(count + 1):
        yield round(start + index * step, 10)


def _unique(points: Iterable[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    return tuple(dict.fromkeys(points))


def _wall_locations() -> tuple[tuple[float, float], ...]:
    """Outer boundary and divider with only direct and lower openings."""

    spacing = 0.5
    points: list[tuple[float, float]] = []
    points.extend((x, -2.75) for x in _frange(-3.25, 3.25, spacing))
    points.extend((x, 2.75) for x in _frange(-3.25, 3.25, spacing))
    points.extend((-3.25, y) for y in _frange(-2.25, 2.25, spacing))
    points.extend((3.25, y) for y in _frange(-2.25, 2.25, spacing))

    # The direct opening contains the traversable danger strip.  The only
    # safe crossing is the lower opening.  In particular, the old upper
    # opening is now filled with wall blocks.
    for y in _frange(-2.5, 2.5, spacing):
        direct_opening = -0.5 <= y <= 0.5
        lower_opening = -2.5 <= y <= -2.0
        if not (direct_opening or lower_opening):
            points.append((0.0, y))
    return _unique(points)


WALL_LOCATIONS = _wall_locations()


@dataclass
class DangerStrip(BoxBlocks):
    """Visible, traversable hazard represented by ordinary lidar."""

    name: str = "danger"
    locations: tuple[tuple[float, float], ...] = DANGER_LOCATIONS
    half_size: float = DANGER_HALF_SIZE
    half_height: float = 0.01
    color: np.ndarray = field(default_factory=lambda: RED.copy())
    group: int = DANGER_GROUP
    is_lidar_observed: bool = True
    is_constrained: bool = False

    def process_config(self, config: dict, layout: dict, rots: Any) -> None:
        del layout, rots
        for index, xy_pos in enumerate(self.locations):
            block_name = f"danger{index}"
            body = self.get_config(xy_pos=np.asarray(xy_pos), rot=0.0)
            body["name"] = block_name
            body["geoms"][0]["name"] = block_name
            config[self.type][block_name] = body

    def get_config(self, xy_pos: np.ndarray, rot: float) -> dict:
        config = super().get_config(xy_pos, rot)
        geom = config["geoms"][0]
        geom["contype"] = 0
        geom["conaffinity"] = 0
        return config

    @property
    def pos(self) -> list[np.ndarray]:
        """Positions consumed by the ordinary Safety-Gymnasium lidar."""

        if self.engine is None:
            return []
        return [
            self.engine.data.body(f"danger{index}").xpos.copy()
            for index in range(self.num)
        ]


class TimeGoalV2Task(GoalLevel0):
    """Point goal task with a dangerous shortcut and safe lower detour."""

    START_POSITION = START_POSITION
    GOAL_POSITION = GOAL_POSITION

    def _add_geoms(self, *added_geoms: Geom) -> None:
        for geom in added_geoms:
            self._geoms[geom.name] = geom
            setattr(self, geom.name, geom)
            geom.set_agent(self.agent)

    def __init__(self, config: dict) -> None:
        super().__init__(config=config)
        self.placements_conf.extents = [-3.5, -3.0, 3.5, 3.0]
        self.floor_conf.size = (3.5, 3.0, 0.1)
        self.mechanism_conf.continue_goal = False

        self.agent.locations = [self.START_POSITION]
        self.agent.placements = None
        self.agent.keepout = 0.25
        self.agent.rot = 0.0
        self.goal.locations = [self.GOAL_POSITION]
        self.goal.placements = None
        self.goal.keepout = 0.30

        self._add_geoms(
            BoxBlocks(
                name="walls",
                locations=WALL_LOCATIONS,
                color=PURPLE.copy(),
                group=WALL_GROUP,
            ),
            DangerStrip(),
        )
        self.placements_conf.placements = None


class TimeGoalV2Env(Builder):
    """Safety-Gymnasium environment for the two-route TimeGoalV2 map."""

    DEFAULT_DEADLINES = (195, 255, 275)

    def __init__(
        self,
        render_mode: str | None = None,
        deadlines: tuple[int, ...] = DEFAULT_DEADLINES,
        step_penalty: float = 0.001,
        goal_reward: float = 1.0,
        collision_penalty: float = 0.1,
        timeout_penalty: float = 1.0,
        danger_penalty: float = 0.2,
        progress_reward_scale: float = 0.2,
        width: int = 512,
        height: int = 512,
        camera_name: str | None = "fixedfar",
    ) -> None:
        if not deadlines or any(int(value) <= 0 for value in deadlines):
            raise ValueError("deadlines must contain positive integers")
        if any(
            value < 0.0
            for value in (
                step_penalty,
                goal_reward,
                collision_penalty,
                timeout_penalty,
                danger_penalty,
                progress_reward_scale,
            )
        ):
            raise ValueError("reward magnitudes must be non-negative")

        self.deadlines = tuple(int(value) for value in deadlines)
        self.deadline = max(self.deadlines)
        self.step_penalty = float(step_penalty)
        self.goal_reward = float(goal_reward)
        self.collision_penalty = float(collision_penalty)
        self.timeout_penalty = float(timeout_penalty)
        self.danger_penalty = float(danger_penalty)
        self.progress_reward_scale = float(progress_reward_scale)
        self.danger_penalty_applied = False
        self.danger_visits = 0
        self._in_collision = False
        self._deadline_random, _ = seeding.np_random(None)

        super().__init__(
            task_id="SafetyPointTimeGoalV2-v0",
            config={
                "agent_name": "Point",
                "observation_flatten": True,
                "lidar_conf.max_dist": 3.0,
                "num_steps": self.deadline,
            },
            render_mode=render_mode,
            width=width,
            height=height,
            camera_name=camera_name,
        )

        base_space = self.task.observation_space
        if not isinstance(base_space, spaces.Box) or base_space.shape is None:
            raise TypeError("expected a flattened Safety-Gymnasium observation")
        self._observation_space = spaces.Box(
            low=np.concatenate([np.asarray(base_space.low).reshape(-1), np.zeros(2)]),
            high=np.concatenate(
                [np.asarray(base_space.high).reshape(-1), np.full(2, np.inf)]
            ),
            dtype=np.float64,
        )

    def _get_task(self) -> TimeGoalV2Task:
        task = TimeGoalV2Task(config=self.config)
        task.build_observation_space()
        return task

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    @property
    def remaining_steps(self) -> int:
        return max(0, int(self.deadline - (self.steps or 0)))

    def _augment_observation(self, observation: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(observation, dtype=np.float64).reshape(-1),
                np.asarray([self.remaining_steps, self.deadline], dtype=np.float64),
            ]
        )

    def _episode_info(self, info: dict) -> dict:
        enriched = dict(info)
        enriched.update(
            {
                "deadline": self.deadline,
                "remaining_steps": self.remaining_steps,
                "success": bool(info.get("goal_met", False)),
                "timeout": bool(self.truncated and not self.terminated),
                "danger_penalty_applied": self.danger_penalty_applied,
                "danger_visits": self.danger_visits,
                "progress_reward_scale": self.progress_reward_scale,
            }
        )
        return enriched

    def _touching_wall(self) -> bool:
        model = self.task.agent.engine.model
        data = self.task.agent.engine.data
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            first = model.geom(int(contact.geom1)).name or ""
            second = model.geom(int(contact.geom2)).name or ""
            if first.startswith("wall") or second.startswith("wall"):
                return True
        return False

    @staticmethod
    def _segment_intersects_square(
        start_xy: np.ndarray, end_xy: np.ndarray, center_xy: tuple[float, float]
    ) -> bool:
        center = np.asarray(center_xy, dtype=np.float64)
        lower, upper = center - DANGER_HALF_SIZE, center + DANGER_HALF_SIZE
        start = np.asarray(start_xy, dtype=np.float64)
        delta = np.asarray(end_xy, dtype=np.float64) - start
        t_min, t_max = 0.0, 1.0
        for axis in range(2):
            if abs(delta[axis]) < 1.0e-12:
                if start[axis] < lower[axis] or start[axis] > upper[axis]:
                    return False
                continue
            enter = (lower[axis] - start[axis]) / delta[axis]
            leave = (upper[axis] - start[axis]) / delta[axis]
            if enter > leave:
                enter, leave = leave, enter
            t_min = max(t_min, float(enter))
            t_max = min(t_max, float(leave))
            if t_min > t_max:
                return False
        return True

    @classmethod
    def _crosses_danger(cls, start_xy: np.ndarray, end_xy: np.ndarray) -> bool:
        return any(
            cls._segment_intersects_square(start_xy, end_xy, center)
            for center in DANGER_LOCATIONS
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        options = {} if options is None else dict(options)
        if seed is not None:
            self._deadline_random, _ = seeding.np_random(seed)
        requested_deadline = options.get("deadline")
        if requested_deadline is None:
            self.deadline = int(self._deadline_random.choice(self.deadlines))
        else:
            self.deadline = int(requested_deadline)
            if self.deadline <= 0:
                raise ValueError("deadline must be a positive integer")

        self.task.num_steps = self.deadline
        self.danger_penalty_applied = False
        self.danger_visits = 0
        self._in_collision = False
        observation, info = super().reset(seed=seed, options=options)
        return self._augment_observation(observation), self._episode_info(info)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, float, bool, bool, dict]:
        previous_xy = np.asarray(self.task.agent.pos[:2], dtype=np.float64).copy()
        goal_xy = np.asarray(GOAL_POSITION, dtype=np.float64)
        previous_goal_distance = float(np.linalg.norm(previous_xy - goal_xy))
        observation, native_reward, native_cost, terminated, truncated, info = super().step(
            action
        )
        collision = self._touching_wall()
        collision_started = collision and not self._in_collision
        self._in_collision = collision
        success = bool(info.get("goal_met", False))
        timeout = bool(truncated and not terminated)

        current_xy = np.asarray(self.task.agent.pos[:2], dtype=np.float64).copy()
        current_goal_distance = float(np.linalg.norm(current_xy - goal_xy))
        progress_reward = self.progress_reward_scale * (
            previous_goal_distance - current_goal_distance
        ) / START_GOAL_DISTANCE

        reward = -self.step_penalty + progress_reward
        if success:
            reward += self.goal_reward
        if collision_started:
            reward -= self.collision_penalty
        if timeout:
            reward -= self.timeout_penalty

        entered_danger = False
        if not self.danger_penalty_applied and self._crosses_danger(
            previous_xy, current_xy
        ):
            reward -= self.danger_penalty
            self.danger_penalty_applied = True
            self.danger_visits = 1
            entered_danger = True

        info = dict(info)
        info.update(
            {
                "entered_danger": entered_danger,
                "collision": collision,
                "collision_started": collision_started,
                "native_reward": float(native_reward),
                "native_cost": float(native_cost),
                "goal_distance": current_goal_distance,
                "progress_reward": progress_reward,
            }
        )
        return (
            self._augment_observation(observation),
            float(reward),
            0.0,
            bool(terminated),
            bool(truncated),
            self._episode_info(info),
        )


def make_time_goal_v2(**kwargs: Any) -> TimeGoalV2Env:
    return TimeGoalV2Env(**kwargs)

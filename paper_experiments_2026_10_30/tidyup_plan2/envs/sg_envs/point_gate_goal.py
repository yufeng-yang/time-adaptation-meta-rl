"""Continuous Point/Goal environment with a trial-persistent hidden gate.

The task is a small continuous analogue of ``ThreeRouteHiddenGateEnvV3``:

* the Safety-Gymnasium Point agent starts on the left;
* the standard Safety-Gymnasium Goal is fixed on the right;
* purple box geoms form the outer boundary and a central divider;
* yellow box geoms fill the direct opening only when the gate is closed;
* the upper and lower openings remain available as detours.

The gate state is fixed across episodes in a trial.  The native
Safety-Gymnasium six-value step API is preserved::

    obs, reward, cost, terminated, truncated, info = env.step(action)

The base Point observation is followed by ``[remaining_steps, deadline]``.
As in the discrete environment, a training wrapper may select only the first
time value if the absolute deadline should not be exposed to the policy.

Safety-Gymnasium and MuJoCo are optional project dependencies.  Install them
before importing this module::

    pip install safety-gymnasium mujoco
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


PURPLE = np.array([0.55, 0.16, 0.82, 1.0], dtype=np.float64)
YELLOW = np.array([1.0, 0.82, 0.05, 1.0], dtype=np.float64)
WALL_GROUP = 2


def _frange(start: float, stop: float, step: float) -> Iterable[float]:
    """Inclusive, numerically stable range for map construction."""

    count = int(round((stop - start) / step))
    for index in range(count + 1):
        yield round(start + index * step, 10)


def _unique(points: Iterable[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """Deduplicate wall blocks while preserving their construction order."""

    return tuple(dict.fromkeys(points))


def _purple_wall_locations() -> tuple[tuple[float, float], ...]:
    """Outer boundary plus a divider with direct/upper/lower openings."""

    spacing = 0.5
    points: list[tuple[float, float]] = []

    # Outer boundary, sized to fit Safety-Gymnasium's fixedfar camera.
    points.extend((x, -2.75) for x in _frange(-3.25, 3.25, spacing))
    points.extend((x, 2.75) for x in _frange(-3.25, 3.25, spacing))
    points.extend((-3.25, y) for y in _frange(-2.25, 2.25, spacing))
    points.extend((3.25, y) for y in _frange(-2.25, 2.25, spacing))

    # The divider has three openings.  The direct opening (-0.5 <= y <= 0.5)
    # is occupied by the yellow gate when closed.  The upper opening is the
    # shorter detour; the lower opening is intentionally farther away.
    for y in _frange(-2.5, 2.5, spacing):
        direct_opening = -0.5 <= y <= 0.5
        upper_opening = 1.25 <= y <= 2.25
        lower_opening = -2.5 <= y <= -2.0
        if not (direct_opening or upper_opening or lower_opening):
            points.append((0.0, y))

    return _unique(points)


PURPLE_WALL_LOCATIONS = _purple_wall_locations()
GATE_LOCATIONS = ((0.0, -0.5), (0.0, 0.0), (0.0, 0.5))


@dataclass
class BoxBlocks(Geom):
    """A set of fixed, collidable MuJoCo box geoms.

    Safety-Gymnasium's bundled ``Walls`` class is only a placeholder in the
    current release, so this small geom implements the required box config.
    """

    name: str = "box_blocks"
    locations: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    half_size: float = 0.26
    half_height: float = 0.30
    keepout: float = 0.0
    color: np.ndarray = field(default_factory=lambda: PURPLE.copy())
    group: int = WALL_GROUP
    is_lidar_observed: bool = True
    is_constrained: bool = False
    active: bool = True

    @property
    def num(self) -> int:
        return len(self.locations)

    def process_config(self, config: dict, layout: dict, rots: Any) -> None:
        """Add every active block directly to the MuJoCo world config."""

        del layout, rots
        if not self.active:
            return
        for index, xy_pos in enumerate(self.locations):
            block_name = f"{self.name[:-1]}{index}"
            body = self.get_config(xy_pos=np.asarray(xy_pos), rot=0.0)
            body["name"] = block_name
            body["geoms"][0]["name"] = block_name
            config[self.type][block_name] = body

    def get_config(self, xy_pos: np.ndarray, rot: float) -> dict:
        """Return one solid box body in Safety-Gymnasium's world format."""

        return {
            "name": self.name,
            "pos": np.array([xy_pos[0], xy_pos[1], self.half_height]),
            "rot": rot,
            "geoms": [
                {
                    "name": self.name,
                    "size": np.array(
                        [self.half_size, self.half_size, self.half_height]
                    ),
                    "type": "box",
                    "group": self.group,
                    "rgba": self.color.copy(),
                }
            ],
        }

    @property
    def pos(self) -> list[np.ndarray]:
        """Positions used by Safety-Gymnasium's pseudo lidar."""

        if not self.active:
            # Upstream pseudo lidar does not accept an empty position list.  A
            # point far beyond max_dist produces the intended all-zero reading.
            return [np.array([1.0e6, 1.0e6, 0.0])]
        if self.engine is None:
            return []
        return [
            self.engine.data.body(f"{self.name[:-1]}{index}").xpos.copy()
            for index in range(self.num)
        ]


class PointThreeRouteGateTask(GoalLevel0):
    """Goal task containing purple walls and a switchable yellow gate."""

    START_POSITION = (-2.5, 0.0)
    GOAL_POSITION = (2.5, 0.0)

    def _add_geoms(self, *added_geoms: Geom) -> None:
        """Register standard and local geom classes.

        Upstream restricts additions to a hard-coded type registry.  The local
        box class follows the same Geom protocol, so this task intentionally
        performs the three normal registration operations without that nominal
        type check.
        """

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
                name="purple_walls",
                locations=PURPLE_WALL_LOCATIONS,
                color=PURPLE.copy(),
            ),
            BoxBlocks(
                name="yellow_gates",
                locations=GATE_LOCATIONS,
                color=YELLOW.copy(),
            ),
        )

        # Rebuild placements because GoalLevel0 registered Goal before the map
        # objects and the object set is now complete.
        self.placements_conf.placements = None

    def set_gate_state(self, gate_state: str) -> None:
        """Make the yellow collision wall present only in the closed state."""

        self.yellow_gates.active = gate_state == "closed"


class PointThreeRouteGateEnv(Builder):
    """Safety-Gymnasium Builder with trial and deadline semantics."""

    GATE_STATES = ("open", "closed")
    DEFAULT_DEADLINES = (150, 200, 250)

    def __init__(
        self,
        render_mode: str | None = None,
        deadlines: tuple[int, ...] = DEFAULT_DEADLINES,
        gate_open_probability: float = 0.5,
        width: int = 512,
        height: int = 512,
        camera_name: str | None = "fixedfar",
    ) -> None:
        if not deadlines or any(int(value) <= 0 for value in deadlines):
            raise ValueError("deadlines must contain positive integers")
        if not 0.0 <= gate_open_probability <= 1.0:
            raise ValueError("gate_open_probability must be in [0, 1]")

        self.deadlines = tuple(int(value) for value in deadlines)
        self.gate_open_probability = float(gate_open_probability)
        self.deadline = max(self.deadlines)
        self.gate_state: str | None = None
        self.trial_id = -1
        self.episode_in_trial = -1
        self._gate_random, _ = seeding.np_random(None)

        super().__init__(
            task_id="SafetyPointThreeRouteGate-v0",
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
            raise TypeError("Expected a flat Safety-Gymnasium Box observation")
        self._observation_space = spaces.Box(
            low=np.concatenate(
                [np.asarray(base_space.low).reshape(-1), np.zeros(2)]
            ),
            high=np.concatenate(
                [
                    np.asarray(base_space.high).reshape(-1),
                    np.full(2, np.inf),
                ]
            ),
            dtype=np.float64,
        )

    def _get_task(self) -> PointThreeRouteGateTask:
        task = PointThreeRouteGateTask(config=self.config)
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
                np.asarray(
                    [self.remaining_steps, self.deadline], dtype=np.float64
                ),
            ]
        )

    def _episode_info(self, info: dict) -> dict:
        enriched = dict(info)
        enriched.update(
            {
                "deadline": self.deadline,
                "remaining_steps": self.remaining_steps,
                # Diagnostic/oracle information only; never part of observation.
                "gate_state": self.gate_state,
                "trial_id": self.trial_id,
                "episode_in_trial": self.episode_in_trial,
                "success": bool(info.get("goal_met", False)),
                "timeout": bool(self.truncated and not self.terminated),
            }
        )
        return enriched

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        options = {} if options is None else dict(options)
        if seed is not None:
            self._gate_random, _ = seeding.np_random(seed)

        requested_gate = options.get("gate_state")
        new_trial = bool(options.get("new_trial", self.gate_state is None))
        if requested_gate is not None:
            requested_gate = str(requested_gate)
            if requested_gate not in self.GATE_STATES:
                raise ValueError(
                    f"gate_state must be one of {self.GATE_STATES}, "
                    f"got {requested_gate!r}"
                )
            self.gate_state = requested_gate
            new_trial = True
        elif new_trial or self.gate_state is None:
            self.gate_state = (
                "open"
                if self._gate_random.random() < self.gate_open_probability
                else "closed"
            )

        if new_trial:
            self.trial_id += 1
            self.episode_in_trial = 0
        else:
            self.episode_in_trial += 1

        requested_deadline = options.get("deadline")
        if requested_deadline is None:
            index = int(self._gate_random.integers(0, len(self.deadlines)))
            self.deadline = self.deadlines[index]
        else:
            self.deadline = int(requested_deadline)
            if self.deadline <= 0:
                raise ValueError("deadline must be a positive integer")

        self.task.num_steps = self.deadline
        self.task.set_gate_state(self.gate_state)

        observation, info = super().reset(seed=seed, options=options)
        return self._augment_observation(observation), self._episode_info(info)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, float, bool, bool, dict]:
        observation, reward, cost, terminated, truncated, info = super().step(
            action
        )
        return (
            self._augment_observation(observation),
            float(reward),
            float(cost),
            bool(terminated),
            bool(truncated),
            self._episode_info(info),
        )


def make_point_gate_env(**kwargs: Any) -> PointThreeRouteGateEnv:
    """Construct the continuous hidden-gate environment."""

    return PointThreeRouteGateEnv(**kwargs)

"""Kinematic PointNav: 16 local directions, 0.2 m per discrete action.

MuJoCo is used ONLY for scene initialization, local observations and rendering.
Movement uses swept disk/box geometry, with no motor controller or mj_step.
One command costs one budget unit; this is a discrete action budget, not a
claim about physical execution time. Gymnasium five-value step API.
"""
from __future__ import annotations

import math

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from continuous_pointnav_adaptation.trial6_merged_lidar_transformer.environment import (
    MergedLidarGateEnv, OBSERVATION_SPEC as BASE_OBSERVATION_SPEC,
)


def wrap_angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def segment_box_distance_squared(start, end, lower, upper):
    """Exact squared distance from a 2D segment to an axis-aligned rectangle.

    Checking the entire segment prevents tunnelling and diagonal corner cuts.
    A disk sweep collides iff this distance is <= its squared radius.
    """
    delta = end - start
    enter, leave = 0., 1.
    intersects = True
    for axis in range(2):
        if abs(delta[axis]) < 1e-14:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                intersects = False
                break
        else:
            first = (lower[axis] - start[axis]) / delta[axis]
            last = (upper[axis] - start[axis]) / delta[axis]
            enter = max(enter, min(first, last))
            leave = min(leave, max(first, last))
            if enter > leave:
                intersects = False
                break
    if intersects:
        return 0.
    endpoint_distances = [np.sum((p - np.clip(p, lower, upper))**2) for p in (start, end)]
    length_squared = float(np.dot(delta, delta))
    corner_distances = []
    for x in (lower[0], upper[0]):
        for y in (lower[1], upper[1]):
            corner = np.array([x, y])
            t = float(np.clip(np.dot(corner - start, delta) / length_squared, 0., 1.)) if length_squared else 0.
            corner_distances.append(float(np.sum((start + t*delta - corner)**2)))
    return min(endpoint_distances + corner_distances)


def first_circle_entry(start, end, center, radius):
    delta, offset = end - start, start - center
    a = float(np.dot(delta, delta))
    c = float(np.dot(offset, offset) - radius**2)
    if c <= 0:
        return 0.
    if a < 1e-20:
        return None
    b = 2 * float(np.dot(offset, delta))
    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return None
    t = (-b - math.sqrt(discriminant)) / (2*a)
    return float(t) if 0 <= t <= 1 else None


class _ObservationScene(MergedLidarGateEnv):
    def __init__(self, *, time_scale, **kwargs):
        self.time_scale = time_scale
        super().__init__(**kwargs)

    def _augment_observation(self, observation):
        return np.concatenate([np.asarray(observation).reshape(-1),
                               [self.remaining_steps/self.time_scale]])


class DiscreteWaypointEnv(gym.Env):
    """Egocentric direction commands with continuous (unsnapped) positions.

    Action i targets 0.2 m at heading_start + i*22.5 degrees. Rotation is
    instantaneous and costs no EXTRA budget; each whole command costs 1.
    The body is approximated by a conservative 0.16 m disk, enclosing the
    original Point sphere and directional marker. A collision stops just short
    of contact; a blocked action still consumes 1. No automatic detour occurs.

    Observation: existing 60 sensor/lidar values plus normalized remaining
    action budget. Absolute coordinates and door labels are diagnostic only.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 8}

    def __init__(self, *, directions=16, distance=.2, robot_radius=.16,
                 deadlines=(50,), episodes_per_trial=3, time_scale=40.,
                 step_penalty=0., collision_penalty=.1, danger_penalty=.2,
                 goal_reward=1., timeout_penalty=1., render_mode=None, **scene_kwargs):
        super().__init__()
        if not isinstance(directions, int) or directions < 2:
            raise ValueError("directions must be an integer >= 2")
        if not deadlines or any(not isinstance(d, int) or d <= 0 for d in deadlines):
            raise ValueError("deadlines must be positive integers")
        if not isinstance(episodes_per_trial, int) or episodes_per_trial <= 0:
            raise ValueError("episodes_per_trial must be a positive integer")
        for name, value in dict(distance=distance, robot_radius=robot_radius, time_scale=time_scale).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for value in (step_penalty, collision_penalty, danger_penalty, goal_reward, timeout_penalty):
            if not math.isfinite(value) or value < 0:
                raise ValueError("reward magnitudes must be finite and nonnegative")
        self.directions, self.distance, self.robot_radius = directions, distance, robot_radius
        self.deadlines, self.episodes_per_trial = tuple(deadlines), episodes_per_trial
        self.time_scale, self.render_mode = time_scale, render_mode
        self.step_penalty, self.collision_penalty = step_penalty, collision_penalty
        self.danger_penalty, self.goal_reward, self.timeout_penalty = danger_penalty, goal_reward, timeout_penalty
        self.scene = self._make_scene(time_scale=time_scale, deadlines=self.deadlines,
                                      render_mode=render_mode, step_penalty=step_penalty,
                                      collision_penalty=collision_penalty, danger_penalty=danger_penalty,
                                      goal_reward=goal_reward, timeout_penalty=timeout_penalty,
                                      **scene_kwargs)
        self.action_space = spaces.Discrete(directions)
        self.observation_space = spaces.Box(self.scene.observation_space.low.astype(np.float32),
                                            self.scene.observation_space.high.astype(np.float32), dtype=np.float32)
        self._episode = -1
        self._done = True
        self._in_collision = False
        self._collision_events = 0
        self._boxes = []
        model = self.scene.task.agent.engine.model
        self._qpos = [int(model.joint(name).qposadr[0]) for name in ("x", "y", "z")]

    def _make_scene(self, **kwargs):
        return _ObservationScene(**kwargs)

    @property
    def observation_spec(self):
        return dict(BASE_OBSERVATION_SPEC, version="kinematic_local16_0p2_v1",
                    time_scale=self.time_scale, action_directions=self.directions,
                    action_distance=self.distance, coordinate_frame="egocentric",
                    robot_radius=self.robot_radius, dynamics="kinematic",
                    deadline_unit="discrete_commands", rotation="instantaneous")

    @property
    def remaining_steps(self):
        return self.scene.remaining_steps

    def get_wrapper_attr(self, name):
        # Compatibility with the project's SB3 2.7 / Gymnasium 0.28 pair.
        return getattr(self, name)

    def pose(self):
        """World position/heading for diagnostics, not policy observations."""
        data = self.scene.task.agent.engine.data
        return np.asarray(self.scene.task.agent.pos[:2]).copy(), float(data.qpos[self._qpos[2]])

    def _observation(self):
        # Forward kinematics and sensor calculation only; no integration.
        return self.scene._augment_observation(self.scene.task.obs()).astype(np.float32)

    def _scene_boxes(self):
        model, data = self.scene.task.agent.engine.model, self.scene.task.agent.engine.data
        boxes = []
        for index in range(model.ngeom):
            name = model.geom(index).name or ""
            if name.startswith(("purple_wall", "yellow_gate")):
                center = np.asarray(data.geom_xpos[index, :2]).copy()
                size = np.asarray(model.geom_size[index, :2]).copy()
                boxes.append((center-size, center+size))
        return boxes

    def _collides(self, start, end):
        # Broad phase avoids expensive exact distance checks against walls far
        # from this short command. Exact rounded-corner checks follow below.
        lower, upper = self._box_bounds
        candidates = (np.all(np.maximum(start, end) >= lower-self.robot_radius, axis=1)
                      & np.all(np.minimum(start, end) <= upper+self.robot_radius, axis=1))
        return any(segment_box_distance_squared(start, end, lower, upper)
                   <= self.robot_radius**2 for lower, upper in zip(lower[candidates], upper[candidates]))

    def _sweep(self, start, proposed):
        if not self._collides(start, proposed):
            return proposed.copy(), False
        low, high = 0., 1.
        for _ in range(25):
            fraction = (low+high)/2
            if self._collides(start, start+fraction*(proposed-start)):
                high = fraction
            else:
                low = fraction
        # Leave a tiny gap so the next command can move away from contact.
        fraction = max(0., low - 1e-5/self.distance)
        return start + fraction*(proposed-start), True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        opts = dict(options or {})
        new_trial = (self._episode < 0 or self._episode + 1 >= self.episodes_per_trial
                     or bool(opts.get("new_trial", False)) or "gate_state" in opts or seed is not None)
        if new_trial:
            self._episode = 0
        else:
            if "deadline" in opts and opts["deadline"] != self.scene.deadline:
                raise ValueError("deadline cannot change within a trial")
            self._episode += 1
        opts["new_trial"] = new_trial
        if not new_trial:
            opts["deadline"] = self.scene.deadline
        _, info = self.scene.reset(seed=seed, options=opts)
        self._boxes = self._scene_boxes()
        self._box_bounds = (np.array([b[0] for b in self._boxes]),
                            np.array([b[1] for b in self._boxes]))
        self._done = False
        self._in_collision = False
        self._collision_events = 0
        position, _ = self.pose()
        if self._collides(position, position):
            raise ValueError("robot footprint intersects a wall at reset")
        return self._observation(), dict(info, dynamics="kinematic", deadline_unit="discrete_commands")

    def step(self, action):
        if self._done:
            raise RuntimeError("reset() is required before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {self.directions})")
        start, heading = self.pose()
        direction = wrap_angle(heading + int(action)*2*math.pi/self.directions)
        proposed = start + self.distance*np.array([math.cos(direction), math.sin(direction)])
        end, collision = self._sweep(start, proposed)
        goal = np.asarray(self.scene.task.goal.pos[:2])
        entry = first_circle_entry(start, end, goal, self.scene.task.goal.size)
        success = entry is not None
        if success:
            end = start + entry*(end-start)
            # A later obstacle beyond the goal is never contacted.
            collision = self._collides(start, end)
        previous_danger = self.scene.danger_penalty_applied
        entered_danger = not previous_danger and self.scene._crosses_danger(start, end)
        if entered_danger:
            self.scene.danger_penalty_applied = True
            self.scene.danger_visits = 1
        collision_started = collision and not self._in_collision
        self._in_collision = collision
        self._collision_events += int(collision_started)
        data = self.scene.task.agent.engine.data
        data.qpos[self._qpos[:2]] += end-start
        data.qpos[self._qpos[2]] = direction
        data.qvel[:] = 0.
        data.ctrl[:] = 0.
        mujoco.mj_forward(self.scene.task.agent.engine.model, data)
        self.scene.steps += 1
        timeout = self.remaining_steps == 0 and not success
        self.scene.terminated = bool(success or timeout)
        self.scene.truncated = False
        self._done = self.scene.terminated
        reward = (-self.step_penalty + self.goal_reward*success - self.timeout_penalty*timeout
                  - self.collision_penalty*collision_started - self.danger_penalty*entered_danger)
        info = dict(action_index=int(action), start_xy=start.tolist(), end_xy=end.tolist(),
                    target_xy=proposed.tolist(), displacement=float(np.linalg.norm(end-start)),
                    target_reached=bool(np.linalg.norm(proposed-end) < 1e-6),
                    motion_end="success" if success else "deadline" if timeout else "collision" if collision else "target_reached",
                    duration_steps=1, success=bool(success), timeout=bool(timeout),
                    collision=bool(collision), collision_started=bool(collision_started),
                    collision_events=self._collision_events,
                    entered_danger=bool(entered_danger), danger_visits=self.scene.danger_visits,
                    deadline=self.scene.deadline, remaining_steps=self.remaining_steps,
                    episode_steps=self.scene.steps, trial_id=self.scene.trial_id,
                    episode_in_trial=self.scene.episode_in_trial,
                    gate_state=self.scene.gate_state, dynamics="kinematic",
                    deadline_unit="discrete_commands")
        return self._observation(), float(reward), self._done, False, info

    def render(self):
        return self.scene.render()

    def close(self):
        self.scene.close()


def make_env(**kwargs):
    return DiscreteWaypointEnv(**kwargs)

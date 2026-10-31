"""Original door/wall/goal sensing on the new kinematic motion interface."""
import numpy as np
import mujoco
from gymnasium import spaces

from continuous_pointnav_adaptation.naive_training.environment import LocalGateLidarTask
from continuous_pointnav_adaptation.discrete_waypoint_16.environment import (
    DiscreteWaypointEnv, _ObservationScene,
)

DEADLINES = (50,)
TIME_SCALE = 60.


class RadarScene(_ObservationScene):
    def _get_task(self):
        task = LocalGateLidarTask(config=self.config)
        task.build_observation_space()
        return task


class RadarWaypointEnv(DiscreteWaypointEnv):
    def __init__(self, *, absolute_xy=True, fixed_gate=None, **kwargs):
        self.absolute_xy = absolute_xy
        self.fixed_gate = fixed_gate
        self._cached_initial_qpos = None
        kwargs.setdefault("deadlines", DEADLINES)
        kwargs.setdefault("time_scale", TIME_SCALE)
        super().__init__(**kwargs)
        if absolute_xy:
            self.observation_space = spaces.Box(
                np.concatenate([self.observation_space.low[:-1], [-np.inf, -np.inf], [0.]]).astype(np.float32),
                np.concatenate([self.observation_space.high[:-1], [np.inf, np.inf], [np.inf]]).astype(np.float32),
                dtype=np.float32)
        offset = 0
        self.field_slices = {}
        for name, space in self.scene.task.obs_info.obs_space_dict.spaces.items():
            width = int(np.prod(space.shape))
            self.field_slices[name] = slice(offset, offset+width)
            offset += width
        assert offset == 60

    def _make_scene(self, **kwargs):
        return RadarScene(**kwargs)

    @property
    def observation_spec(self):
        return dict(version="kinematic_separate_radars_xy_v1" if self.absolute_xy else "kinematic_separate_radars_v1",
                    obs_dim=63 if self.absolute_xy else 61,
                    door_range=1., wall_range=3., goal_range=3., bins=16,
                    door_channel=True, merged_obstacles=False, danger_channel=False,
                    absolute_xy=self.absolute_xy, xy_input="raw world x,y before remaining time" if self.absolute_xy else None,
                    time_scale=self.time_scale, deadlines=list(self.deadlines),
                    directions=16, distance=.2, coordinate_frame="egocentric",
                    deadline_unit="discrete_commands", dynamics="kinematic",
                    body_sensors=12, pseudo_lidar=True)

    def _observation(self):
        obs = super()._observation()
        if not self.absolute_xy:
            return obs
        return np.concatenate([obs[:-1], self.pose()[0], obs[-1:]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        opts = dict(options or {})
        if self.fixed_gate is None:
            return super().reset(seed=seed, options=opts)
        if opts.get("gate_state", self.fixed_gate) != self.fixed_gate:
            raise ValueError("fixed-gate scene cache cannot switch gate state")
        if self._cached_initial_qpos is None:
            opts["gate_state"] = self.fixed_gate
            observation, info = super().reset(seed=seed, options=opts)
            self._cached_initial_qpos = self.scene.task.agent.engine.data.qpos.copy()
            return observation, info
        # Each collector owns one scene for each gate. The static map is never
        # modified, so repeated episodes only reset dynamic state. This avoids
        # compiling identical MuJoCo scenes thousands of times. No mj_step.
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        new_trial = (seed is not None or bool(opts.get("new_trial", False))
                     or "gate_state" in opts or self._episode+1 >= self.episodes_per_trial)
        if new_trial:
            self._episode = 0
            self.scene.trial_id += 1
            self.scene.deadline = int(opts.get("deadline", self.np_random.choice(self.deadlines)))
        else:
            if opts.get("deadline", self.scene.deadline) != self.scene.deadline:
                raise ValueError("deadline cannot change within a trial")
            self._episode += 1
        if self.scene.deadline <= 0:
            raise ValueError("deadline must be positive")
        self.scene.episode_in_trial = self._episode
        self.scene.task.num_steps = self.scene.deadline
        self.scene.steps = 0
        self.scene.terminated = self.scene.truncated = False
        self.scene.danger_penalty_applied = False
        self.scene.danger_visits = 0
        self.scene.cost = {"cost_sum": 0.}
        self._done = False
        self._in_collision = False
        self._collision_events = 0
        data = self.scene.task.agent.engine.data
        mujoco.mj_resetData(self.scene.task.agent.engine.model, data)
        data.qpos[:] = self._cached_initial_qpos
        mujoco.mj_forward(self.scene.task.agent.engine.model, data)
        info = self.scene._episode_info({})
        return self._observation(), dict(info, dynamics="kinematic", deadline_unit="discrete_commands")


def make_env(**kwargs):
    return RadarWaypointEnv(**kwargs)

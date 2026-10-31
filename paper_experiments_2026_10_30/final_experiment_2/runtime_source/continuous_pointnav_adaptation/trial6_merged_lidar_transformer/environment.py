"""2m merged obstacle lidar, 2m danger lidar, separate unchanged 3m goal lidar."""
from collections import OrderedDict

from gymnasium import spaces
from gymnasium.spaces.utils import flatten, flatten_space
import mujoco
import numpy as np

from continuous_pointnav_adaptation.envs.point_gate_goal import (
    PointThreeRouteGateTask, WALL_GROUP,
)
from continuous_pointnav_adaptation.trial4_gate_transformer_meta_rl.environment import (
    DEADLINES, EPISODES_PER_TRIAL, TIME_SCALE, RemainingTimeGateEnv,
)

OBSTACLE_RANGE=2.
DANGER_RANGE=2.
DANGER_GROUP=4
OBSERVATION_VERSION="merged_obstacle_2m_danger_2m_v1"
OBSERVATION_SPEC=dict(version=OBSERVATION_VERSION,obs_dim=61,goal_range=3.,
    obstacle_range=OBSTACLE_RANGE,danger_range=DANGER_RANGE,door_channel=False,
    bins=16,time_scale=TIME_SCALE,pseudo_lidar=True)


class MergedLidarTask(PointThreeRouteGateTask):
    def __init__(self,config):
        super().__init__(config=config)
        self.yellow_gates.is_lidar_observed=False
        self.danger_patches.is_lidar_observed=True
        self.danger_patches.group=DANGER_GROUP

    def build_observation_space(self):
        if self.observe_vision:
            raise ValueError("This vector observation version does not use RGB")
        fields=OrderedDict(self.agent.build_sensor_observation_space())
        for name in ("goal_lidar","obstacles_lidar","danger_lidar"):
            fields[name]=spaces.Box(0.,1.,shape=(self.lidar_conf.num_bins,),dtype=np.float64)
        self.obs_info.obs_space_dict=spaces.Dict(fields)
        self.observation_space=(flatten_space(self.obs_info.obs_space_dict)
                                if self.observation_flatten else self.obs_info.obs_space_dict)

    def _obs_lidar(self,positions,group):
        original_range=self.lidar_conf.max_dist
        if group==WALL_GROUP:
            # All solid obstacles share one observation; no door-only reading
            # is computed or passed to the actor/episode encoder.
            positions=list(self.purple_walls.pos)+list(self.yellow_gates.pos)
            self.lidar_conf.max_dist=OBSTACLE_RANGE
        elif group==DANGER_GROUP:
            self.lidar_conf.max_dist=DANGER_RANGE
        try:
            return super()._obs_lidar(positions,group)
        finally:
            self.lidar_conf.max_dist=original_range

    def obstacle_lidar(self):
        return self._obs_lidar(self.purple_walls.pos,WALL_GROUP)

    def obs(self):
        mujoco.mj_forward(self.model,self.data)
        result=self.agent.obs_sensor()
        result["goal_lidar"]=self._obs_lidar(self.goal.pos,self.goal.group)
        result["obstacles_lidar"]=self.obstacle_lidar()
        result["danger_lidar"]=self._obs_lidar(self.danger_patches.pos,DANGER_GROUP)
        assert self.obs_info.obs_space_dict.contains(result)
        return flatten(self.obs_info.obs_space_dict,result) if self.observation_flatten else result


class MergedLidarGateEnv(RemainingTimeGateEnv):
    def _get_task(self):
        task=MergedLidarTask(config=self.config)
        task.build_observation_space()
        return task


def make_env(**kwargs):
    kwargs.setdefault("deadlines",DEADLINES)
    kwargs.setdefault("collision_penalty",.1)
    return MergedLidarGateEnv(**kwargs)

"""Three high actions, frozen learned Point low actor, 0.2 s per command."""
import hashlib
import math
from pathlib import Path

from gymnasium import spaces
import mujoco
import numpy as np

from continuous_pointnav_adaptation.discrete_waypoint_16.environment import first_circle_entry, wrap_angle
from continuous_pointnav_adaptation.discrete_waypoint_16.six_case_calibrated.environment import (
    SixCaseEnv,SeparatedRouteScene,SeparatedRouteTask,wall_locations as original_wall_locations,
)
from continuous_pointnav_adaptation.lowlevel_heading_speed.policy import HeadingSpeedPolicy

HERE=Path(__file__).resolve().parent
LOW_CHECKPOINT=HERE/"frozen_low_actor.pt"
DEADLINES=(75,145,230)
TIME_SCALE=50.
COMMAND_DT=.2
STEP_PENALTY=.002
TURN_SIGNS=(1,0,-1)  # left, straight, right
LEFT_WALL_X=-4.75
MAP_VERSION='physical_wide_left_center_spawn_v3'
MAP_X_LIMITS=(-5.05,3.55)
# Midpoint between the left wall's inner face and the entry baffle's tip.
SPAWN_POSITION=((LEFT_WALL_X+.26-2.26)/2,0.)
CALIBRATION_DIR=HERE/'calibration_wide_left_v3_D145bonus002_D230normal'
FIRST_EPISODE_DANGER_BONUS=.02
FIRST_EPISODE_BONUS_DEADLINES=(145,)


def wall_locations():
    """Expand the left outer boundary; retain the interior map geometry."""
    points={(x,y) for x,y in original_wall_locations() if x!=-3.25}
    for x in np.arange(LEFT_WALL_X,-3.25+.001,.5):
        points.add((float(x),-5.75));points.add((float(x),3.75))
    for y in np.arange(-5.25,3.251,.5):
        points.add((LEFT_WALL_X,float(y)))
    return tuple(sorted(points))


class PhysicalTask(SeparatedRouteTask):
    START_POSITION=SPAWN_POSITION

    def __init__(self,config):
        super().__init__(config)
        self.purple_walls.locations=wall_locations()
        self.placements_conf.extents=[-5.05,-6.,3.55,4.]
        self.floor_conf.size=(5.1,6.1,.1)
        self.agent.rot=0.


class PhysicalScene(SeparatedRouteScene):
    def _get_task(self):
        task=PhysicalTask(self.config)
        task.build_observation_space()
        return task


class FixedTimeEnv(SixCaseEnv):
    def __init__(self, *, low_checkpoint=LOW_CHECKPOINT,
                 first_episode_danger_bonus=FIRST_EPISODE_DANGER_BONUS,
                 first_episode_bonus_deadlines=FIRST_EPISODE_BONUS_DEADLINES, **kwargs):
        # None reproduces checkpoints trained with a penalty in every episode.
        self.first_episode_danger_bonus=first_episode_danger_bonus
        # None keeps the legacy bonus in the first episode of every deadline.
        self.first_episode_bonus_deadlines=(None if first_episode_bonus_deadlines is None
                                           else tuple(first_episode_bonus_deadlines))
        self.low=HeadingSpeedPolicy(low_checkpoint)
        self.low_sha256=hashlib.sha256(Path(low_checkpoint).read_bytes()).hexdigest()
        kwargs.setdefault("deadlines",DEADLINES)
        kwargs.setdefault("time_scale",TIME_SCALE)
        kwargs.setdefault("step_penalty",STEP_PENALTY)
        super().__init__(**kwargs)
        self.action_space=spaces.Discrete(3)
        self.last_path=[]

    def _make_scene(self,**kwargs):return PhysicalScene(**kwargs)

    @property
    def observation_spec(self):
        return super().observation_spec | dict(version="physical_frozen_low_three_actions_v1",
            map_version=MAP_VERSION,left_wall_x=LEFT_WALL_X,upper_entry_clear_width=2.23,
            spawn_position=list(SPAWN_POSITION),
            reward_version=('all_episodes_danger_penalty_v1' if self.first_episode_danger_bonus is None
                else 'ep1_danger_bonus_v1' if self.first_episode_bonus_deadlines is None
                else 'deadline_specific_ep1_danger_bonus_v2'),
            first_episode_danger_bonus=self.first_episode_danger_bonus,
            first_episode_bonus_deadlines=(None if self.first_episode_bonus_deadlines is None else list(self.first_episode_bonus_deadlines)),
            danger_reward_frequency='first entry only, reset each episode',
            dynamics="MuJoCo implicit; learned frozen heading/speed actor",directions=None,distance=None,
            high_action_names=["left","straight","right"],action_dim=3,
            command_period=COMMAND_DT,max_turn_degrees=15.,target_speed=.4,
            action_target_speeds=[0.,.4,0.],turn_semantics="zero-speed turning with measured small drift",
            low_control_period=.01,physics_timestep=.002,integrator="implicit",
            low_sha256=self.low_sha256,deadline_unit="fixed_duration_commands",
            initial_heading_degrees=0.,
            time_input="remaining_commands/50 = remaining_seconds/10")

    def reset(self,**kwargs):
        obs,info=super().reset(**kwargs)
        model=self.scene.task.agent.engine.model
        model.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICIT
        camera=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,'fixedfar')
        if camera>=0:model.cam_pos[camera]=[-.75,-1.,14.]
        self.last_path=[]
        return obs,dict(info,dynamics="physical_frozen_low",deadline_unit="fixed_duration_commands")

    def step(self,action):
        if self._done:raise RuntimeError("reset required")
        if not self.action_space.contains(action):raise ValueError(action)
        agent=self.scene.task.agent
        model,data=agent.engine.model,agent.engine.data
        start,heading=self.pose()
        target_heading=heading+TURN_SIGNS[int(action)]*self.low.max_yaw_rate*COMMAND_DT
        started=float(data.time)
        success=False;entered_danger=False;collision_starts=0;any_collision=False
        path=[start.tolist()]
        goal=np.asarray(self.scene.task.goal.pos[:2])
        for _ in range(20):
            previous=agent.pos[:2].copy()
            data.ctrl[:]=self.low.action(agent,target_heading,.4 if int(action)==1 else 0.)
            mujoco.mj_step(model,data,nstep=5)
            mujoco.mj_forward(model,data)
            current=agent.pos[:2].copy()
            success |= first_circle_entry(previous,current,goal,self.scene.task.goal.size) is not None
            if not self.scene.danger_penalty_applied and self.scene._crosses_danger(previous,current):
                entered_danger=True;self.scene.danger_penalty_applied=True;self.scene.danger_visits=1
            collision=self.scene._touching_barrier()
            collision_starts+=int(collision and not self._in_collision)
            self._in_collision=collision;any_collision |= collision
            path.append(current.tolist())
        assert abs(float(data.time)-started-COMMAND_DT)<1e-8
        self.last_path=path
        self.scene.steps+=1
        self._collision_events+=collision_starts
        timeout=self.remaining_steps==0 and not success
        self._done=bool(success or timeout)
        self.scene.terminated,self.scene.truncated=self._done,False
        danger_reward=(self.first_episode_danger_bonus
                       if self.scene.episode_in_trial==0 and self.first_episode_danger_bonus is not None
                       and (self.first_episode_bonus_deadlines is None or self.scene.deadline in self.first_episode_bonus_deadlines)
                       else -self.danger_penalty) if entered_danger else 0.
        reward=(-self.step_penalty+self.goal_reward*success-self.timeout_penalty*timeout
                +danger_reward-self.collision_penalty*collision_starts)
        end,end_heading=self.pose()
        info=dict(success=bool(success),timeout=bool(timeout),collision=bool(any_collision),
            collision_started=bool(collision_starts),collision_events=self._collision_events,
            danger_visits=self.scene.danger_visits,entered_danger=entered_danger,
            danger_reward=float(danger_reward),
            action_index=int(action),action_name=self.observation_spec['high_action_names'][int(action)],
            start_xy=start.tolist(),end_xy=end.tolist(),start_heading=heading,end_heading=end_heading,
            target_heading=target_heading,heading_error=wrap_angle(target_heading-end_heading),
            speed=float(np.linalg.norm(agent.get_sensor("velocimeter")[:2])),
            duration_seconds=COMMAND_DT,physics_time=float(data.time),duration_steps=20,
            deadline=self.scene.deadline,remaining_steps=self.remaining_steps,
            episode_steps=self.scene.steps,trial_id=self.scene.trial_id,
            episode_in_trial=self.scene.episode_in_trial,gate_state=self.scene.gate_state,
            dynamics="physical_frozen_low",deadline_unit="fixed_duration_commands")
        return self._observation(),float(reward),self._done,False,info


def make_env(**kwargs):return FixedTimeEnv(**kwargs)

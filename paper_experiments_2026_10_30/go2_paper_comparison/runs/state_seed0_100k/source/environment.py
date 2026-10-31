"""Known near/far loading stations with fixed values and trial-persistent gates."""
from dataclasses import asdict
import numpy as np
from layout import Config

STATE_DIM = 19
STATE_FIELDS = ('base_body_x_over8','base_body_y_over8','fallback_body_x_over8','fallback_body_y_over8',
                'goal_body_x_over8','goal_body_y_over8',
                'sin_heading','cos_heading','body_vx_over06','body_vy_over04',
                'remaining_fraction','deadline_over120','carrying','carried_value_over4',
                'previous_normalized_vx','previous_normalized_vy','previous_reward',
                'episode_index_over2','terminal')


class GateTrial:
    def __init__(self, backend, config=Config(), seed=0):
        self.backend, self.config = backend, config
        self.rng = np.random.default_rng(seed)

    def reset(self, deadline=None, doors=None):
        self.deadline = float(self.rng.choice(self.config.deadlines) if deadline is None else deadline)
        if self.deadline <= 0 or abs(self.deadline/self.config.dt-round(self.deadline/self.config.dt))>1e-6:
            raise ValueError('Deadline must be a positive multiple of the high-level dt')
        self._doors = (self.rng.random(2)<self.config.gate_open_probability if doors is None
                       else np.asarray(doors,dtype=bool).copy())
        if self._doors.shape != (2,): raise ValueError('Expected two gate states')
        self.backend.set_task(self._doors)
        self.records = []
        self.episode = 0
        return self._reset_episode()

    def _reset_episode(self):
        self.backend.reset()
        self.elapsed = 0.
        self.carrying = False
        self.carried_value = 0.
        self.pickup_time = None
        self.previous_action = np.zeros(2)
        self.previous_reward = 0.
        self.terminal = False
        self.wall_contacts = 0
        return self.observe()

    def next_episode(self):
        if not self.terminal: raise RuntimeError('Current episode has not ended')
        if self.episode+1 >= self.config.episodes: raise RuntimeError('Trial finished; call reset for a new trial')
        self.episode += 1
        return self._reset_episode()

    @property
    def remaining(self): return max(0.,self.deadline-self.elapsed)

    def observe(self):
        pose = self.backend.pose
        c,s = np.cos(pose[2]),np.sin(pose[2])
        rotation = np.array([[c,-s],[s,c]])
        targets = (np.asarray([self.config.base,self.config.fallback_goal,self.config.goal])-pose[:2])@rotation/self.config.coordinate_scale
        velocity = np.asarray(self.backend.velocity)[:2]@rotation/np.array(self.config.command_limits[:2])
        state = np.r_[targets.ravel(),s,c,velocity,self.remaining/self.deadline,self.deadline/120.,
                      float(self.carrying),self.carried_value/self.config.goal_value,
                      self.previous_action,self.previous_reward,self.episode/2.,float(self.terminal)]
        assert state.shape == (STATE_DIM,)
        return dict(state=state.astype(np.float32),grid=self.backend.local_map()['grid'].copy())

    def step(self, action, *, advance_motion=True):
        if self.terminal: raise RuntimeError('Call next_episode or reset before stepping again')
        action = np.asarray(action,dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all(): raise ValueError('Expected finite normalized body vx/vy')
        action = np.clip(action,-1,1)
        if advance_motion:
            self.backend.step(np.r_[action*np.array(self.config.command_limits[:2]),0.],self.config.dt)
        self.elapsed = min(self.deadline,self.elapsed+self.config.dt)
        self.previous_action = action
        picked_up = False
        pickup_kind = None
        for kind,goal,value in [('fallback',self.config.fallback_goal,self.config.fallback_value),
                                 ('high',self.config.goal,self.config.goal_value)]:
            if value>self.carried_value and np.linalg.norm(self.backend.pose[:2]-goal)<=self.config.pickup_radius:
                self.carrying = True
                self.carried_value = value
                self.pickup_time = self.elapsed
                picked_up = True
                pickup_kind = kind
        failed = bool(self.backend.failed)
        delivered = bool(self.carrying and not failed and
                         np.linalg.norm(self.backend.pose[:2]-self.config.base)<=self.config.delivery_radius)
        timeout = self.remaining <= 1e-8
        self.terminal = bool(delivered or timeout or failed)
        self.wall_contacts += int(self.backend.wall_contact)
        reward = self.carried_value/self.config.goal_value if delivered else 0.
        self.previous_reward = reward
        obs = self.observe()
        info = dict(episode_done=self.terminal,trial_done=self.terminal and self.episode+1==self.config.episodes,
                    delivered=delivered,picked_up=picked_up,pickup_kind=pickup_kind,
                    delivered_value=self.carried_value if delivered else 0.,
                    motion_failed=failed,timeout=timeout and not delivered,
                    wall_contact=self.backend.wall_contact)
        if self.terminal:
            self.records.append(dict(episode=self.episode+1,delivered=delivered,reward=reward,
                delivered_value=self.carried_value if delivered else 0.,
                seconds=self.elapsed,pickup_time=self.pickup_time,motion_failed=failed,
                wall_contact_decisions=self.wall_contacts,final_position=self.backend.pose[:2].tolist()))
        return obs,reward,self.terminal,info

    def metadata(self):
        return dict(task=asdict(self.config),state_fields=STATE_FIELDS,
            state_dim=STATE_DIM,grid_shape=[3,self.config.grid_size,self.config.grid_size],
            observation='known base/goal relative coordinates + ideal odometry + local raycast voxel projection; no global map or gate labels',
            reward='delivered fixed value / 4: near=0.25, far=1; timeout/fall=0; one delivery per episode',
            cargo='Known fixed values 1 and 4; identical unloaded dynamics; logical loading; higher-value contact upgrades cargo, never downgrades it',
            trial='two independent Bernoulli gates sampled once; unchanged over 3 episodes; deadline fixed within trial',
            backend=self.backend.name,reset='step ends an episode; next_episode keeps hidden gates; reset starts a new trial')

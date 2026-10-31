"""Standalone learned policy and a fixed-time interface for a Point agent.

Runtime uses only the neural network. The BC teacher is never called here.
"""
import math
from pathlib import Path

import mujoco
import numpy as np
import torch


class HeadingSpeedPolicy:
    def __init__(self, checkpoint=None):
        checkpoint = Path(__file__).parent/"trained_actor.pt" if checkpoint is None else checkpoint
        self.checkpoint = Path(checkpoint)
        saved = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.config = {k:v for k,v in saved.items() if k != "actor_state_dict"}
        state = saved["actor_state_dict"]
        self.layers = [(state[k+".weight"].numpy(), state[k+".bias"].numpy()) for k in (
            "mlp_extractor.policy_net.0", "mlp_extractor.policy_net.2", "action_net")]
        self.action_scale = np.array(saved["action_scale"], np.float64)
        self.control_dt = saved["control_dt"]
        self.command_dt = saved["command_dt"]
        self.max_yaw_rate = saved["max_yaw_rate"]

    def normalized_action(self, observation):
        x = np.asarray(observation, np.float32)
        if x.shape != (6,):
            raise ValueError(f"expected 6-dimensional observation, got {x.shape}")
        for index,(weight,bias) in enumerate(self.layers):
            x=weight@x+bias
            if index<2:x=np.tanh(x)
        return np.clip(x,-1.,1.)

    @staticmethod
    def observation(agent, target_heading, speed=.4):
        heading=math.atan2(float(agent.mat[1,0]),float(agent.mat[0,0]))
        error=target_heading-heading
        velocity=agent.get_sensor("velocimeter")
        yaw=float(agent.get_sensor("gyro")[2])
        return np.array([math.cos(error),math.sin(error),velocity[0]/.4,
                         velocity[1]/.4,yaw/math.radians(75.),speed/.4],np.float32)

    def action(self, agent, target_heading, speed=.4):
        """Physical actuator controls; call at 100 Hz with a latched target."""
        return self.normalized_action(self.observation(agent,target_heading,speed))*self.action_scale

    def execute(self, agent, turn, *, speed=.4, period=None, record=True):
        """Run a complete physical macro command; turn=-1/right, 0, +1/left.

        Actual turning/velocity are learned tracking responses, not hard caps.
        Every command has exactly the requested duration, including when
        touching a wall. Task rewards/termination remain the outer task's job.
        """
        if turn not in (-1,0,1):raise ValueError("turn must be -1, 0, or +1")
        period=self.command_dt if period is None else period
        ticks=round(period/self.control_dt)
        if period<=0 or abs(period-ticks*self.control_dt)>1e-9:
            raise ValueError("period must be a positive multiple of control_dt")
        if period>.2+1e-9:raise ValueError("validated command periods are <=0.2 seconds")
        model,data=agent.engine.model,agent.engine.data
        if int(model.opt.integrator)!=int(mujoco.mjtIntegrator.mjINT_IMPLICIT):
            raise ValueError("use the same implicit integrator as training")
        substeps=round(self.control_dt/model.opt.timestep)
        if abs(substeps*model.opt.timestep-self.control_dt)>1e-9:
            raise ValueError("control period must contain an integer number of physics steps")
        heading=math.atan2(float(agent.mat[1,0]),float(agent.mat[0,0]))
        target_heading=heading+turn*self.max_yaw_rate*period
        started=float(data.time)
        trace=[]
        for tick in range(ticks):
            data.ctrl[:]=self.action(agent,target_heading,speed)
            mujoco.mj_step(model,data,nstep=substeps)
            mujoco.mj_forward(model,data)
            if record:
                trace.append(dict(time=float(data.time),xy=agent.pos[:2].copy().tolist(),
                    speed=float(np.linalg.norm(agent.get_sensor("velocimeter")[:2])),
                    heading=math.atan2(float(agent.mat[1,0]),float(agent.mat[0,0])),
                    control=data.ctrl.copy().tolist()))
        assert abs(float(data.time)-started-period)<1e-8
        return dict(target_heading=target_heading,target_speed=speed,turn=turn,
                    duration=float(data.time)-started,trace=trace)

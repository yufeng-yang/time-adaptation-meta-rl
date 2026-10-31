"""Radar-informed teacher. Gate labels never enter the learned observation."""
import math
from pathlib import Path
import numpy as np

from continuous_pointnav_adaptation.discrete_waypoint_16.environment import wrap_angle
from . import core
from .environment import DEADLINES,make_env,CALIBRATION_DIR,SPAWN_POSITION

CALIBRATION_FILE=CALIBRATION_DIR/'results.json'
PROBE_UPPER_RETURN=(-2.8,.25)
ROUTES={
    "direct":((2.5,0.),),
    "upper":((SPAWN_POSITION[0],1.75),(-.65,2.75),(.7,2.75),(2.5,0.)),
    "lower":((-.9,-4.75),(.8,-4.75),(2.5,0.)),
}


class DemonstrationTeacher:
    def __init__(self,deadline,remembered_gate=None,forced_route=None):
        self.deadline=deadline;self.observed_gate=remembered_gate
        self.first_probe=remembered_gate is None
        self.route=forced_route;self.index=0
        self.points=ROUTES[forced_route] if forced_route else None

    def action(self,env,observation):
        position,heading=env.pose()
        if self.observed_gate is None:
            if np.any(observation[env.field_slices['yellow_gates_lidar']]>1e-7):self.observed_gate='closed'
            elif np.linalg.norm(position)<.95:self.observed_gate='open'
        if self.observed_gate is None:
            target=np.array([-.9,0.])
        else:
            if self.route is None:
                self.route=('direct' if self.observed_gate=='open' else
                            'upper' if self.deadline==DEADLINES[1] else 'lower')
                self.points=ROUTES[self.route]
                if self.route=='upper' and self.first_probe:
                    # The horizontal baffle blocks a direct diagonal from the
                    # probing point. Retreat below its left tip before going up.
                    self.points=(PROBE_UPPER_RETURN,)+self.points
            while self.index+1<len(self.points) and np.linalg.norm(np.array(self.points[self.index])-position)<.24:
                self.index+=1
            target=np.array(self.points[self.index])
        desired=math.atan2(float(target[1]-position[1]),float(target[0]-position[0]))
        velocity=env.scene.task.agent.get_sensor('velocimeter')
        # Compensate measured sideslip; teacher only, no oracle dynamic state.
        if np.linalg.norm(velocity[:2])>.15:
            desired-=math.atan2(float(velocity[1]),max(.1,float(velocity[0])))
        error=wrap_angle(desired-heading)
        return 0 if error>math.radians(8.) else 2 if error<-math.radians(8.) else 1


def collect_trial(*args,**kwargs):
    return core.collect_trial(*args,**kwargs,teacher_factory=DemonstrationTeacher)


def expected_success(gate,deadline,episode):
    # Probe + safe upper retreat takes 196 commands; known-gate upper takes 135.
    return gate=='open' or (deadline==DEADLINES[-1]) or (deadline==DEADLINES[1] and episode>0)

"""Diagnostic waypoint controller; learns gate states from actual local rays.

Static geometry/waypoints are hand specified for calibration. This is neither
a trained policy nor a planner for unseen layouts. It never reads hidden gates.
"""
import numpy as np
from layout import Config, GATES, ROUTES, PROBE_X, EXIT_X
from simulation import goal_commands


def observe_gate(scan, index, max_range):
    box=GATES[index]
    points=scan['points'];distances=scan['distances']
    origin=scan['origins'];directions=scan['directions']
    blocked=(np.isfinite(distances) & (np.abs(points[:,0]-box.xy[0])<.2) &
             (np.abs(points[:,1]-box.xy[1])<.70) & (points[:,2]>.1)&(points[:,2]<1.2))
    if blocked.sum()>=3: return False
    # Positive evidence of a ray passing through the opening; missing returns
    # alone are not interpreted as an open gate.
    dx=directions[:,0]
    tau=(box.xy[0]-origin[0])/np.where(np.abs(dx)<1e-8,1e-8,dx)
    crossing=origin+tau[:,None]*directions
    limit=np.where(np.isfinite(distances),distances,max_range)
    clear=(dx>.05)&(tau>.1)&(tau<limit-.2)&(np.abs(crossing[:,1]-box.xy[1])<.55)&(crossing[:,2]>.15)&(crossing[:,2]<.8)
    return True if clear.sum()>=3 else None


class HistoryController:
    def __init__(self, config=Config()):
        self.config=config
        self.known=[None,None]

    def start_episode(self):
        self.returning=False
        self.events=[]
        if self.known[0] is not False: self.route='middle'
        elif self.known[1] is not False: self.route='upper'
        else: self.route='lower'
        self.queue=list(ROUTES[self.route])
        self.waits=0

    def _fallback(self, pose):
        self.route='fallback'
        self.queue=([(PROBE_X,-1.6)] if pose[0]>1.7 else [])+list(ROUTES['fallback'])

    def choose(self, pose, carrying, scan, elapsed=0., deadline=160.):
        # Value/time priors are calibrated static-route bounds for this expert,
        # not latent gate labels. The learned policy will receive only its obs.
        if elapsed==0.:
            # Use any evidence genuinely available within the limited range.
            for index in range(2):
                result=observe_gate(scan,index,self.config.sensor_range)
                if result is not None:
                    if self.known[index] is None:
                        self.events.append(dict(gate=index,opened=result,time=elapsed,position=pose[:2].tolist()))
                    self.known[index]=result
            self.route=('middle' if self.known[0] is not False else
                        'upper' if self.known[1] is not False else 'lower')
            self.queue=list(ROUTES[self.route])
            route_budget={'middle':26.,'upper':44.,'lower':61.}[self.route]
            if deadline<route_budget: self._fallback(pose)
        if carrying and not self.returning:
            self.returning=True
            self.queue=list(ROUTES[self.route][-2::-1])+[(0.,0.)]
        while self.queue:
            target=self.queue[0]
            tolerance=.4 if target in ((6.,0.),(0.,0.)) else .2
            if np.linalg.norm(pose[:2]-target)>=tolerance: break
            gate_index=0 if target==(PROBE_X,0.) else 1 if target==(PROBE_X,3.6) else None
            if not self.returning and gate_index is not None and self.known[gate_index] is None:
                result=observe_gate(scan,gate_index,self.config.sensor_range)
                if result is None:
                    self.waits+=1
                    if self.waits>4: raise RuntimeError('No definite gate evidence at the probe waypoint')
                    return np.zeros(2)
                self.known[gate_index]=result
                self.events.append(dict(gate=gate_index,opened=result,time=elapsed,position=pose[:2].tolist()))
                remaining=deadline-elapsed
                if result and remaining<({0:28.,1:35.}[gate_index]):
                    self._fallback(pose)
                    continue
                if not result:
                    if gate_index==0:
                        self.route='upper'
                        self.queue=[(PROBE_X,3.6),(EXIT_X,3.6),(6.,0.)]
                    else:
                        if remaining>=70.:
                            self.route='lower'
                            self.queue=[(PROBE_X,-5.2),(EXIT_X,-5.2),(6.,0.)]
                        else: self._fallback(pose)
                    continue
            self.queue.pop(0)
        if not self.queue: return np.zeros(2)
        command=goal_commands(np.asarray(pose)[None],np.asarray(self.queue[:1]))[0]
        return command[:2]/np.array([.6,.4])

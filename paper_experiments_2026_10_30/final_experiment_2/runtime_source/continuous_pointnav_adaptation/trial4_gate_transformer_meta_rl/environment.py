"""Reuse the existing three-route continuous environment without map changes.

No policy, waypoint execution, or new observation channels are introduced here.
The policy/training extension is separate from this environment preparation.
"""
import numpy as np
from gymnasium import spaces

from continuous_pointnav_adaptation.trial3_gate_meta_bc.environment import (
    ContinuousMetaGateEnv,
    EPISODES_PER_TRIAL,
)

# Keep the fixed normalization scale unchanged; D=500 is valid above 1 because
# remaining time is not clipped. Only the task's largest budget changes.
DEADLINES = (195, 255, 500)
TIME_SCALE = 275.


class RemainingTimeGateEnv(ContinuousMetaGateEnv):
    """Same physical map/reward; expose only (D-t)/275, not absolute D."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base = self.task.observation_space
        self._observation_space = spaces.Box(
            low=np.concatenate([base.low, [0.]]),
            high=np.concatenate([base.high, [np.inf]]), dtype=np.float64,
        )

    def _augment_observation(self, observation):
        return np.concatenate([np.asarray(observation, dtype=np.float64).reshape(-1),
                               [self.remaining_steps/TIME_SCALE]])


def make_env(**kwargs):
    kwargs.setdefault("deadlines", DEADLINES)
    kwargs.setdefault("collision_penalty", .1)
    return RemainingTimeGateEnv(**kwargs)


__all__ = ["RemainingTimeGateEnv", "DEADLINES", "EPISODES_PER_TRIAL", "TIME_SCALE", "make_env"]

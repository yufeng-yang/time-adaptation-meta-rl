"""Read-only access to the frozen continuous-navigation experiment."""
import os
from pathlib import Path
import sys
import json
import torch

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'final_experiment_2'
RUNTIME=ARCHIVE/'runtime_source'
sys.dont_write_bytecode=True
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(RUNTIME))

from continuous_pointnav_adaptation.hierarchical_fixedtime import core,scenario
from continuous_pointnav_adaptation.hierarchical_fixedtime.environment import make_env as archived_make_env


def archive_config(): return json.loads((ARCHIVE/'config.json').read_text())


def make_env(gate):
    cfg=archive_config()
    kwargs={k:cfg[k] for k in ('step_penalty','collision_penalty','danger_penalty','goal_reward','timeout_penalty',
                              'first_episode_danger_bonus','first_episode_bonus_deadlines')}
    kwargs.update(deadlines=tuple(cfg['deadlines']),time_scale=cfg['observation_spec']['time_scale'],
                  low_checkpoint=ARCHIVE/'frozen_low_actor.pt',fixed_gate=gate)
    env=archived_make_env(**kwargs)
    first=next(d for g,d in cfg['training_cases'] if g==gate)
    env.reset(seed=100000+cfg['seed'],options=dict(new_trial=True,gate_state=gate,deadline=first))
    return env


class AblationPolicy(core.TrialPolicy):
    def __init__(self,obs_dim=63,mode='full'):
        super().__init__(obs_dim)
        if mode not in ('state','time','context','full'):raise ValueError(mode)
        self.mode=mode

    def normalize(self,obs):
        result=super().normalize(obs)
        if self.mode in ('state','context'):
            result=result.clone();result[...,-1]=0
        return result

    def forward(self,obs,hidden):
        if self.mode in ('state','time'):hidden=torch.zeros_like(hidden)
        return super().forward(obs,hidden)

    def update_hidden(self,hidden,episode):
        if self.mode in ('state','time'):return torch.zeros_like(hidden)
        return super().update_hidden(hidden,episode)


def assert_frozen_imports():
    for name,module in list(sys.modules.items()):
        if name.startswith('continuous_pointnav_adaptation') and getattr(module,'__file__',None):
            assert Path(module.__file__).resolve().is_relative_to(RUNTIME),module.__file__

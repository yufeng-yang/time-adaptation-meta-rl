"""Read-only access to the archived nine-option pick-place experiment."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'final_experiment_3'
RUNTIME=ARCHIVE/'runtime_source/high_level'
sys.dont_write_bytecode=True
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.path.insert(0,str(RUNTIME))

import torch
import env as archived_env
import model as archived_model
import train as engine


class AblationPolicy(archived_model.TransformerMetaPolicy):
    def __init__(self,mode='full'):
        super().__init__()
        if mode not in ('state','time','context','full'):raise ValueError(mode)
        self.mode=mode

    def forward_episode(self,observations,lengths,hidden):
        if self.mode in ('state','context'):
            observations=observations.clone();observations[...,21]=0.
        if self.mode in ('state','time'):
            # A state-only/time-only actor must not retain within-episode
            # Transformer history either. Encode each current observation alone.
            batch,steps,dim=observations.shape
            flat=observations.reshape(batch*steps,1,dim)
            logits,values,_=super().forward_episode(flat,torch.ones(batch*steps,dtype=torch.long,device=lengths.device),
                                                   self.initial_hidden(batch*steps))
            return logits.reshape(batch,steps,-1),values.reshape(batch,steps),self.initial_hidden(batch)
        return super().forward_episode(observations,lengths,hidden)


def assert_frozen_imports():
    for module in (archived_env,archived_model,engine):
        assert Path(module.__file__).resolve().is_relative_to(RUNTIME),module.__file__

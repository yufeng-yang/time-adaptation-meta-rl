"""Matched input ablations; GRU updates remain at episode boundaries."""
import torch
from original_model import MetaPolicy as OriginalMetaPolicy, log_probability
class MetaPolicy(OriginalMetaPolicy):
    def __init__(self, width=64, action_dim=2, mode='full'):
        super().__init__(width,action_dim)
        self.mode=mode
    def encode(self, states, grids):
        if self.mode in ('state','context'):
            states=states.clone()
            states[...,10:12]=0
        return super().encode(states,grids)
    def forward_features(self,encoded,lengths,hidden):
        if self.mode not in ('state','time'):
            return super().forward_features(encoded,lengths,hidden)
        batch,steps,dim=encoded.shape
        mean,value,_=super().forward_features(encoded.reshape(batch*steps,1,dim),
            torch.ones(batch*steps,dtype=torch.long,device=lengths.device),
            self.initial_hidden(batch*steps))
        return mean.reshape(batch,steps,-1),value.reshape(batch,steps),self.initial_hidden(batch)

"""Causal within-episode Transformer; GRU updates only at episode boundaries."""
import torch
from torch import nn

from env import N_ACTIONS, OBS_DIM


class TransformerMetaPolicy(nn.Module):
    def __init__(self, width=64, hidden_dim=64, layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.obs_encoder = nn.Sequential(nn.Linear(OBS_DIM, width), nn.Tanh())
        block = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=width * 2,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            block, num_layers=layers, norm=nn.LayerNorm(width), enable_nested_tensor=False,
        )
        for layer in self.transformer.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.zeros_(layer.self_attn.in_proj_bias)
            for linear in (layer.self_attn.out_proj, layer.linear1, layer.linear2):
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
        self.event_attention = nn.Linear(width, 1)
        self.episode_encoder = nn.Sequential(nn.Linear(width * 2, width), nn.Tanh())
        self.meta_gru = nn.GRUCell(width, hidden_dim)
        self.body = nn.Sequential(nn.Linear(width + hidden_dim, 128), nn.Tanh())
        self.actor = nn.Linear(128, N_ACTIONS)
        self.critic = nn.Linear(128, 1)

    def initial_hidden(self, batch=1):
        return self.actor.weight.new_zeros(batch, self.hidden_dim)

    def forward_episode(self, observations, lengths, hidden):
        steps = observations.shape[1]
        valid = torch.arange(steps, device=observations.device)[None] < lengths[:, None]
        causal = torch.ones(steps, steps, dtype=torch.bool, device=observations.device).triu(1)
        contextual = self.transformer(
            self.obs_encoder(observations), mask=causal,
            src_key_padding_mask=~valid,
        )
        features = self.body(torch.cat([
            contextual, hidden[:, None].expand(-1, steps, -1),
        ], dim=-1))
        logits = self.actor(features)
        values = self.critic(features).squeeze(-1)
        weights = self.event_attention(contextual).squeeze(-1).masked_fill(~valid, -1e9).softmax(dim=1)
        attended = (weights[:, :, None] * contextual).sum(dim=1)
        strongest = contextual.masked_fill(~valid[:, :, None], -1e9).max(dim=1).values
        summary = self.episode_encoder(torch.cat([attended, strongest], dim=-1))
        next_hidden = self.meta_gru(summary, hidden)
        return logits, values, next_hidden

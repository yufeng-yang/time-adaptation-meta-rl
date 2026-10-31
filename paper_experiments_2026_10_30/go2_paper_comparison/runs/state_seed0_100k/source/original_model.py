"""2D CNN + causal episode Transformer + boundary-only GRU; velocity actor."""
import torch
from torch import nn
from torch.distributions import Normal
from environment import STATE_DIM


class MetaPolicy(nn.Module):
    def __init__(self, width=64, action_dim=2):
        super().__init__()
        self.width = width
        self.cnn = nn.Sequential(nn.Conv2d(3, 12, 3, 2, 1), nn.ReLU(),
                                 nn.Conv2d(12, 24, 3, 2, 1), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(),
                                 nn.Linear(216, 32), nn.Tanh())
        self.encoder = nn.Sequential(nn.Linear(32 + STATE_DIM, width), nn.Tanh())
        layer = nn.TransformerEncoderLayer(width, 4, 128, dropout=0.,
                                          activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2, nn.LayerNorm(width), enable_nested_tensor=False)
        self.attention = nn.Linear(width, 1)
        self.summary = nn.Sequential(nn.Linear(width * 2, width), nn.Tanh())
        self.gru = nn.GRUCell(width, width)
        self.body = nn.Sequential(nn.Linear(width * 2, 128), nn.Tanh())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.6))

    def initial_hidden(self, batch=1):
        return self.actor.weight.new_zeros(batch, self.width)

    def encode(self, states, grids):
        shape = states.shape[:-1]
        features = self.cnn(grids.reshape(-1, 3, grids.shape[-2], grids.shape[-1]))
        return self.encoder(torch.cat([states, features.reshape(*shape, 32)], dim=-1))

    def forward_features(self, encoded, lengths, hidden):
        steps = encoded.shape[1]
        valid = torch.arange(steps, device=encoded.device)[None] < lengths[:, None]
        causal = torch.ones(steps, steps, dtype=torch.bool, device=encoded.device).triu(1)
        context = self.transformer(encoded, mask=causal, src_key_padding_mask=~valid)
        features = self.body(torch.cat([context, hidden[:, None].expand(-1, steps, -1)], -1))
        weights = self.attention(context).squeeze(-1).masked_fill(~valid, -1e9).softmax(1)
        attended = (weights[..., None] * context).sum(1)
        strongest = context.masked_fill(~valid[..., None], -1e9).max(1).values
        next_hidden = self.gru(self.summary(torch.cat([attended, strongest], -1)), hidden)
        return self.actor(features), self.critic(features).squeeze(-1), next_hidden

    def forward_episode(self, states, grids, lengths, hidden):
        return self.forward_features(self.encode(states, grids), lengths, hidden)


def log_probability(mean, log_std, raw_action):
    """Log density for the exact tanh-transformed Gaussian used by rollout."""
    normal = Normal(mean, log_std.clamp(-3., 0.).exp())
    correction = 2 * (torch.log(torch.tensor(2., device=mean.device)) - raw_action - nn.functional.softplus(-2 * raw_action))
    return (normal.log_prob(raw_action) - correction).sum(-1)

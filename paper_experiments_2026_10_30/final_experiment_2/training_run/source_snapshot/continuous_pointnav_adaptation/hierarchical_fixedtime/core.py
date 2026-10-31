"""Completed-episode Transformer/GRU trial policy and information-safe teacher."""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from continuous_pointnav_adaptation.discrete_waypoint_16.environment import wrap_angle


class TrialPolicy(nn.Module):
    def __init__(self, obs_dim, hidden_dim=32, action_dim=3):
        super().__init__()
        self.obs_dim, self.hidden_dim = obs_dim, hidden_dim
        self.action_dim=action_dim
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_scale", torch.ones(obs_dim))
        self.transition_encoder = nn.Sequential(nn.Linear(2*obs_dim+action_dim+2, 128), nn.Tanh(),
                                                nn.Linear(128, 64), nn.Tanh())
        layer = nn.TransformerEncoderLayer(64, 4, dim_feedforward=128, dropout=0.,
                                          activation="gelu", batch_first=True, norm_first=True)
        self.trajectory_transformer = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(64),
                                                            enable_nested_tensor=False)
        for block in self.trajectory_transformer.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.zeros_(block.self_attn.in_proj_bias)
            for linear in (block.self_attn.out_proj, block.linear1, block.linear2):
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
        self.attention = nn.Linear(64, 1)
        self.episode_encoder = nn.Sequential(nn.Linear(128, 64), nn.Tanh())
        self.meta_gru = nn.GRUCell(64, hidden_dim)
        self.body = nn.Sequential(nn.Linear(obs_dim+hidden_dim, 256), nn.Tanh(),
                                  nn.Linear(256, 256), nn.Tanh())
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def normalize(self, obs):
        sensors = ((obs[..., :-1]-self.obs_mean[:-1])/self.obs_scale[:-1]).clamp(-5., 5.)
        return torch.cat([sensors, obs[..., -1:]], dim=-1)

    def initial_hidden(self):
        return torch.zeros(self.hidden_dim, device=self.obs_mean.device)

    def forward(self, obs, hidden):
        if obs.ndim == 2:
            hidden = hidden.unsqueeze(0).expand(obs.shape[0], -1)
        features = self.body(torch.cat([self.normalize(obs), hidden], dim=-1))
        return Categorical(logits=self.actor(features)), self.critic(features).squeeze(-1)

    def update_hidden(self, hidden, episode):
        data = tensors(episode)
        transitions = torch.cat([self.normalize(data[0]), F.one_hot(data[1], self.action_dim).float(),
                                  torch.stack([data[2], data[4]], dim=-1), self.normalize(data[3])], dim=-1)
        encoded = self.transition_encoder(transitions)
        contextual = self.trajectory_transformer(encoded.unsqueeze(0)).squeeze(0)
        weights = self.attention(contextual).squeeze(-1).softmax(0)
        pooled = torch.cat([(weights[:, None]*contextual).sum(0), contextual.max(0).values])
        return self.meta_gru(self.episode_encoder(pooled), hidden)


@dataclass
class Episode:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    old_logs: np.ndarray
    old_values: np.ndarray
    positions: np.ndarray
    success: bool
    route: str
    collisions: int
    danger_visits: int
    observed_gate: str | None

    @property
    def steps(self):
        return len(self.rewards)


@dataclass
class Trial:
    gate: str
    deadline: int
    episodes: list[Episode]


def tensors(episode):
    return (torch.as_tensor(episode.observations, dtype=torch.float32),
            torch.as_tensor(episode.actions, dtype=torch.long),
            torch.as_tensor(episode.rewards, dtype=torch.float32),
            torch.as_tensor(episode.next_observations, dtype=torch.float32),
            torch.as_tensor(episode.dones, dtype=torch.float32))


def route_name(positions, danger):
    positions = np.asarray(positions)
    crossings = np.flatnonzero((positions[:-1, 0] <= 0) & (positions[1:, 0] > 0))
    if len(crossings):
        y = positions[crossings[0]+1, 1]
        return "upper" if y > 1 else "lower" if y < -1 else "direct"
    return "upper" if danger else "lower" if positions[:, 1].min() < -1.2 else "other"


class DemonstrationTeacher:
    """Known static map/odometry, but no hidden gate label. Demos only.

    The first episode probes the central opening through the ordinary local
    door channel; later episodes use only that acquired memory. Closed first
    episodes always take a safe lower detour (or a safe timeout prefix).
    """
    def __init__(self, deadline, remembered_gate=None):
        self.deadline = deadline
        self.observed_gate = remembered_gate
        self.first_probe = remembered_gate is None
        self.route = None
        self.index = 0

    def action(self, env, observation):
        position, heading = env.pose()
        if self.observed_gate is None:
            door = observation[env.field_slices["yellow_gates_lidar"]]
            if np.any(door > 1e-7):
                self.observed_gate = "closed"
            elif np.linalg.norm(position) < .95:
                self.observed_gate = "open"
        if self.observed_gate is None:
            target = np.array([-.9, 0.])
        else:
            route = ("direct" if self.observed_gate == "open" else
                     "upper" if not self.first_probe and 31 <= self.deadline < 35 else "lower")
            if self.route is None:
                self.route = route
                self.points = {"direct": ((2.5, 0.),),
                               "upper": ((-.8, 1.75), (.8, 1.75), (2.5, 0.)),
                               "lower": ((-.8, -2.1), (.8, -2.1), (2.5, 0.))}[route]
            while self.index+1 < len(self.points) and np.linalg.norm(np.array(self.points[self.index])-position) < .12:
                self.index += 1
            target = np.array(self.points[self.index])
        delta = target-position
        relative = wrap_angle(math.atan2(float(delta[1]), float(delta[0]))-heading)
        return int(np.rint(relative/(2*math.pi/16))) % 16


@torch.no_grad()
def collect_trial(model, env, gate, deadline, seed, *, teacher=False, deterministic=False,
                  zero_memory=False, teacher_factory=None):
    if not teacher:
        torch.manual_seed(seed)
    hidden = model.initial_hidden() if model is not None else None
    remembered = None
    episodes = []
    trial_id = None
    for index in range(3):
        options = dict(new_trial=index == 0, deadline=deadline)
        if index == 0:
            options["gate_state"] = gate
        obs, info = env.reset(seed=seed if index == 0 else None, options=options)
        if trial_id is None:
            trial_id = info["trial_id"]
        assert info["trial_id"] == trial_id and info["gate_state"] == gate and info["deadline"] == deadline
        assert info["episode_in_trial"] == index and env.remaining_steps == deadline
        factory = teacher_factory or DemonstrationTeacher
        control = factory(deadline, remembered) if teacher else None
        observations, actions, rewards, next_observations, dones, logs, values = [], [], [], [], [], [], []
        positions = [env.pose()[0].copy()]
        while True:
            if teacher:
                action = control.action(env, obs)
                log = value = 0.
            else:
                distribution, value_tensor = model(torch.as_tensor(obs), hidden)
                sample = distribution.logits.argmax() if deterministic else distribution.sample()
                action, log, value = int(sample), float(distribution.log_prob(sample)), float(value_tensor)
            nxt, reward, term, trunc, info = env.step(action)
            observations.append(obs.copy())
            actions.append(action)
            rewards.append(reward)
            next_observations.append(nxt.copy())
            dones.append(float(term or trunc))
            logs.append(log)
            values.append(value)
            positions.append(env.pose()[0].copy())
            obs = nxt
            if term or trunc:
                break
        arrays = [np.asarray(x, np.float32) for x in (observations, rewards, next_observations, dones, logs, values, positions)]
        episode = Episode(arrays[0], np.asarray(actions, np.int64), *arrays[1:],
                          bool(info["success"]), route_name(positions, info["danger_visits"]),
                          info["collision_events"], info["danger_visits"],
                          control.observed_gate if teacher else None)
        episodes.append(episode)
        if teacher:
            remembered = control.observed_gate
            assert remembered is not None, "teacher failed to obtain local door evidence"
        if model is not None and index < 2:
            hidden = model.update_hidden(hidden, episode) if not zero_memory else model.initial_hidden()
    return Trial(gate, deadline, episodes)


def bc_step_weights(actions, critical_weight=1., turn_window=0):
    """Weight expert turns and their neighboring steps within one episode."""
    if not math.isfinite(critical_weight) or critical_weight<=0:
        raise ValueError('critical_weight must be finite and positive')
    if not isinstance(turn_window,int) or turn_window<0:
        raise ValueError('turn_window must be a nonnegative integer')
    turns=actions!=1
    critical=turns.clone()
    for offset in range(1,min(turn_window,len(actions)-1)+1):
        critical[offset:] |= turns[:-offset]
        critical[:-offset] |= turns[offset:]
    return torch.where(critical, float(critical_weight), 1.).float()


def demonstration_loss(model, trials, *, critical_weight=1., turn_window=0):
    losses, weights = [], []
    for trial in trials:
        hidden = model.initial_hidden()
        for index, episode in enumerate(trial.episodes):
            data = tensors(episode)
            distribution, _ = model(data[0], hidden)
            losses.append(-distribution.log_prob(data[1]))
            weights.append(bc_step_weights(data[1],critical_weight,turn_window))
            if index < 2:
                hidden = model.update_hidden(hidden, episode)
    losses,weights=torch.cat(losses),torch.cat(weights)
    return (losses*weights).sum()/weights.sum()


def discounted(rewards, gamma):
    result = np.empty(len(rewards), np.float32)
    running = 0.
    for index in range(len(rewards)-1, -1, -1):
        running = float(rewards[index])+gamma*running
        result[index] = running
    return result


def bc_weight(steps, args):
    floor = getattr(args, "bc_min_weight", 0.)
    return floor+(args.bc_initial_weight-floor)*max(0., 1.-steps/args.bc_decay_steps)


def episode_reward_weights(objective, count=3, weights=(.2, .5, 1.)):
    if objective == "all_episodes":
        return (1.,)*count
    if objective == "last_episode":
        return (0.,)*(count-1)+(1.,)
    if objective != "weighted_episodes":
        raise ValueError(f"unknown return objective: {objective}")
    weights = tuple(float(value) for value in weights)
    if len(weights) != count or any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
        raise ValueError("episode weights must match the trial length, be finite/nonnegative and not all zero")
    return weights


def trial_targets(trials, gamma, objective="all_episodes", weights=(.2, .5, 1.)):
    targets = []
    for trial in trials:
        # Weight only the optimization rewards. Episode objects retain their raw
        # rewards for trajectory memory, BC and evaluation. Earlier actions
        # still receive the last episode's return across episode boundaries.
        factors = episode_reward_weights(objective, len(trial.episodes), weights)
        rewards = [episode.rewards*factor for episode, factor in zip(trial.episodes, factors)]
        targets.append(discounted(np.concatenate(rewards), gamma))
    return np.concatenate(targets)


def update(model, optimizer, trials, demonstrations, steps, args):
    target = torch.as_tensor(trial_targets(trials, args.gamma, getattr(args, "return_objective", "all_episodes"),
                                          getattr(args, "episode_weights", (.2, .5, 1.))))
    old_values = torch.as_tensor(np.concatenate([e.old_values for t in trials for e in t.episodes]))
    old_logs = torch.as_tensor(np.concatenate([e.old_logs for t in trials for e in t.episodes]))
    advantage = target-old_values
    advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    weight = bc_weight(steps, args)
    for _ in range(args.ppo_epochs):
        logs, values, entropies = [], [], []
        for trial in trials:
            hidden = model.initial_hidden()
            for index, episode in enumerate(trial.episodes):
                data = tensors(episode)
                distribution, value = model(data[0], hidden)
                logs.append(distribution.log_prob(data[1]))
                values.append(value)
                entropies.append(distribution.entropy())
                if index < 2:
                    hidden = model.update_hidden(hidden, episode)
        ratio = (torch.cat(logs)-old_logs).exp()
        actor = -torch.minimum(ratio*advantage,
                               ratio.clamp(1-args.clip_range, 1+args.clip_range)*advantage).mean()
        critic = .5*(torch.cat(values)-target).square().mean()
        entropy = torch.cat(entropies).mean()
        bc = demonstration_loss(model, demonstrations,
            critical_weight=getattr(args,'bc_critical_weight',1.),
            turn_window=getattr(args,'bc_turn_window',0)) if weight else torch.zeros(())
        loss = actor+args.value_coef*critic-args.entropy_coef*entropy+weight*bc
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite PPO/BC loss")
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("non-finite gradient")
        optimizer.step()
        metrics = dict(loss=float(loss.detach()), actor_loss=float(actor.detach()),
                       critic_loss=float(critic.detach()), entropy=float(entropy.detach()),
                       bc_loss=float(bc.detach()), bc_weight=weight,
                       clip_fraction=float(((ratio-1).abs()>args.clip_range).float().mean().detach()),
                       grad_norm=float(grad_norm))
    return metrics


def episode_records(trial):
    return [dict(gate=trial.gate, deadline=trial.deadline, episode=i+1,
                 success=e.success, route=e.route, steps=e.steps, return_=float(e.rewards.sum()),
                 collision_events=e.collisions, danger_visits=e.danger_visits,
                 observed_gate=e.observed_gate) for i, e in enumerate(trial.episodes)]

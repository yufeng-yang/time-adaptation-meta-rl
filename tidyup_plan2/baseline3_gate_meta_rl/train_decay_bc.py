"""Time-conditioned trial PPO with a decaying behavior-cloning auxiliary.

The environment resets three times per trial while gate and deadline stay fixed.
All episodes use the same reward. PPO optimizes a hierarchical trial return, so
later rewards assign credit to information-gathering actions in earlier episodes.

Six scripted demonstration trials (open/closed x D={15,17,19}) teach route
execution and trial-level adaptation without exposing the true gate state to the
policy.  PPO remains the main objective and the BC coefficient decays to zero.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for path in (ROOT, ROOT / "Minigrid", HERE.parent, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tidyup_plan2.envs import ThreeRouteHiddenGateEnvV4

DEADLINES = (15, 17, 19)
EPISODES_PER_TRIAL = 3
VIEW_SIZE, N_CELL_TYPES, N_ACTIONS = 5, 6, 3
STATE_DIM = VIEW_SIZE * VIEW_SIZE * N_CELL_TYPES + 4
OBS_DIM = STATE_DIM + 1


class TimeConditionedHiddenGateObservation(gym.ObservationWrapper):
    """Local state plus D-t; open/closed doors are distinct only in view."""

    def __init__(self, env: gym.Env, time_cap: float = 20.0) -> None:
        super().__init__(env)
        self.time_cap = float(time_cap)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(OBS_DIM, np.float32),
            high=np.ones(OBS_DIM, np.float32),
            dtype=np.float32,
        )

    def observation(self, observation: dict[str, Any]) -> np.ndarray:
        image = np.asarray(observation["image"], np.uint8)
        objects = image[..., 0]
        cells = np.zeros((VIEW_SIZE, VIEW_SIZE), np.int64)
        cells[objects == 2] = 1
        cells[objects == 8] = 2
        cells[(objects == 4) & (image[..., 2] != 0)] = 3
        cells[(objects == 4) & (image[..., 2] == 0)] = 4
        cells[VIEW_SIZE // 2, VIEW_SIZE // 2] = 5
        semantic = np.eye(N_CELL_TYPES, dtype=np.float32)[cells].reshape(-1)
        direction = np.zeros(4, np.float32)
        direction[int(observation["direction"])] = 1.0
        remaining = float(np.asarray(observation["time"])[0])
        time_feature = np.asarray(
            [min(remaining, self.time_cap) / self.time_cap], np.float32
        )
        return np.concatenate([semantic, direction, time_feature]).astype(np.float32)


def visible_gate_state(observation: np.ndarray) -> str | None:
    """Read gate evidence from the policy's local observation, never env info."""
    cells = observation[:VIEW_SIZE * VIEW_SIZE * N_CELL_TYPES].reshape(
        VIEW_SIZE, VIEW_SIZE, N_CELL_TYPES
    )
    if np.any(cells[..., 3]):
        return "closed"
    if np.any(cells[..., 4]):
        return "open"
    return None


def make_env(args: argparse.Namespace) -> gym.Env:
    return TimeConditionedHiddenGateObservation(
        ThreeRouteHiddenGateEnvV4(
            min_deadline=min(DEADLINES),
            max_deadline=500,
            step_penalty=args.step_penalty,
            progress_scale=0.0,
        ),
        time_cap=args.time_cap,
    )


class TrialActorCritic(nn.Module):
    """One shared controller and a completed-episode context encoder."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        transition_dim = OBS_DIM * 2 + N_ACTIONS + 2
        self.transition_encoder = nn.Sequential(
            nn.Linear(transition_dim, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
        )
        self.event_attention = nn.Linear(64, 1)
        self.episode_encoder = nn.Sequential(nn.Linear(128, 64), nn.Tanh())
        self.meta_gru = nn.GRUCell(64, hidden_dim)
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM + hidden_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )
        self.actor = nn.Linear(128, N_ACTIONS)
        self.critic = nn.Linear(128, 1)

    def initial_hidden(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.hidden_dim, device=device)

    def forward(
        self, observations: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.ndim == 1:
            joined = torch.cat([observations, hidden], dim=-1)
        else:
            expanded = hidden.unsqueeze(0).expand(observations.shape[0], -1)
            joined = torch.cat([observations, expanded], dim=-1)
        features = self.body(joined)
        return self.actor(features), self.critic(features).squeeze(-1)

    def update_hidden(
        self,
        hidden: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        one_hot = F.one_hot(actions.long(), N_ACTIONS).float()
        extras = torch.stack([rewards.clamp(-2.0, 2.0), dones.float()], dim=-1)
        transitions = torch.cat(
            [observations, one_hot, extras, next_observations], dim=-1
        )
        encoded = self.transition_encoder(transitions)
        weights = torch.softmax(self.event_attention(encoded).squeeze(-1), dim=0)
        attended = (encoded * weights.unsqueeze(-1)).sum(dim=0)
        strongest = encoded.max(dim=0).values
        episode_code = self.episode_encoder(torch.cat([attended, strongest]))
        return self.meta_gru(episode_code, hidden)


@dataclass
class Episode:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    episode_return: float
    discounted_return: float
    success: bool
    route: str
    steps: int
    deadline: int
    probed_gate: bool


@dataclass
class Trial:
    gate_state: str
    episodes: list[Episode]


def classify_route(positions: list[tuple[int, int]]) -> str:
    if ThreeRouteHiddenGateEnvV4.GATE_POSITION in positions:
        return "direct"
    if any(p in ThreeRouteHiddenGateEnvV4.DANGER_POSITIONS for p in positions):
        return "medium"
    if any(p in {(3, 5), (3, 6), (4, 6), (5, 6)} for p in positions):
        return "detour"
    return "other"


def discounted(rewards: np.ndarray, gamma: float) -> np.ndarray:
    result = np.empty_like(rewards, dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        result[index] = running
    return result


def episode_tensors(episode: Episode, device: torch.device):
    return (
        torch.as_tensor(episode.observations, dtype=torch.float32, device=device),
        torch.as_tensor(episode.actions, dtype=torch.long, device=device),
        torch.as_tensor(episode.rewards, dtype=torch.float32, device=device),
        torch.as_tensor(episode.next_observations, dtype=torch.float32, device=device),
        torch.as_tensor(episode.dones, dtype=torch.float32, device=device),
    )


def curriculum_deadline(steps: int, curriculum_steps: int) -> int:
    if curriculum_steps <= 0 or steps >= curriculum_steps:
        return int(np.random.choice(DEADLINES))
    fraction = steps / curriculum_steps
    if fraction < 0.25:
        return 500
    if fraction < 0.45:
        return 100
    if fraction < 0.65:
        return 50
    if fraction < 0.80:
        return 25
    return int(np.random.choice(DEADLINES))


@torch.no_grad()
def collect_trial(
    model: TrialActorCritic,
    env: gym.Env,
    gate: str,
    seed: int,
    steps: int,
    args: argparse.Namespace,
    device: torch.device,
    deterministic: bool = False,
    trial_deadline: int | None = None,
) -> Trial:
    # Sample the task once. Curriculum transitions apply to the next trial only.
    deadline = (
        int(trial_deadline) if trial_deadline is not None
        else curriculum_deadline(steps, args.curriculum_steps)
    )
    hidden = model.initial_hidden(device)
    episodes: list[Episode] = []
    for episode_index in range(EPISODES_PER_TRIAL):
        options: dict[str, Any] = {
            "new_trial": episode_index == 0,
            "deadline": deadline,
        }
        if episode_index == 0:
            options["gate_state"] = gate
        observation, _ = env.reset(
            seed=seed if episode_index == 0 else None, options=options
        )
        base = env.unwrapped
        if base.gate_state != gate or base.deadline != deadline:
            raise RuntimeError("Gate or deadline changed within a trial")
        positions = [tuple(map(int, base.agent_pos))]
        obs_items: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        next_items: list[np.ndarray] = []
        dones: list[float] = []
        old_logs: list[float] = []
        old_values: list[float] = []
        probed_gate = visible_gate_state(observation) is not None
        done = False
        final_info: dict[str, Any] = {}
        while not done:
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
            logits, value = model(obs_tensor, hidden)
            distribution = Categorical(logits=logits)
            action_tensor = logits.argmax() if deterministic else distribution.sample()
            action = int(action_tensor.item())
            nxt, reward, terminated, truncated, final_info = env.step(action)
            done = bool(terminated or truncated)
            probed_gate = probed_gate or visible_gate_state(nxt) is not None
            obs_items.append(observation.copy())
            actions.append(action)
            rewards.append(float(reward))
            next_items.append(nxt.copy())
            dones.append(float(done))
            old_logs.append(float(distribution.log_prob(action_tensor).item()))
            old_values.append(float(value.item()))
            observation = nxt
            positions.append(tuple(map(int, base.agent_pos)))

        rewards_array = np.asarray(rewards, np.float32)
        episode = Episode(
            observations=np.asarray(obs_items, np.float32),
            actions=np.asarray(actions, np.int64),
            rewards=rewards_array,
            next_observations=np.asarray(next_items, np.float32),
            dones=np.asarray(dones, np.float32),
            old_log_probs=np.asarray(old_logs, np.float32),
            old_values=np.asarray(old_values, np.float32),
            episode_return=float(rewards_array.sum()),
            discounted_return=float(discounted(rewards_array, args.gamma)[0]),
            success=bool(final_info.get("success", False)),
            route=classify_route(positions),
            steps=int(base.step_count),
            deadline=deadline,
            probed_gate=probed_gate,
        )
        episodes.append(episode)
        if episode_index < EPISODES_PER_TRIAL - 1:
            hidden = model.update_hidden(hidden, *episode_tensors(episode, device))
    return Trial(gate, episodes)


def expert_route(gate: str, deadline: int, episode_index: int) -> str:
    """Information-seeking first episode, then the best gate-aware route."""
    if episode_index == 0 or gate == "open":
        return "direct"
    if deadline == 17:
        return "medium"
    # Closed D=15 is infeasible.  The detour prefix avoids both danger and
    # repeated gate collisions, giving the least-bad timeout demonstration.
    return "detour"


def first_episode_actions(observation: np.ndarray) -> tuple[int, ...]:
    """After three forward steps the door first enters the 5x5 view.

    The closed-door branch returns to the fork and takes the safe lower path.
    Even the fastest diversion would take 20 steps (medium), exceeding all
    target deadlines; the safe prefix avoids danger and collision penalties.
    """
    gate = visible_gate_state(observation)
    if gate == "open":
        return ThreeRouteHiddenGateEnvV4.DIRECT_SAFE_ACTIONS[3:]
    if gate == "closed":
        # At (4,4), east: turn west, return to (3,4), turn south.
        return (0, 0, 2, 0) + ThreeRouteHiddenGateEnvV4.DETOUR_SAFE_ACTIONS[3:]
    raise RuntimeError("Expert reached its sensing point without visible gate evidence")


def route_actions(route: str) -> tuple[int, ...]:
    return {
        "direct": ThreeRouteHiddenGateEnvV4.DIRECT_SAFE_ACTIONS,
        "medium": ThreeRouteHiddenGateEnvV4.MEDIUM_PATH_ACTIONS,
        "detour": ThreeRouteHiddenGateEnvV4.DETOUR_SAFE_ACTIONS,
    }[route]


@torch.no_grad()
def collect_expert_trial(
    env: gym.Env,
    gate: str,
    deadline: int,
    seed: int,
    args: argparse.Namespace,
) -> Trial:
    """Collect one scripted three-episode trial without oracle policy inputs."""
    episodes: list[Episode] = []
    for episode_index in range(EPISODES_PER_TRIAL):
        options: dict[str, Any] = {
            "new_trial": episode_index == 0,
            "deadline": deadline,
        }
        if episode_index == 0:
            options["gate_state"] = gate
        observation, _ = env.reset(
            seed=seed if episode_index == 0 else None, options=options
        )
        base = env.unwrapped
        positions = [tuple(map(int, base.agent_pos))]
        obs_items: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        next_items: list[np.ndarray] = []
        dones: list[float] = []
        probed_gate = visible_gate_state(observation) is not None
        final_info: dict[str, Any] = {}
        script = (
            (2, 2, 2) if episode_index == 0
            else route_actions(expert_route(gate, deadline, episode_index))
        )
        step_index = 0
        done = False
        while not done:
            if episode_index == 0 and step_index == 3:
                script += first_episode_actions(observation)
            # If a script ends, harmless alternating turns spend remaining time.
            action = int(script[step_index]) if step_index < len(script) else step_index % 2
            step_index += 1
            nxt, reward, terminated, truncated, final_info = env.step(action)
            done = bool(terminated or truncated)
            probed_gate = probed_gate or visible_gate_state(nxt) is not None
            obs_items.append(observation.copy())
            actions.append(action)
            rewards.append(float(reward))
            next_items.append(nxt.copy())
            dones.append(float(done))
            observation = nxt
            positions.append(tuple(map(int, base.agent_pos)))

        rewards_array = np.asarray(rewards, np.float32)
        episodes.append(Episode(
            observations=np.asarray(obs_items, np.float32),
            actions=np.asarray(actions, np.int64),
            rewards=rewards_array,
            next_observations=np.asarray(next_items, np.float32),
            dones=np.asarray(dones, np.float32),
            old_log_probs=np.zeros(len(actions), np.float32),
            old_values=np.zeros(len(actions), np.float32),
            episode_return=float(rewards_array.sum()),
            discounted_return=float(discounted(rewards_array, args.gamma)[0]),
            success=bool(final_info.get("success", False)),
            route=classify_route(positions),
            steps=int(base.step_count),
            deadline=deadline,
            probed_gate=probed_gate,
        ))
    return Trial(gate, episodes)


def build_expert_trials(env: gym.Env, args: argparse.Namespace) -> list[Trial]:
    return [
        collect_expert_trial(
            env, gate, deadline,
            args.seed + 100_000 + gate_index * 100 + deadline,
            args,
        )
        for gate_index, gate in enumerate(("open", "closed"))
        for deadline in DEADLINES
    ]


def bc_coefficient(total_steps: int, args: argparse.Namespace) -> float:
    if args.bc_decay_steps <= 0:
        return float(args.bc_final_weight)
    fraction = min(max(total_steps / args.bc_decay_steps, 0.0), 1.0)
    return float(
        args.bc_initial_weight
        + fraction * (args.bc_final_weight - args.bc_initial_weight)
    )


def demonstration_loss(
    model: TrialActorCritic,
    expert_trials: list[Trial],
    device: torch.device,
) -> torch.Tensor:
    """BC over full trials; h is inferred only from demonstrated transitions."""
    losses: list[torch.Tensor] = []
    for trial in expert_trials:
        hidden = model.initial_hidden(device)
        for episode_index, episode in enumerate(trial.episodes):
            obs, actions, rewards, nxt, dones = episode_tensors(episode, device)
            logits, _ = model(obs, hidden)
            losses.append(F.cross_entropy(logits, actions, reduction="none"))
            if episode_index < EPISODES_PER_TRIAL - 1:
                hidden = model.update_hidden(
                    hidden, obs, actions, rewards, nxt, dones
                )
    return torch.cat(losses).mean()


def build_meta_targets(
    trials: list[Trial], gamma: float, beta: float
) -> tuple[list[list[np.ndarray]], np.ndarray]:
    """Give each action its own suffix plus all later episode returns."""
    targets: list[list[np.ndarray]] = []
    flat_advantages: list[np.ndarray] = []
    for trial in trials:
        episode_returns = [episode.discounted_return for episode in trial.episodes]
        trial_targets: list[np.ndarray] = []
        for episode_index, episode in enumerate(trial.episodes):
            future = sum(
                beta ** (later - episode_index) * episode_returns[later]
                for later in range(episode_index + 1, EPISODES_PER_TRIAL)
            )
            target = discounted(episode.rewards, gamma) + future
            trial_targets.append(target.astype(np.float32))
            flat_advantages.append(target - episode.old_values)
        targets.append(trial_targets)
    advantages = np.concatenate(flat_advantages).astype(np.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return targets, advantages


def ppo_update(
    model: TrialActorCritic,
    optimizer: torch.optim.Optimizer,
    trials: list[Trial],
    targets: list[list[np.ndarray]],
    advantages: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    expert_trials: list[Trial],
    total_steps: int,
) -> dict[str, float]:
    old_logs = torch.as_tensor(
        np.concatenate([e.old_log_probs for t in trials for e in t.episodes]),
        dtype=torch.float32,
        device=device,
    )
    target_tensor = torch.as_tensor(
        np.concatenate([
            targets[t][e]
            for t in range(len(trials)) for e in range(EPISODES_PER_TRIAL)
        ]),
        dtype=torch.float32,
        device=device,
    )
    advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    metrics: dict[str, float] = {}
    for _ in range(args.ppo_epochs):
        logs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for trial in trials:
            hidden = model.initial_hidden(device)
            for episode_index, episode in enumerate(trial.episodes):
                obs, actions, rewards, nxt, dones = episode_tensors(episode, device)
                logits, value = model(obs, hidden)
                distribution = Categorical(logits=logits)
                logs.append(distribution.log_prob(actions))
                values.append(value)
                entropies.append(distribution.entropy())
                if episode_index < EPISODES_PER_TRIAL - 1:
                    hidden = model.update_hidden(
                        hidden, obs, actions, rewards, nxt, dones
                    )
        new_logs = torch.cat(logs)
        values_tensor = torch.cat(values)
        entropy = torch.cat(entropies).mean()
        ratios = torch.exp(new_logs - old_logs)
        unclipped = ratios * advantage_tensor
        clipped = torch.clamp(
            ratios, 1.0 - args.clip_range, 1.0 + args.clip_range
        ) * advantage_tensor
        actor_loss = -torch.minimum(unclipped, clipped).mean()
        critic_loss = 0.5 * (values_tensor - target_tensor).pow(2).mean()
        bc_loss = demonstration_loss(model, expert_trials, device)
        bc_weight = bc_coefficient(total_steps, args)
        loss = (
            actor_loss
            + args.value_coef * critic_loss
            - args.entropy_coef * entropy
            + bc_weight * bc_loss
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        metrics = {
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "bc_loss": float(bc_loss.item()),
            "bc_weight": float(bc_weight),
            "entropy": float(entropy.item()),
            "clip_fraction": float(
                (torch.abs(ratios - 1.0) > args.clip_range).float().mean().item()
            ),
        }
    return metrics


def save_checkpoint(path, model, optimizer, args, total_steps, updates) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(args),
        "total_steps": total_steps,
        "updates": updates,
    }, path)


def train(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    env = make_env(args)
    expert_trials = build_expert_trials(env, args)
    model = TrialActorCritic(args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "update", "trial", "total_steps", "gate_state", "trial_return",
        "episode_1_return", "episode_2_return", "episode_3_return",
        "episode_1_success", "episode_2_success", "episode_3_success",
        "episode_1_route", "episode_2_route", "episode_3_route",
        "episode_1_deadline", "episode_2_deadline", "episode_3_deadline",
        "episode_1_probe", "loss", "bc_loss", "bc_weight", "entropy",
        "clip_fraction",
    ]
    total_steps = updates = trial_count = 0
    recent_success: list[float] = []
    recent_probe: list[float] = []
    recent_return: list[float] = []
    with (output / "train_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while total_steps < args.total_steps:
            gates = ["open"] * (args.trials_per_update // 2)
            gates += ["closed"] * (args.trials_per_update // 2)
            random.shuffle(gates)
            trials: list[Trial] = []
            for gate in gates:
                trial = collect_trial(
                    model, env, gate, args.seed + trial_count,
                    total_steps, args, device,
                )
                trials.append(trial)
                total_steps += sum(e.steps for e in trial.episodes)
                trial_count += 1
            targets, advantages = build_meta_targets(
                trials, args.gamma, args.beta_trial
            )
            metrics = ppo_update(
                model, optimizer, trials, targets, advantages, args, device,
                expert_trials, total_steps,
            )
            updates += 1
            for offset, trial in enumerate(trials):
                trial_return = sum(e.episode_return for e in trial.episodes)
                recent_success.append(float(trial.episodes[-1].success))
                recent_probe.append(float(trial.episodes[0].probed_gate))
                recent_return.append(trial_return)
                row: dict[str, Any] = {
                    "update": updates,
                    "trial": trial_count - len(trials) + offset,
                    "total_steps": total_steps,
                    "gate_state": trial.gate_state,
                    "trial_return": trial_return,
                    "episode_1_probe": trial.episodes[0].probed_gate,
                    "loss": metrics["loss"],
                    "bc_loss": metrics["bc_loss"],
                    "bc_weight": metrics["bc_weight"],
                    "entropy": metrics["entropy"],
                    "clip_fraction": metrics["clip_fraction"],
                }
                for number, episode in enumerate(trial.episodes, 1):
                    row[f"episode_{number}_return"] = episode.episode_return
                    row[f"episode_{number}_success"] = episode.success
                    row[f"episode_{number}_route"] = episode.route
                    row[f"episode_{number}_deadline"] = episode.deadline
                writer.writerow(row)
            handle.flush()
            recent_success = recent_success[-50:]
            recent_probe = recent_probe[-50:]
            recent_return = recent_return[-50:]
            if updates == 1 or updates % args.log_interval == 0:
                print(
                    f"update={updates} trials={trial_count} steps={total_steps} "
                    f"trial_return50={np.mean(recent_return):.3f} "
                    f"final_success50={np.mean(recent_success):.3f} "
                    f"probe50={np.mean(recent_probe):.3f} "
                    f"bc={metrics['bc_loss']:.3f}*{metrics['bc_weight']:.3f} "
                    f"entropy={metrics['entropy']:.3f} "
                    f"clip={metrics['clip_fraction']:.3f}"
                )
            if updates % args.checkpoint_interval == 0:
                save_checkpoint(
                    output / "checkpoints" / f"trial_ppo_{updates}.pt",
                    model, optimizer, args, total_steps, updates,
                )
    env.close()
    save_checkpoint(
        output / "final_model.pt", model, optimizer, args, total_steps, updates
    )
    return model, {
        "method": "time_conditioned_trial_ppo_with_decaying_bc",
        "objective": "trial PPO plus decaying full-trial behavior cloning",
        "gate_observation": "open/closed state visible only inside the local 5x5 view",
        "probe_definition": "gate enters local policy observation",
        "trial_protocol": "three episodes; fixed gate and deadline; inherited h",
        "total_steps": total_steps,
        "trials": trial_count,
        "updates": updates,
        "recent_trial_return": float(np.mean(recent_return)),
        "recent_final_success": float(np.mean(recent_success)),
        "recent_episode1_probe": float(np.mean(recent_probe)),
    }


@torch.no_grad()
def evaluate(model: TrialActorCritic, args: argparse.Namespace, device: torch.device):
    model.eval()
    env = make_env(args)
    records: list[dict[str, Any]] = []
    hidden_after_one: dict[str, list[np.ndarray]] = {"open": [], "closed": []}
    try:
        for gate in ("open", "closed"):
            for trial_index in range(args.eval_trials_per_gate):
                for deadline in DEADLINES:
                    trial = collect_trial(
                        model, env, gate, args.eval_seed + trial_index,
                        args.total_steps, args, device,
                        deterministic=True, trial_deadline=deadline,
                    )
                    hidden = model.initial_hidden(device)
                    hidden = model.update_hidden(
                        hidden, *episode_tensors(trial.episodes[0], device)
                    )
                    hidden_after_one[gate].append(hidden.cpu().numpy())
                    for episode_index, episode in enumerate(trial.episodes, 1):
                        records.append({
                            "gate_state": gate,
                            "trial": trial_index,
                            "episode": episode_index,
                            "deadline": episode.deadline,
                            "success": episode.success,
                            "return": episode.episode_return,
                            "steps": episode.steps,
                            "route": episode.route,
                            "probed_gate": episode.probed_gate,
                            "deadline_sequence": [deadline] * EPISODES_PER_TRIAL,
                        })
    finally:
        env.close()
    aggregate: dict[str, Any] = {"open": {}, "closed": {}}
    for gate in aggregate:
        for deadline in DEADLINES:
            subset = [
                r for r in records
                if r["gate_state"] == gate
                and r["episode"] == 3
                and r["deadline"] == deadline
            ]
            aggregate[gate][str(deadline)] = {
                "success_rate": float(np.mean([r["success"] for r in subset])),
                "mean_return": float(np.mean([r["return"] for r in subset])),
                "mean_steps": float(np.mean([r["steps"] for r in subset])),
                "route_rates": {
                    name: float(np.mean([r["route"] == name for r in subset]))
                    for name in ("direct", "medium", "detour", "other")
                },
            }
    distance = float(np.linalg.norm(
        np.mean(hidden_after_one["open"], axis=0)
        - np.mean(hidden_after_one["closed"], axis=0)
    ))
    return {
        "aggregate": aggregate,
        "trial_protocol": "fixed gate and deadline across all three episodes",
        "episode1_hidden_mean_distance": distance,
        "episodes": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-steps", type=int, default=551_179)
    parser.add_argument("--curriculum-steps", type=int, default=500_000)
    parser.add_argument("--trials-per-update", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--beta-trial", type=float, default=1.0)
    parser.add_argument("--ppo-epochs", type=int, default=6)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--bc-initial-weight", type=float, default=1.0)
    parser.add_argument("--bc-final-weight", type=float, default=0.0)
    parser.add_argument("--bc-decay-steps", type=int, default=600_000)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--time-cap", type=float, default=20.0)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=2)
    parser.add_argument("--eval-trials-per-gate", type=int, default=3)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=str(HERE / "release_0_64p5w/runs/episode_from_scratch"))
    parser.set_defaults(trial_deadline_mode="fixed", episodes_per_trial=EPISODES_PER_TRIAL)
    args = parser.parse_args()
    if args.trials_per_update <= 0 or args.trials_per_update % 2:
        parser.error("--trials-per-update must be a positive even number")
    if not 0.0 <= args.beta_trial <= 1.0:
        parser.error("--beta-trial must be in [0, 1]")
    if args.time_cap <= 0:
        parser.error("--time-cap must be positive")
    if args.bc_initial_weight < 0 or args.bc_final_weight < 0:
        parser.error("BC weights must be non-negative")
    if args.bc_decay_steps < 0:
        parser.error("--bc-decay-steps must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Output is not empty; select a fresh --output-dir')
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2))
    model, training = train(args)
    device = torch.device(args.device)
    evaluation = evaluate(model, args, device)
    summary = {
        "training": training,
        "evaluation": evaluation["aggregate"],
        "episode1_hidden_mean_distance": evaluation[
            "episode1_hidden_mean_distance"
        ],
    }
    (output / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

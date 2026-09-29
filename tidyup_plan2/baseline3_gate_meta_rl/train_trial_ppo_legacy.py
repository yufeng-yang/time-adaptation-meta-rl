"""Final-episode Meta-RL for a hidden gate, with no time observation.

A trial has three episodes and one fixed hidden gate.  Each episode draws its
deadline independently from {15, 17, 19}, so previous episodes cannot reveal
the current deadline through h.  D, t, and D-t are never observed.
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

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
WORKSPACE_ROOT = PACKAGE_DIR.parent
LOCAL_MINIGRID_DIR = WORKSPACE_ROOT / "Minigrid"
for path in (WORKSPACE_ROOT, LOCAL_MINIGRID_DIR, PACKAGE_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tidyup_plan2.envs import ThreeRouteHiddenGateEnvV4

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts"
DEADLINES = (15, 17, 19)
EPISODES_PER_TRIAL = 3
VIEW_SIZE = 5
N_CELL_TYPES = 6
OBSERVATION_DIM = VIEW_SIZE * VIEW_SIZE * N_CELL_TYPES + 4
N_ACTIONS = 3
OBJECT_WALL, OBJECT_DOOR, OBJECT_GOAL = 2, 4, 8


class NoTimeHiddenGateObservation(gym.ObservationWrapper):
    """Local semantic map + heading; door state and all time are removed."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(OBSERVATION_DIM, dtype=np.float32),
            high=np.ones(OBSERVATION_DIM, dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, observation: dict[str, Any]) -> np.ndarray:
        objects = np.asarray(observation["image"], dtype=np.uint8)[..., 0]
        cells = np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.int64)
        cells[objects == OBJECT_WALL] = 1
        cells[objects == OBJECT_GOAL] = 2
        cells[objects == OBJECT_DOOR] = 4  # open/closed state channel discarded
        cells[VIEW_SIZE // 2, VIEW_SIZE // 2] = 5
        sensor = np.eye(N_CELL_TYPES, dtype=np.float32)[cells].reshape(-1)
        direction = np.zeros(4, dtype=np.float32)
        direction[int(observation["direction"])] = 1.0
        return np.concatenate([sensor, direction]).astype(np.float32)


class EpisodeMetaActorCritic(nn.Module):
    """Policy plus an event-sensitive completed-episode encoder."""

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        transition_dim = OBSERVATION_DIM * 2 + N_ACTIONS + 2
        self.transition_encoder = nn.Sequential(
            nn.Linear(transition_dim, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
        )
        self.event_attention = nn.Linear(64, 1)
        self.episode_encoder = nn.Sequential(nn.Linear(128, 64), nn.Tanh())
        self.meta_gru = nn.GRUCell(64, hidden_dim)
        self.policy_body = nn.Sequential(
            nn.Linear(OBSERVATION_DIM + hidden_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )
        self.explore_actor = nn.Linear(128, N_ACTIONS)
        self.task_actor = nn.Linear(128, N_ACTIONS)
        self.explore_critic = nn.Linear(128, 1)
        self.task_critic = nn.Linear(128, 1)

    def initial_hidden(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.hidden_dim, device=device)

    def forward(
        self, observations: torch.Tensor, hidden: torch.Tensor, role: int = 2
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.ndim == 1:
            joined = torch.cat([observations, hidden])
        else:
            expanded = hidden.unsqueeze(0).expand(observations.shape[0], -1)
            joined = torch.cat([observations, expanded], dim=-1)
        features = self.policy_body(joined)
        if role == 0:
            return (
                self.explore_actor(features),
                self.explore_critic(features).squeeze(-1),
            )
        return self.task_actor(features), self.task_critic(features).squeeze(-1)

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
        auxiliary = torch.stack([rewards.clamp(-2, 2), dones.float()], dim=-1)
        transitions = torch.cat(
            [observations, one_hot, auxiliary, next_observations], dim=-1
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
    probe_rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    episode_return: float
    success: bool
    route: str
    steps: int
    route_failure: bool
    deadline: int


@dataclass
class Trial:
    gate_state: str
    episodes: list[Episode]


def make_env(args: argparse.Namespace) -> gym.Env:
    return NoTimeHiddenGateObservation(
        ThreeRouteHiddenGateEnvV4(
            min_deadline=min(DEADLINES),
            max_deadline=max(DEADLINES),
            step_penalty=args.step_penalty,
            progress_scale=0.0,
        )
    )


def reset_episode(
    env: gym.Env,
    first: bool,
    seed: int | None,
    gate_state: str | None,
    deadline: int,
) -> tuple[np.ndarray, dict]:
    options: dict[str, Any] = {"new_trial": first, "deadline": int(deadline)}
    if first and gate_state is not None:
        options["gate_state"] = gate_state
    return env.reset(seed=seed, options=options)


def classify_route(positions: list[tuple[int, int]]) -> str:
    if ThreeRouteHiddenGateEnvV4.GATE_POSITION in positions:
        return "direct"
    if any(p in ThreeRouteHiddenGateEnvV4.DANGER_POSITIONS for p in positions):
        return "medium"
    detour_markers = {(3, 5), (3, 6), (4, 6), (5, 6)}
    if any(p in detour_markers for p in positions):
        return "detour"
    return "other"


def as_tensors(
    episode: Episode, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(episode.observations, dtype=torch.float32, device=device),
        torch.as_tensor(episode.actions, dtype=torch.long, device=device),
        torch.as_tensor(episode.rewards, dtype=torch.float32, device=device),
        torch.as_tensor(episode.next_observations, dtype=torch.float32, device=device),
        torch.as_tensor(episode.dones, dtype=torch.float32, device=device),
    )


@torch.no_grad()
def collect_trial(
    model: EpisodeMetaActorCritic,
    env: gym.Env,
    gate_state: str,
    seed: int,
    device: torch.device,
) -> Trial:
    hidden = model.initial_hidden(device)
    episodes: list[Episode] = []
    for episode_index in range(EPISODES_PER_TRIAL):
        deadline = int(np.random.choice(DEADLINES))
        observation, _ = reset_episode(
            env,
            first=episode_index == 0,
            seed=seed if episode_index == 0 else None,
            gate_state=gate_state if episode_index == 0 else None,
            deadline=deadline,
        )
        base = env.unwrapped
        positions = [tuple(map(int, base.agent_pos))]
        obs_list: list[np.ndarray] = []
        action_list: list[int] = []
        reward_list: list[float] = []
        probe_reward_list: list[float] = []
        next_obs_list: list[np.ndarray] = []
        done_list: list[float] = []
        log_prob_list: list[float] = []
        value_list: list[float] = []
        done = False
        final_info: dict[str, Any] = {}
        route_failure = False
        found_gate_evidence = False
        while not done:
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
            logits, value = model(obs_tensor, hidden, episode_index)
            distribution = Categorical(logits=logits)
            action_tensor = distribution.sample()
            action = int(action_tensor.item())
            previous_position = tuple(map(int, base.agent_pos))
            previous_direction = int(base.agent_dir)
            next_observation, reward, terminated, truncated, final_info = env.step(action)
            done = bool(terminated or truncated)
            next_position = tuple(map(int, base.agent_pos))
            positions.append(next_position)

            direction_vectors = ((1, 0), (0, 1), (-1, 0), (0, -1))
            dx, dy = direction_vectors[previous_direction]
            attempted_position = (previous_position[0] + dx, previous_position[1] + dy)
            gate_evidence = (
                action == 2
                and attempted_position == ThreeRouteHiddenGateEnvV4.GATE_POSITION
            )
            probe_reward = float(gate_evidence and not found_gate_evidence)
            found_gate_evidence = found_gate_evidence or gate_evidence
            obs_list.append(observation.copy())
            action_list.append(action)
            reward_list.append(float(reward))
            probe_reward_list.append(probe_reward)
            next_obs_list.append(next_observation.copy())
            done_list.append(float(done))
            log_prob_list.append(float(distribution.log_prob(action_tensor).item()))
            value_list.append(float(value.item()))
            observation = next_observation

        episode = Episode(
            observations=np.asarray(obs_list, np.float32),
            actions=np.asarray(action_list, np.int64),
            rewards=np.asarray(reward_list, np.float32),
            probe_rewards=np.asarray(probe_reward_list, np.float32),
            next_observations=np.asarray(next_obs_list, np.float32),
            dones=np.asarray(done_list, np.float32),
            old_log_probs=np.asarray(log_prob_list, np.float32),
            old_values=np.asarray(value_list, np.float32),
            episode_return=float(np.sum(reward_list)),
            success=bool(final_info.get("success", False)),
            route=classify_route(positions),
            steps=int(base.step_count),
            route_failure=route_failure,
            deadline=deadline,
        )
        episodes.append(episode)
        if episode_index < 2:
            hidden = model.update_hidden(hidden, *as_tensors(episode, device))
    return Trial(gate_state, episodes)


def discounted(rewards: np.ndarray, gamma: float) -> np.ndarray:
    result = np.empty_like(rewards, dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        result[index] = running
    return result


def build_targets(
    trials: list[Trial], gamma: float, probe_coef: float
) -> tuple[list[list[np.ndarray]], list[np.ndarray]]:
    """Credit episode-3 performance to every earlier exploratory action."""
    targets: list[list[np.ndarray]] = []
    raw_advantages: list[list[np.ndarray]] = [[], [], []]
    for trial in trials:
        final = trial.episodes[2]
        final_score = final.episode_return
        per_trial: list[np.ndarray] = []
        for role, episode in enumerate(trial.episodes):
            if role == 0:
                role_target = probe_coef * discounted(episode.probe_rewards, gamma)
            elif role == 1:
                role_target = np.full(len(episode.rewards), final_score, np.float32)
            else:
                role_target = discounted(episode.rewards, gamma)
            per_trial.append(role_target)
            raw_advantages[role].append(role_target - episode.old_values)
        targets.append(per_trial)

    normalized: list[np.ndarray] = []
    for parts in raw_advantages:
        values = np.concatenate(parts)
        normalized.append(((values - values.mean()) / (values.std() + 1e-8)).astype(np.float32))
    return targets, normalized


def ppo_update(
    model: EpisodeMetaActorCritic,
    optimizer: torch.optim.Optimizer,
    trials: list[Trial],
    targets: list[list[np.ndarray]],
    advantages: list[np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    old_log_probs = [
        torch.as_tensor(
            np.concatenate([t.episodes[r].old_log_probs for t in trials]),
            dtype=torch.float32, device=device,
        ) for r in range(3)
    ]
    target_tensors = [
        torch.as_tensor(
            np.concatenate([targets[i][r] for i in range(len(trials))]),
            dtype=torch.float32, device=device,
        ) for r in range(3)
    ]
    advantage_tensors = [
        torch.as_tensor(a, dtype=torch.float32, device=device) for a in advantages
    ]
    metrics: dict[str, float] = {}
    for _ in range(args.ppo_epochs):
        role_logs: list[list[torch.Tensor]] = [[], [], []]
        role_values: list[list[torch.Tensor]] = [[], [], []]
        role_entropies: list[list[torch.Tensor]] = [[], [], []]
        for trial in trials:
            hidden = model.initial_hidden(device)
            for role, episode in enumerate(trial.episodes):
                observations, actions, rewards, next_observations, dones = as_tensors(episode, device)
                logits, values = model(observations, hidden, role)
                distribution = Categorical(logits=logits)
                role_logs[role].append(distribution.log_prob(actions))
                role_values[role].append(values)
                role_entropies[role].append(distribution.entropy())
                if role < 2:
                    hidden = model.update_hidden(
                        hidden, observations, actions, rewards, next_observations, dones
                    )

        actor_parts: list[torch.Tensor] = []
        critic_parts: list[torch.Tensor] = []
        entropy_parts: list[torch.Tensor] = []
        clip_parts: list[torch.Tensor] = []
        for role in range(3):
            logs = torch.cat(role_logs[role])
            values = torch.cat(role_values[role])
            ratios = torch.exp(logs - old_log_probs[role])
            unclipped = ratios * advantage_tensors[role]
            clipped = torch.clamp(
                ratios, 1 - args.clip_range, 1 + args.clip_range
            ) * advantage_tensors[role]
            actor_parts.append(-torch.minimum(unclipped, clipped).mean())
            critic_parts.append(0.5 * (values - target_tensors[role]).pow(2).mean())
            entropy_parts.append(torch.cat(role_entropies[role]).mean())
            clip_parts.append((torch.abs(ratios - 1) > args.clip_range).float().mean())

        actor_loss = torch.stack(actor_parts).mean()
        critic_loss = torch.stack(critic_parts).mean()
        entropy = torch.stack(entropy_parts).mean()
        loss = actor_loss + args.value_coef * critic_loss - args.entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        metrics = {
            "loss": float(loss.item()),
            "entropy": float(entropy.item()),
            "clip_fraction": float(torch.stack(clip_parts).mean().item()),
        }
    return metrics


def save_model(
    path: Path,
    model: EpisodeMetaActorCritic,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    steps: int,
    update: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
            "total_steps": steps,
            "update": update,
        }, path,
    )


def train(args: argparse.Namespace) -> tuple[EpisodeMetaActorCritic, dict[str, Any]]:
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    env = make_env(args)
    model = EpisodeMetaActorCritic(args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        "update", "trial", "total_steps", "gate_state",
        "episode_1_return", "episode_2_return", "episode_3_return",
        "episode_1_success", "episode_2_success", "episode_3_success",
        "episode_1_route", "episode_2_route", "episode_3_route",
        "episode_1_route_failure", "episode_2_route_failure", "episode_3_route_failure",
        "episode_1_deadline", "episode_2_deadline", "episode_3_deadline",
        "episode_1_probe", "probe_coef", "loss", "entropy", "clip_fraction",
    ]
    total_steps = update_index = trial_index = 0
    recent_returns: list[float] = []
    recent_success: list[float] = []
    recent_correct: list[float] = []
    recent_probe: list[float] = []
    with (output / "train_trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while total_steps < args.total_steps:
            gates = ["open"] * (args.trials_per_update // 2)
            gates += ["closed"] * (args.trials_per_update // 2)
            random.shuffle(gates)
            trials: list[Trial] = []
            for gate in gates:
                trial = collect_trial(model, env, gate, args.seed + trial_index, device)
                trials.append(trial)
                total_steps += sum(e.steps for e in trial.episodes)
                trial_index += 1
            anneal_fraction = min(total_steps / max(args.probe_anneal_steps, 1), 1.0)
            probe_coef = args.initial_probe_coef + anneal_fraction * (
                args.final_probe_coef - args.initial_probe_coef
            )
            targets, advantages = build_targets(trials, args.gamma, probe_coef)
            metrics = ppo_update(model, optimizer, trials, targets, advantages, args, device)
            update_index += 1
            for offset, trial in enumerate(trials):
                final = trial.episodes[2]
                correct = final.route == ("direct" if trial.gate_state == "open" else "detour")
                recent_returns.append(final.episode_return)
                recent_success.append(float(final.success))
                recent_correct.append(float(correct and final.success))
                probed = float(np.sum(trial.episodes[0].probe_rewards) > 0)
                recent_probe.append(probed)
                row: dict[str, Any] = {
                    "update": update_index,
                    "trial": trial_index - len(trials) + offset,
                    "total_steps": total_steps,
                    "gate_state": trial.gate_state,
                    "episode_1_probe": bool(probed),
                    "probe_coef": probe_coef,
                    **metrics,
                }
                for number, episode in enumerate(trial.episodes, 1):
                    row[f"episode_{number}_return"] = episode.episode_return
                    row[f"episode_{number}_success"] = episode.success
                    row[f"episode_{number}_route"] = episode.route
                    row[f"episode_{number}_route_failure"] = episode.route_failure
                    row[f"episode_{number}_deadline"] = episode.deadline
                writer.writerow(row)
            handle.flush()
            recent_returns = recent_returns[-40:]
            recent_success = recent_success[-40:]
            recent_correct = recent_correct[-40:]
            recent_probe = recent_probe[-40:]
            if update_index == 1 or update_index % args.log_interval == 0:
                print(
                    f"update={update_index} trials={trial_index} steps={total_steps} "
                    f"final_return40={np.mean(recent_returns):.3f} "
                    f"success40={np.mean(recent_success):.3f} "
                    f"correct_route40={np.mean(recent_correct):.3f} "
                    f"probe40={np.mean(recent_probe):.3f} probe_w={probe_coef:.2f} "
                    f"entropy={metrics['entropy']:.3f} clip={metrics['clip_fraction']:.3f}"
                )
            if update_index % args.checkpoint_interval == 0:
                save_model(
                    output / "checkpoints" / f"meta_ppo_update_{update_index}.pt",
                    model, optimizer, args, total_steps, update_index,
                )
    env.close()
    save_model(output / "final_model.pt", model, optimizer, args, total_steps, update_index)
    return model, {
        "total_steps": total_steps,
        "trials": trial_index,
        "updates": update_index,
        "recent_final_return": float(np.mean(recent_returns)),
        "recent_final_success": float(np.mean(recent_success)),
        "recent_correct_route": float(np.mean(recent_correct)),
        "recent_episode1_probe": float(np.mean(recent_probe)),
    }


@torch.no_grad()
def evaluate(model: EpisodeMetaActorCritic, args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    model.eval()
    env = make_env(args)
    records: list[dict[str, Any]] = []
    h1: dict[str, list[np.ndarray]] = {"open": [], "closed": []}
    try:
        for gate in ("open", "closed"):
            for trial_index in range(args.eval_trials_per_gate):
                hidden = model.initial_hidden(device)
                for final_deadline in DEADLINES:
                    trial = collect_trial_deterministic(
                        model, env, gate, args.eval_seed + trial_index,
                        hidden, device, final_deadline,
                    )
                    records.extend(trial[0])
                    h1[gate].append(trial[1])
    finally:
        env.close()
    aggregate: dict[str, Any] = {}
    for gate in ("open", "closed"):
        aggregate[gate] = {}
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
                    route: float(np.mean([r["route"] == route for r in subset]))
                    for route in ("direct", "medium", "detour", "other")
                },
            }
    distance = float(np.linalg.norm(np.mean(h1["open"], axis=0) - np.mean(h1["closed"], axis=0)))
    return {"aggregate": aggregate, "episode1_hidden_mean_distance": distance, "episodes": records}


@torch.no_grad()
def collect_trial_deterministic(
    model: EpisodeMetaActorCritic,
    env: gym.Env,
    gate: str,
    seed: int,
    hidden: torch.Tensor,
    device: torch.device,
    final_deadline: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    records: list[dict[str, Any]] = []
    hidden_after_first = np.zeros(model.hidden_dim, np.float32)
    deadlines = (15, 17, int(final_deadline))
    for role in range(3):
        observation, _ = reset_episode(
            env, role == 0, seed if role == 0 else None,
            gate if role == 0 else None, deadlines[role],
        )
        base = env.unwrapped
        positions = [tuple(map(int, base.agent_pos))]
        obs_list: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        next_list: list[np.ndarray] = []
        dones: list[float] = []
        final_info: dict[str, Any] = {}
        route_failure = False
        done = False
        while not done:
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
            logits, _ = model(obs_tensor, hidden, role)
            action = int(logits.argmax().item())
            nxt, reward, terminated, truncated, final_info = env.step(action)
            done = bool(terminated or truncated)
            positions.append(tuple(map(int, base.agent_pos)))
            obs_list.append(observation.copy())
            actions.append(action)
            rewards.append(float(reward))
            next_list.append(nxt.copy())
            dones.append(float(done))
            observation = nxt
        route = classify_route(positions)
        records.append({
            "gate_state": gate,
            "episode": role + 1,
            "success": bool(final_info.get("success", False)),
            "return": float(np.sum(rewards)),
            "steps": int(base.step_count),
            "route": route,
            "route_failure": route_failure,
            "deadline": deadlines[role],
        })
        if role < 2:
            episode = Episode(
                np.asarray(obs_list, np.float32), np.asarray(actions, np.int64),
                np.asarray(rewards, np.float32), np.zeros(len(rewards), np.float32),
                np.asarray(next_list, np.float32),
                np.asarray(dones, np.float32), np.empty(0), np.empty(0),
                float(np.sum(rewards)), bool(final_info.get("success", False)),
                route, int(base.step_count), route_failure, deadlines[role],
            )
            hidden = model.update_hidden(hidden, *as_tensors(episode, device))
            if role == 0:
                hidden_after_first = hidden.cpu().numpy().copy()
    return records, hidden_after_first


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--episodes-per-trial", type=int, default=3)
    parser.add_argument("--trials-per-update", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--ppo-epochs", type=int, default=8)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--initial-probe-coef", type=float, default=1.0)
    parser.add_argument("--final-probe-coef", type=float, default=1.0)
    parser.add_argument("--probe-anneal-steps", type=int, default=300_000)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=2)
    parser.add_argument("--eval-trials-per-gate", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    if args.episodes_per_trial != 3:
        parser.error("Baseline 3 fixes three episodes per trial")
    if args.trials_per_update <= 0 or args.trials_per_update % 2:
        parser.error("trials-per-update must be a positive even number")
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    model, training = train(args)
    evaluation = evaluate(model, args)
    summary = {
        "training": training,
        "evaluation": evaluation["aggregate"],
        "episode1_hidden_mean_distance": evaluation["episode1_hidden_mean_distance"],
    }
    with (output / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, indent=2)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

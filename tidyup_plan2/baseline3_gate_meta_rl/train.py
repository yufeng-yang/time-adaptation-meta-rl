"""Time-conditioned trial PPO for a hidden gate.

The environment resets three times per trial while the hidden gate stays fixed.
All episodes use the same reward. PPO optimizes a hierarchical trial return, so
later rewards assign credit to information-gathering actions in earlier episodes.
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
    """Local state plus D-t; physical door state remains hidden."""

    def __init__(self, env: gym.Env, time_cap: float = 20.0) -> None:
        super().__init__(env)
        self.time_cap = float(time_cap)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(OBS_DIM, np.float32),
            high=np.ones(OBS_DIM, np.float32),
            dtype=np.float32,
        )

    def observation(self, observation: dict[str, Any]) -> np.ndarray:
        objects = np.asarray(observation["image"], np.uint8)[..., 0]
        cells = np.zeros((VIEW_SIZE, VIEW_SIZE), np.int64)
        cells[objects == 2] = 1
        cells[objects == 8] = 2
        cells[objects == 4] = 4
        cells[VIEW_SIZE // 2, VIEW_SIZE // 2] = 5
        semantic = np.eye(N_CELL_TYPES, dtype=np.float32)[cells].reshape(-1)
        direction = np.zeros(4, np.float32)
        direction[int(observation["direction"])] = 1.0
        remaining = float(np.asarray(observation["time"])[0])
        time_feature = np.asarray(
            [min(remaining, self.time_cap) / self.time_cap], np.float32
        )
        return np.concatenate([semantic, direction, time_feature]).astype(np.float32)


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
    deadline_sequence: tuple[int, int, int] | None = None,
) -> Trial:
    hidden = model.initial_hidden(device)
    episodes: list[Episode] = []
    for episode_index in range(EPISODES_PER_TRIAL):
        deadline = (
            int(deadline_sequence[episode_index])
            if deadline_sequence is not None
            else curriculum_deadline(steps, args.curriculum_steps)
        )
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
        old_logs: list[float] = []
        old_values: list[float] = []
        probed_gate = False
        done = False
        final_info: dict[str, Any] = {}
        while not done:
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
            logits, value = model(obs_tensor, hidden)
            distribution = Categorical(logits=logits)
            action_tensor = logits.argmax() if deterministic else distribution.sample()
            action = int(action_tensor.item())
            previous_position = tuple(map(int, base.agent_pos))
            previous_direction = int(base.agent_dir)
            nxt, reward, terminated, truncated, final_info = env.step(action)
            done = bool(terminated or truncated)
            if action == 2:
                vectors = ((1, 0), (0, 1), (-1, 0), (0, -1))
                dx, dy = vectors[previous_direction]
                attempted = (previous_position[0] + dx, previous_position[1] + dy)
                probed_gate = probed_gate or attempted == base.GATE_POSITION
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
        steps += episode.steps
        if episode_index < EPISODES_PER_TRIAL - 1:
            hidden = model.update_hidden(hidden, *episode_tensors(episode, device))
    return Trial(gate, episodes)


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
        loss = actor_loss + args.value_coef * critic_loss - args.entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        metrics = {
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
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
        "episode_1_probe", "loss", "entropy", "clip_fraction",
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
                model, optimizer, trials, targets, advantages, args, device
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
        "method": "time_conditioned_trial_ppo",
        "objective": "equal environment rewards with cross-episode credit",
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
                for d1 in DEADLINES:
                    for d2 in DEADLINES:
                        for final_d in DEADLINES:
                            trial = collect_trial(
                                model, env, gate, args.eval_seed + trial_index,
                                args.total_steps, args, device,
                                deterministic=True,
                                deadline_sequence=(d1, d2, final_d),
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
                                    "deadline_sequence": [d1, d2, final_d],
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
        "episode1_hidden_mean_distance": distance,
        "episodes": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
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
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--time-cap", type=float, default=20.0)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=2)
    parser.add_argument("--eval-trials-per-gate", type=int, default=3)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=str(HERE / "artifacts"))
    args = parser.parse_args()
    if args.trials_per_update <= 0 or args.trials_per_update % 2:
        parser.error("--trials-per-update must be a positive even number")
    if not 0.0 <= args.beta_trial <= 1.0:
        parser.error("--beta-trial must be in [0, 1]")
    if args.time_cap <= 0:
        parser.error("--time-cap must be positive")
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
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

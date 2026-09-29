"""Trial 2: PPO with BC initialization and a decaying demonstration loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from continuous_pointnav_adaptation.options import WaypointOptionController
from continuous_pointnav_adaptation.train_time_goal_v2 import (
    DEADLINE,
    evaluate,
    make_env,
)


HERE = Path(__file__).resolve().parent


def collect_expert_demonstrations(
    output_path: Path,
    *,
    episodes: int,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Collect corrective direct-route labels along mildly perturbed rollouts."""

    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []
    attempts = 0
    while len(summaries) < episodes:
        attempts += 1
        if attempts > episodes * 10:
            raise RuntimeError("could not collect enough successful demonstrations")
        env = make_env()
        observation, _ = env.reset(seed=seed + attempts, options={"deadline": DEADLINE})
        controller = WaypointOptionController("direct")
        episode_observations: list[np.ndarray] = []
        episode_actions: list[np.ndarray] = []
        total_reward = 0.0
        final_info: dict[str, Any] = {}
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            # Store the clean expert correction for the current, possibly
            # perturbed state; execute a noisy version to widen state coverage.
            expert_action = controller.action(env.unwrapped)
            executed_action = np.clip(
                expert_action + rng.normal(0.0, noise_std, size=expert_action.shape),
                env.action_space.low,
                env.action_space.high,
            )
            episode_observations.append(np.asarray(observation, dtype=np.float32))
            episode_actions.append(np.asarray(expert_action, dtype=np.float32))
            observation, reward, terminated, truncated, final_info = env.step(
                executed_action
            )
            total_reward += float(reward)
            steps += 1
        env.close()
        if not final_info.get("success", False):
            continue
        observations.extend(episode_observations)
        actions.extend(episode_actions)
        summaries.append(
            {
                "episode": len(summaries) + 1,
                "steps": steps,
                "return": total_reward,
                "danger_visits": int(final_info.get("danger_visits", 0)),
            }
        )

    observation_array = np.asarray(observations, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.float32)
    np.savez_compressed(
        output_path,
        observations=observation_array,
        actions=action_array,
    )
    return observation_array, action_array, summaries


def bc_gradient_steps(
    model: PPO,
    normalizer: VecNormalize,
    raw_observations: np.ndarray,
    expert_actions: np.ndarray,
    *,
    gradient_steps: int,
    batch_size: int,
    weight: float,
    rng: np.random.Generator,
) -> float:
    """Apply maximum-likelihood continuous-action BC updates."""

    losses: list[float] = []
    for _ in range(gradient_steps):
        indices = rng.integers(0, len(raw_observations), size=batch_size)
        normalized = normalizer.normalize_obs(raw_observations[indices].copy())
        observation_tensor = torch.as_tensor(
            normalized, dtype=torch.float32, device=model.device
        )
        action_tensor = torch.as_tensor(
            expert_actions[indices], dtype=torch.float32, device=model.device
        )
        distribution = model.policy.get_distribution(observation_tensor)
        # Negative log likelihood is the continuous-action counterpart of the
        # standard BC objective.  Unlike mean-only MSE, it also learns a useful
        # initial exploration variance for PPO.
        loss = -weight * distribution.log_prob(action_tensor).mean()
        model.policy.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
        model.policy.optimizer.step()
        # Avoid a numerically degenerate near-deterministic Gaussian while
        # still allowing BC to make exploration much tighter than std=1.
        model.policy.log_std.data.clamp_(-2.5, 0.5)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


class DecayingBehaviorCloning(BaseCallback):
    """Keep demonstrations in training without treating them as PPO samples."""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        normalizer: VecNormalize,
        *,
        total_timesteps: int,
        initial_weight: float,
        final_weight: float,
        gradient_steps: int,
        batch_size: int,
        seed: int,
    ) -> None:
        super().__init__(verbose=0)
        self.observations = observations
        self.actions = actions
        self.normalizer = normalizer
        self.total_timesteps = total_timesteps
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        fraction = min(1.0, self.num_timesteps / max(1, self.total_timesteps))
        weight = self.initial_weight + fraction * (
            self.final_weight - self.initial_weight
        )
        loss = bc_gradient_steps(
            self.model,
            self.normalizer,
            self.observations,
            self.actions,
            gradient_steps=self.gradient_steps,
            batch_size=self.batch_size,
            weight=weight,
            rng=self.rng,
        )
        self.logger.record("demo/bc_weight", weight)
        self.logger.record("demo/bc_loss", loss)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--expert-episodes", type=int, default=20)
    parser.add_argument("--expert-noise", type=float, default=0.05)
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--bc-initial-weight", type=float, default=1.0)
    parser.add_argument("--bc-final-weight", type=float, default=0.05)
    parser.add_argument("--bc-steps-per-rollout", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=HERE / "trial2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    demo_path = args.output_dir / "expert_demonstrations.npz"
    observations, actions, summaries = collect_expert_demonstrations(
        demo_path,
        episodes=args.expert_episodes,
        noise_std=args.expert_noise,
        seed=args.seed,
    )
    with (args.output_dir / "expert_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)

    def monitored_env(rank: int):
        def build() -> Monitor:
            env = make_env()
            env.reset(seed=args.seed + 1000 + rank)
            return Monitor(
                env,
                filename=str(args.output_dir / f"train_{rank}.monitor.csv"),
                info_keywords=("success", "timeout", "danger_visits", "deadline"),
            )

        return build

    vector_env = SubprocVecEnv(
        [monitored_env(rank) for rank in range(args.n_envs)], start_method="fork"
    )
    vector_env = VecNormalize(vector_env, norm_obs=True, norm_reward=False, gamma=0.995)
    # Initialize observation statistics from the demonstrations so BC and PPO
    # immediately use the same normalization convention.
    vector_env.obs_rms.update(observations)

    model = PPO(
        "MlpPolicy",
        vector_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(args.output_dir / "tensorboard"),
    )

    rng = np.random.default_rng(args.seed)
    batches_per_epoch = max(1, int(np.ceil(len(observations) / 256)))
    pretrain_loss = bc_gradient_steps(
        model,
        vector_env,
        observations,
        actions,
        gradient_steps=args.bc_epochs * batches_per_epoch,
        batch_size=256,
        weight=1.0,
        rng=rng,
    )
    model.save(args.output_dir / "bc_initial_model")
    vector_env.save(args.output_dir / "bc_vec_normalize.pkl")

    before_rl = evaluate(model, vector_env, episodes=1, deterministic=True)
    with (args.output_dir / "bc_evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(before_rl, file, indent=2)
    print("BC deterministic:", before_rl[0])
    vector_env.training = True
    vector_env.norm_reward = False

    demo_callback = DecayingBehaviorCloning(
        observations,
        actions,
        vector_env,
        total_timesteps=args.timesteps,
        initial_weight=args.bc_initial_weight,
        final_weight=args.bc_final_weight,
        gradient_steps=args.bc_steps_per_rollout,
        batch_size=256,
        seed=args.seed + 10_000,
    )
    checkpoint = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=str(args.output_dir / "checkpoints"),
        name_prefix="timegoalv2_trial2",
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList([checkpoint, demo_callback]),
    )
    model.save(args.output_dir / "final_model")
    vector_env.save(args.output_dir / "vec_normalize.pkl")

    deterministic = evaluate(model, vector_env, episodes=1, deterministic=True)
    stochastic = evaluate(model, vector_env, episodes=20, deterministic=False)
    evaluation = deterministic + stochastic
    with (args.output_dir / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(evaluation, file, indent=2)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args) | {"output_dir": str(args.output_dir), "deadline": DEADLINE}, file, indent=2)
    vector_env.close()
    successes = sum(record["success"] for record in stochastic)
    print("final deterministic:", deterministic[0])
    print(f"final stochastic success: {successes}/{len(stochastic)}")


if __name__ == "__main__":
    main()

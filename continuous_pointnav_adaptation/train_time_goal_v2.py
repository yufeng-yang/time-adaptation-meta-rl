"""Train a direct continuous-action PPO policy on TimeGoalV2 at D=250."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from continuous_pointnav_adaptation.envs import TimeGoalV2Env


HERE = Path(__file__).resolve().parent
DEADLINE = 250


class RewardOnlyWrapper(gym.Wrapper):
    """Drop the unused Safety-Gymnasium cost from the step API."""

    render_mode = None

    def get_wrapper_attr(self, name: str):
        """Compatibility with newer SB3 and the installed Gymnasium version."""

        if name == "render_mode":
            return self.render_mode
        return getattr(self, name)

    def step(self, action: np.ndarray):
        observation, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["safety_cost"] = float(cost)
        return observation, reward, terminated, truncated, info


def make_env() -> RewardOnlyWrapper:
    return RewardOnlyWrapper(
        TimeGoalV2Env(
            deadlines=(DEADLINE,),
            collision_penalty=0.1,
            progress_reward_scale=0.2,
        )
    )


def evaluate(
    model: PPO,
    normalizer: VecNormalize,
    *,
    episodes: int,
    deterministic: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    normalizer.training = False
    normalizer.norm_reward = False
    for _ in range(episodes):
        observation = normalizer.reset()
        total_reward = 0.0
        steps = 0
        final_info: dict[str, Any] = {}
        done = np.asarray([False])
        while not bool(done[0]):
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, reward, done, infos = normalizer.step(action)
            total_reward += float(reward[0])
            steps += 1
            final_info = infos[0]
        records.append(
            {
                "deterministic": deterministic,
                "success": bool(final_info.get("success", False)),
                "timeout": bool(final_info.get("timeout", False)),
                "steps": steps,
                "return": total_reward,
                "danger_visits": int(final_info.get("danger_visits", 0)),
                "goal_distance": float(final_info.get("goal_distance", np.nan)),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "trial1"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checked = make_env()
    check_env(checked, warn=True)
    checked.close()

    def monitored_env(rank: int):
        def build() -> Monitor:
            env = make_env()
            env.reset(seed=args.seed + rank)
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
    checkpoint = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=str(args.output_dir / "checkpoints"),
        name_prefix="timegoalv2_d250",
    )
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
    model.learn(total_timesteps=args.timesteps, callback=checkpoint)
    model.save(args.output_dir / "final_model")
    vector_env.save(args.output_dir / "vec_normalize.pkl")

    deterministic = evaluate(model, vector_env, episodes=1, deterministic=True)
    stochastic = evaluate(model, vector_env, episodes=20, deterministic=False)
    evaluation = deterministic + stochastic
    with (args.output_dir / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(evaluation, file, indent=2)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "deadline": DEADLINE,
                "timesteps": args.timesteps,
                "seed": args.seed,
                "n_envs": args.n_envs,
                "step_penalty": 0.001,
                "progress_reward_scale": 0.2,
                "goal_reward": 1.0,
                "danger_penalty": 0.2,
                "collision_penalty": 0.1,
                "timeout_penalty": 1.0,
            },
            file,
            indent=2,
        )
    vector_env.close()

    successes = sum(record["success"] for record in stochastic)
    print("deterministic:", deterministic[0])
    print(f"stochastic success: {successes}/{len(stochastic)}")


if __name__ == "__main__":
    main()

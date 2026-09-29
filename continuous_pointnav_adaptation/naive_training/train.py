"""Train a single end-to-end PPO policy from continuous actions."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from continuous_pointnav_adaptation.naive_training.environment import make_naive_env


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checked = make_naive_env()
    check_env(checked, warn=True)
    checked.close()

    env = Monitor(
        make_naive_env(),
        filename=str(args.output_dir / "train.monitor.csv"),
        info_keywords=("gate_state", "success", "timeout", "danger_visits"),
    )
    callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(args.output_dir / "checkpoints"),
        name_prefix="naive_ppo",
    )
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(args.output_dir / "tensorboard"),
        policy_kwargs={"net_arch": [256, 256]},
    )
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(args.output_dir / "final_model")
    finally:
        env.close()


if __name__ == "__main__":
    main()

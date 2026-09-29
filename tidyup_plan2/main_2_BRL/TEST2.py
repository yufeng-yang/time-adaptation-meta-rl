"""Closed D=17 control experiment with no cross-episode trial mechanism.

This keeps the same exact-belief sensor and PPO architecture as TEST.py, but
sets episodes_per_trial=1. Every base episode is therefore a real PPO terminal;
the next reset starts a new independent episode with P(open)=0.5.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
WORKSPACE_ROOT = PACKAGE_DIR.parent
LOCAL_MINIGRID_DIR = WORKSPACE_ROOT / "Minigrid"
for path in (WORKSPACE_ROOT, LOCAL_MINIGRID_DIR, PACKAGE_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from belief_trial_env import ExactBeliefTrialEnv
from tidyup_plan2.envs import ThreeRouteHiddenGateEnvV3


GATE_STATE = "closed"
DEADLINE = 17
PRIOR_OPEN = 0.5
EPISODES_PER_TRIAL = 1
DEFAULT_MODEL_PATH = (
    SCRIPT_DIR
    / "artifacts"
    / "test2_closed_d17_no_trial"
    / "brl_closed_d17_no_trial"
)


def make_env(args: argparse.Namespace, *, monitor: bool) -> gym.Env:
    base_env = ThreeRouteHiddenGateEnvV3(
        min_deadline=DEADLINE,
        max_deadline=DEADLINE,
        step_penalty=args.step_penalty,
        progress_scale=args.progress_scale,
        danger_penalty=args.danger_penalty,
        defer_danger_penalty=True,
    )
    # With exactly one subepisode, the wrapper never performs an internal
    # cross-episode reset: each base termination is immediately returned to PPO.
    env: gym.Env = ExactBeliefTrialEnv(
        base_env,
        deadlines=(DEADLINE,),
        episodes_per_trial=EPISODES_PER_TRIAL,
        prior_open=PRIOR_OPEN,
        fixed_gate_state=GATE_STATE,
    )
    if monitor:
        artifact_dir = Path(args.model_path).parent
        artifact_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(
            env,
            filename=str(artifact_dir / "episode_monitor.csv"),
            info_keywords=("gate_state", "belief_open", "trial_return"),
        )
    return env


def build_model(env: gym.Env, args: argparse.Namespace) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.ppo_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        policy_kwargs={"net_arch": {"pi": [256, 128], "vf": [256, 128]}},
        seed=args.seed,
        verbose=1,
        device=args.device,
    )


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def evaluate(model: PPO, args: argparse.Namespace) -> dict:
    env = make_env(args, monitor=False)
    episodes: list[dict] = []
    try:
        for episode_index in range(args.eval_episodes):
            observation, initial_info = env.reset(seed=args.seed + episode_index)
            terminated = truncated = False
            episode_return = 0.0
            final_info: dict = {}
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, final_info = env.step(
                    int(action)
                )
                episode_return += float(reward)

            base_summary = final_info["subepisodes"][0]
            episodes.append(
                {
                    "episode": episode_index + 1,
                    "initial_belief_open": float(initial_info["belief_open"]),
                    "final_belief_open": float(final_info["belief_open"]),
                    "return": episode_return,
                    "success": bool(base_summary["success"]),
                    "timeout": bool(base_summary["timeout"]),
                }
            )
    finally:
        env.close()

    successes = sum(int(item["success"]) for item in episodes)
    mean_return = sum(float(item["return"]) for item in episodes) / len(episodes)
    result = {
        "gate_state": GATE_STATE,
        "deadline": DEADLINE,
        "trial_mechanism": False,
        "initial_belief_open": PRIOR_OPEN,
        "successes": successes,
        "eval_episodes": args.eval_episodes,
        "success_rate": successes / args.eval_episodes,
        "mean_return": mean_return,
        "episodes": episodes,
    }
    print(
        "\nClosed D=17 evaluation without trial mechanism\n"
        f"  initial belief P(open)={PRIOR_OPEN:.1f} every episode\n"
        f"  successes={successes}/{args.eval_episodes}\n"
        f"  success rate={result['success_rate']:.1%}\n"
        f"  mean return={mean_return:.3f}"
    )
    return result


def train(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    save_json(
        model_path.parent / "config.json",
        {
            **vars(args),
            "gate_state": GATE_STATE,
            "deadline": DEADLINE,
            "initial_belief_open": PRIOR_OPEN,
            "episodes_per_trial": EPISODES_PER_TRIAL,
            "trial_mechanism": False,
            "initialization": "random",
            "policy_gate_access": False,
            "danger_penalty_timing": "episode_end",
        },
    )
    env = make_env(args, monitor=True)
    check_env(env, warn=True)
    model = build_model(env, args)
    callback = CheckpointCallback(
        save_freq=max(args.checkpoint_frequency, 1),
        save_path=str(model_path.parent / "checkpoints"),
        name_prefix="brl_closed_d17_no_trial",
    )
    try:
        print(
            f"Training from scratch: gate={GATE_STATE}, D={DEADLINE}\n"
            "  trial mechanism=False (episodes_per_trial=1)\n"
            f"  timesteps={args.timesteps}\n"
            f"  belief resets to P(open)={PRIOR_OPEN} every episode\n"
            f"  danger penalty=-{args.danger_penalty} at episode end\n"
            "  real gate state is not included in the policy observation"
        )
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(str(model_path))
        print(f"Saved model to {model_path.with_suffix('.zip')}")
    finally:
        env.close()

    result = evaluate(model, args)
    save_json(model_path.parent / "evaluation.json", result)


def evaluate_saved(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    saved_path = model_path.with_suffix(".zip")
    if not saved_path.exists():
        raise FileNotFoundError(f"model not found: {saved_path}")
    env = make_env(args, monitor=False)
    try:
        model = PPO.load(str(model_path), env=env, device=args.device)
        result = evaluate(model, args)
    finally:
        env.close()
    save_json(model_path.parent / "evaluation.json", result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Closed D=17 PPO control experiment without trials"
    )
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--progress-scale", type=float, default=0.02)
    parser.add_argument("--danger-penalty", type=float, default=0.1)
    parser.add_argument("--checkpoint-frequency", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if args.eval_episodes <= 0:
        parser.error("--eval-episodes must be positive")
    if args.n_steps <= 0:
        parser.error("--n-steps must be positive")
    if args.batch_size <= 0 or args.batch_size > args.n_steps:
        parser.error("--batch-size must be in [1, n-steps]")
    if args.danger_penalty < 0:
        parser.error("--danger-penalty must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate_saved(args)


if __name__ == "__main__":
    main()

"""Train BRL from scratch with a closed-only stage and joint fine-tuning.

Both stages begin every new trial with b_0=P(open)=0.5.  The closed curriculum
fixes only the environment's hidden task; it does not reveal that choice in the
policy observation.  Belief changes to 0/1 only when the local sensor sees the
door and is retained across subepisodes in the same trial.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn, update_learning_rate


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
WORKSPACE_ROOT = PACKAGE_DIR.parent
LOCAL_MINIGRID_DIR = WORKSPACE_ROOT / "Minigrid"
for path in (WORKSPACE_ROOT, LOCAL_MINIGRID_DIR, PACKAGE_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from belief_trial_env import ExactBeliefTrialEnv
from tidyup_plan2.envs import ThreeRouteHiddenGateEnvV3
from train_brl import evaluate


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "staged"
ALL_DEADLINES = (15, 17, 19)
CLOSED_DEADLINES = (17, 19)


def make_stage_env(
    args: argparse.Namespace,
    *,
    deadlines: tuple[int, ...],
    fixed_gate_state: str | None,
    monitor_name: str,
) -> gym.Env:
    base_env = ThreeRouteHiddenGateEnvV3(
        min_deadline=min(ALL_DEADLINES),
        max_deadline=max(ALL_DEADLINES),
        step_penalty=args.step_penalty,
        progress_scale=args.progress_scale,
    )
    env: gym.Env = ExactBeliefTrialEnv(
        base_env,
        deadlines=deadlines,
        episodes_per_trial=args.episodes_per_trial,
        # This stays 0.5 even when the curriculum fixes the hidden gate.
        prior_open=0.5,
        fixed_gate_state=fixed_gate_state,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return Monitor(
        env,
        filename=str(output_dir / monitor_name),
        info_keywords=("gate_state", "belief_open", "trial_return"),
    )


def set_learning_rate(model: PPO, learning_rate: float) -> None:
    model.learning_rate = float(learning_rate)
    model.lr_schedule = get_schedule_fn(float(learning_rate))
    update_learning_rate(model.policy.optimizer, float(learning_rate))


def build_fresh_model(env: gym.Env, args: argparse.Namespace) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=args.closed_learning_rate,
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


def evaluation_args(args: argparse.Namespace, model_path: Path) -> argparse.Namespace:
    # ``train_brl.evaluate`` needs this common subset of arguments.
    return argparse.Namespace(
        deadlines=ALL_DEADLINES,
        episodes_per_trial=args.episodes_per_trial,
        prior_open=0.5,
        step_penalty=args.step_penalty,
        progress_scale=args.progress_scale,
        model_path=str(model_path),
        seed=args.seed,
        device=args.device,
    )


def run_stage(
    model: PPO,
    args: argparse.Namespace,
    *,
    name: str,
    steps: int,
    learning_rate: float,
    deadlines: tuple[int, ...],
    fixed_gate_state: str | None,
) -> PPO:
    if steps <= 0:
        print(f"Skipping {name}: steps={steps}")
        return model

    output_dir = Path(args.output_dir)
    env = make_stage_env(
        args,
        deadlines=deadlines,
        fixed_gate_state=fixed_gate_state,
        monitor_name=f"{name}_monitor.csv",
    )
    model.set_env(env)
    set_learning_rate(model, learning_rate)
    callback = CheckpointCallback(
        save_freq=max(args.checkpoint_frequency, 1),
        save_path=str(output_dir / "checkpoints" / name),
        name_prefix=f"brl_{name}",
    )
    print(
        f"\nStage {name}\n"
        f"  steps={steps}\n"
        f"  deadlines={deadlines}\n"
        f"  environment gate={fixed_gate_state or 'uniform open/closed'}\n"
        "  policy initial belief=0.5"
    )
    try:
        model.learn(
            total_timesteps=steps,
            callback=callback,
            reset_num_timesteps=False,
        )
        stage_path = output_dir / f"brl_{name}"
        model.save(str(stage_path))
        results = evaluate(model, evaluation_args(args, stage_path))
        save_json(output_dir / f"{name}_evaluation.json", results)
        print(f"Saved stage model to {stage_path.with_suffix('.zip')}")
    finally:
        env.close()
    return model


def train(args: argparse.Namespace) -> None:
    initial_model = Path(args.initial_model) if args.initial_model else None
    if initial_model is not None and not initial_model.exists():
        raise FileNotFoundError(f"initial model not found: {initial_model}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "config.json",
        {
            **vars(args),
            "initial_model": str(initial_model) if initial_model else None,
            "initialization": "checkpoint" if initial_model else "random",
            "output_dir": str(output_dir),
            "stages": [
                {
                    "name": "closed",
                    "gate": "closed",
                    "deadlines": list(CLOSED_DEADLINES),
                    "initial_belief": 0.5,
                },
                {
                    "name": "joint",
                    "gate": "uniform open/closed",
                    "deadlines": list(ALL_DEADLINES),
                    "initial_belief": 0.5,
                },
            ],
        },
    )

    bootstrap_env = make_stage_env(
        args,
        deadlines=CLOSED_DEADLINES,
        fixed_gate_state="closed",
        monitor_name="bootstrap_monitor.csv",
    )
    try:
        if initial_model is None:
            print("Initializing a new PPO policy from scratch")
            model = build_fresh_model(bootstrap_env, args)
        else:
            print(f"Explicitly resuming from {initial_model}")
            model = PPO.load(str(initial_model), env=bootstrap_env, device=args.device)
    finally:
        bootstrap_env.close()

    model = run_stage(
        model,
        args,
        name="closed",
        steps=args.closed_steps,
        learning_rate=args.closed_learning_rate,
        deadlines=CLOSED_DEADLINES,
        fixed_gate_state="closed",
    )
    model = run_stage(
        model,
        args,
        name="joint",
        steps=args.joint_steps,
        learning_rate=args.joint_learning_rate,
        deadlines=ALL_DEADLINES,
        fixed_gate_state=None,
    )

    final_path = output_dir / "exact_belief_ppo_staged"
    model.save(str(final_path))
    results = evaluate(model, evaluation_args(args, final_path))
    save_json(output_dir / "evaluation.json", results)
    print(f"\nFinal staged model: {final_path.with_suffix('.zip')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train exact-belief PPO with closed then joint curriculum"
    )
    parser.add_argument(
        "--initial-model",
        default=None,
        help=(
            "optional checkpoint to resume; omitted by default so training "
            "starts from a randomly initialized policy"
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--closed-steps", type=int, default=200_000)
    parser.add_argument("--joint-steps", type=int, default=300_000)
    parser.add_argument("--closed-learning-rate", type=float, default=3e-4)
    parser.add_argument("--joint-learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--episodes-per-trial", type=int, default=5)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--progress-scale", type=float, default=0.02)
    parser.add_argument("--checkpoint-frequency", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.closed_steps < 0 or args.joint_steps < 0:
        parser.error("stage step counts must be non-negative")
    if args.closed_steps + args.joint_steps <= 0:
        parser.error("at least one stage must have positive steps")
    if args.episodes_per_trial <= 0:
        parser.error("--episodes-per-trial must be positive")
    if args.n_steps <= 0:
        parser.error("--n-steps must be positive")
    if args.batch_size <= 0 or args.batch_size > args.n_steps:
        parser.error("--batch-size must be in [1, n-steps]")
    return args


if __name__ == "__main__":
    train(parse_args())

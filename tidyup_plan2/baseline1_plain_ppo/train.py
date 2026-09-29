"""V4 PPO control: no remaining-time input, deadline fixed at 500.

The policy sees only the 5x5 semantic map and heading.  The door open/closed
channels are collapsed, matching the deadline-only baseline.  Every episode
uses gate=closed and D=500, so time pressure is removed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
WORKSPACE_ROOT = PACKAGE_DIR.parent
MINIGRID_DIR = WORKSPACE_ROOT / "Minigrid"
for path in (WORKSPACE_ROOT, MINIGRID_DIR, PACKAGE_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tidyup_plan2.envs import ThreeRouteHiddenGateEnvV4

CELL_EMPTY = 0
CELL_WALL = 1
CELL_GOAL = 2
CELL_DOOR_CLOSED = 4
CELL_AGENT = 5
N_CELL_TYPES = 6
VIEW_SIZE = 5
OBJECT_WALL = 2
OBJECT_DOOR = 4
OBJECT_GOAL = 8


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts"
FIXED_DEADLINE = 500
FIXED_GATE = "closed"
EVAL_DEADLINES = (15, 17, 19, FIXED_DEADLINE)
STEP_PENALTY = 0.01
PROGRESS_SCALE = 0.0


def make_base_env(*, render_mode: str | None = None) -> gym.Env:
    return ThreeRouteHiddenGateEnvV4(
        render_mode=render_mode,
        min_deadline=FIXED_DEADLINE,
        max_deadline=FIXED_DEADLINE,
        step_penalty=STEP_PENALTY,
        progress_scale=PROGRESS_SCALE,
    )


def classify_route(positions: list[tuple[int, int]]) -> str:
    if ThreeRouteHiddenGateEnvV4.GATE_POSITION in positions:
        return "direct"
    if any(
        position in ThreeRouteHiddenGateEnvV4.DANGER_POSITIONS
        for position in positions
    ):
        return "medium"
    if ThreeRouteHiddenGateEnvV4.GOAL_POSITION in positions:
        return "detour"
    return "other"


class FixedDeadlineClosed(gym.Wrapper):
    """Every reset uses D=500 and a closed gate."""

    def reset(self, *, seed=None, options=None):
        fixed_options = dict(options or {})
        fixed_options.update(
            {
                "deadline": FIXED_DEADLINE,
                "gate_state": FIXED_GATE,
                "new_trial": True,
            }
        )
        return self.env.reset(seed=seed, options=fixed_options)


class NoTimeObservation(gym.ObservationWrapper):
    """150-D cell one-hot + 4 direction one-hot. No D-t, no door state."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        sensor_dim = VIEW_SIZE * VIEW_SIZE * N_CELL_TYPES
        observation_dim = sensor_dim + 4
        self.observation_space = spaces.Box(
            low=np.zeros(observation_dim, dtype=np.float32),
            high=np.ones(observation_dim, dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, observation):
        image = np.asarray(observation["image"], dtype=np.uint8)
        objects = image[..., 0]

        cells = np.full((VIEW_SIZE, VIEW_SIZE), CELL_EMPTY, dtype=np.int64)
        cells[objects == OBJECT_WALL] = CELL_WALL
        cells[objects == OBJECT_GOAL] = CELL_GOAL
        cells[objects == OBJECT_DOOR] = CELL_DOOR_CLOSED
        cells[VIEW_SIZE // 2, VIEW_SIZE // 2] = CELL_AGENT
        local_sensor = np.eye(N_CELL_TYPES, dtype=np.float32)[cells].reshape(-1)

        direction = np.zeros(4, dtype=np.float32)
        direction[int(observation["direction"])] = 1.0
        return np.concatenate([local_sensor, direction]).astype(np.float32)


def make_training_env(output_dir: Path) -> gym.Env:
    env: gym.Env = make_base_env()
    env = FixedDeadlineClosed(env)
    env = NoTimeObservation(env)
    return Monitor(
        env,
        filename=str(output_dir / "train.monitor.csv"),
        info_keywords=(
            "deadline",
            "gate_state",
            "success",
            "timeout",
            "collision",
        ),
    )


def make_evaluation_env() -> gym.Env:
    return NoTimeObservation(make_base_env())


def initial_action_probabilities(model: PPO, observation) -> np.ndarray:
    observation_tensor, _ = model.policy.obs_to_tensor(observation)
    distribution = model.policy.get_distribution(observation_tensor)
    return distribution.distribution.probs.detach().cpu().numpy()[0]


def evaluate(model: PPO, seed: int) -> list[dict]:
    env = make_evaluation_env()
    results: list[dict] = []
    print(
        "\ngate   D    success timeout return steps route   "
        "P(left) P(right) P(forward)"
    )
    print("-" * 84)
    try:
        for gate_state in ("open", "closed"):
            for deadline in EVAL_DEADLINES:
                observation, _ = env.reset(
                    seed=seed,
                    options={
                        "new_trial": True,
                        "deadline": deadline,
                        "gate_state": gate_state,
                    },
                )
                probabilities = initial_action_probabilities(model, observation)
                positions = [tuple(map(int, env.unwrapped.agent_pos))]
                actions = []
                episode_return = 0.0
                terminated = truncated = False
                final_info = {}
                while not (terminated or truncated):
                    action, _ = model.predict(observation, deterministic=True)
                    actions.append(int(action))
                    observation, reward, terminated, truncated, final_info = (
                        env.step(int(action))
                    )
                    episode_return += float(reward)
                    positions.append(tuple(map(int, env.unwrapped.agent_pos)))
                record = {
                    "gate_state": gate_state,
                    "deadline": deadline,
                    "success": bool(final_info["success"]),
                    "timeout": bool(final_info["timeout"]),
                    "collision": bool(final_info["collision"]),
                    "return": episode_return,
                    "steps": int(env.unwrapped.step_count),
                    "route": classify_route(positions),
                    "actions": actions,
                    "positions": positions,
                    "initial_action_probabilities": probabilities.tolist(),
                }
                results.append(record)
                print(
                    f"{gate_state:6s} {deadline:3d} "
                    f"{str(record['success']):>7s} "
                    f"{str(record['timeout']):>7s} "
                    f"{episode_return:6.2f} "
                    f"{record['steps']:5d} "
                    f"{record['route']:7s} "
                    f"{probabilities[0]:7.3f} "
                    f"{probabilities[1]:8.3f} "
                    f"{probabilities[2]:10.3f}"
                )
    finally:
        env.close()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--checkpoint-frequency", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("--timesteps must be positive")
    if args.n_steps <= 0:
        parser.error("--n-steps must be positive")
    if args.batch_size <= 0 or args.batch_size > args.n_steps:
        parser.error("--batch-size must be in [1, n-steps]")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    check_env(NoTimeObservation(make_base_env()), warn=True)
    train_env = make_training_env(output_dir)
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.ppo_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        seed=args.seed,
        verbose=1,
        device=args.device,
    )
    callback = CheckpointCallback(
        save_freq=max(args.checkpoint_frequency, 1),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="plain_ppo",
    )
    try:
        print(
            f"\nTraining PPO for {args.timesteps} steps\n"
            f"  D={FIXED_DEADLINE}, gate={FIXED_GATE}\n"
            "  observation=5x5 sensor + direction (no D-t, no door state)\n"
            "  env=ThreeRouteHiddenGateEnvV4, progress_scale=0"
        )
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            reset_num_timesteps=True,
        )
        model.save(str(output_dir / "final_model"))
        results = evaluate(model, seed=args.seed)
        with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    finally:
        train_env.close()


if __name__ == "__main__":
    main()

"""Hierarchical exact-belief PPO for the continuous PointNav environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from continuous_pointnav_adaptation.envs import PointThreeRouteGateEnv
from continuous_pointnav_adaptation.options import ROUTE_NAMES, execute_option


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts_hierarchical"
DEADLINES = (195, 255, 275)
EPISODES_PER_TRIAL = 3
GATE_SENSOR_RANGE = 1.5


def route_observed_gate(positions: list[list[float]]) -> bool:
    """Whether a continuous option legally entered the gate sensor range."""

    gate_points = np.asarray(((0.0, -0.5), (0.0, 0.0), (0.0, 0.5)))
    trajectory = np.asarray(positions, dtype=np.float64)
    distances = np.linalg.norm(
        trajectory[:, None, :] - gate_points[None, :, :], axis=-1
    )
    return bool(np.min(distances) <= GATE_SENSOR_RANGE)


def build_option_table() -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Measure every frozen option once in the real MuJoCo environment."""

    env = PointThreeRouteGateEnv(
        render_mode=None,
        deadlines=DEADLINES,
        step_penalty=0.001,
        goal_reward=1.0,
        collision_penalty=0.05,
        timeout_penalty=1.0,
        danger_penalty=0.2,
    )
    table: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    try:
        for gate_state in ("open", "closed"):
            table[gate_state] = {}
            for deadline in DEADLINES:
                table[gate_state][str(deadline)] = {}
                for route in ROUTE_NAMES:
                    env.reset(
                        seed=0,
                        options={
                            "new_trial": True,
                            "gate_state": gate_state,
                            "deadline": deadline,
                        },
                    )
                    result = execute_option(env, route)
                    table[gate_state][str(deadline)][route] = {
                        "route": route,
                        "success": bool(result["success"]),
                        "timeout": bool(result["timeout"]),
                        "return": float(result["return"]),
                        "steps": int(result["steps"]),
                        "danger_visits": int(result["danger_visits"]),
                        "collision_events": int(result["collision_events"]),
                        "observed_gate": route_observed_gate(result["positions"]),
                    }
    finally:
        env.close()
    return table


class ContinuousBeliefTrial(gym.Env[np.ndarray, int]):
    """Three-route SMDP with a persistent hidden gate over three episodes."""

    metadata = {"render_modes": []}
    render_mode = None

    def __init__(
        self,
        option_table: dict[str, dict[str, dict[str, dict[str, Any]]]],
        *,
        seed: int,
        prior_open: float,
        training: bool,
    ) -> None:
        super().__init__()
        self.option_table = option_table
        self.action_space = spaces.Discrete(len(ROUTE_NAMES))
        # deadline one-hot + belief(open) + episode index one-hot
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self._rng = np.random.default_rng(seed)
        self.prior_open = float(prior_open)
        self.training = bool(training)
        self._deadline = DEADLINES[0]
        self._gate_state = "open"
        self._belief_open = self.prior_open
        self._episode_index = 0
        self._trial_return = 0.0
        self._subepisodes: list[dict[str, Any]] = []

    def _observation(self) -> np.ndarray:
        deadline = np.zeros(len(DEADLINES), dtype=np.float32)
        deadline[DEADLINES.index(self._deadline)] = 1.0
        episode = np.zeros(EPISODES_PER_TRIAL, dtype=np.float32)
        episode[self._episode_index] = 1.0
        return np.concatenate(
            [deadline, np.asarray([self._belief_open], np.float32), episode]
        ).astype(np.float32)

    def _info(self, **extra: Any) -> dict[str, Any]:
        info: dict[str, Any] = {
            "deadline": self._deadline,
            "gate_state": self._gate_state,
            "belief_open": self._belief_open,
            "trial_return": self._trial_return,
            "subepisode_index": self._episode_index + 1,
        }
        info.update(extra)
        return info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = dict(options or {})
        self._deadline = int(
            options.get("deadline", self._rng.choice(DEADLINES))
        )
        self._gate_state = str(
            options.get("gate_state", self._rng.choice(("open", "closed")))
        )
        if self._deadline not in DEADLINES:
            raise ValueError(f"deadline must be one of {DEADLINES}")
        if self._gate_state not in ("open", "closed"):
            raise ValueError("gate_state must be open or closed")
        self._belief_open = self.prior_open
        self._episode_index = 0
        self._trial_return = 0.0
        self._subepisodes = []
        return self._observation(), self._info()

    def step(self, action: int):
        route_index = int(action)
        if not self.action_space.contains(route_index):
            raise ValueError(f"invalid route option {route_index}")
        route = ROUTE_NAMES[route_index]
        belief_start = self._belief_open
        result = self.option_table[self._gate_state][str(self._deadline)][route]
        if result["observed_gate"]:
            self._belief_open = 1.0 if self._gate_state == "open" else 0.0

        summary = {
            "episode": self._episode_index + 1,
            "deadline": self._deadline,
            "route": route,
            "success": result["success"],
            "timeout": result["timeout"],
            "return": result["return"],
            "steps": result["steps"],
            "danger_visits": result["danger_visits"],
            "collision_events": result["collision_events"],
            "belief_start": belief_start,
            "belief_end": self._belief_open,
        }
        self._subepisodes.append(summary)
        self._trial_return += float(result["return"])

        trial_done = self._episode_index + 1 >= EPISODES_PER_TRIAL
        if trial_done:
            info = self._info(
                subepisode_done=True,
                subepisodes=list(self._subepisodes),
                selected_route=route,
                success=result["success"],
            )
            return self._observation(), result["return"], True, False, info

        self._episode_index += 1
        info = self._info(
            subepisode_done=True,
            completed_subepisode=summary,
            selected_route=route,
            success=result["success"],
        )
        return self._observation(), result["return"], False, False, info

    def close(self) -> None:
        return None


def evaluate(
    model: PPO,
    args: argparse.Namespace,
    option_table: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> list[dict[str, Any]]:
    env = ContinuousBeliefTrial(
        option_table,
        seed=args.seed,
        prior_open=args.prior_open,
        training=False,
    )
    records: list[dict[str, Any]] = []
    print("\ngate   D ep success return steps route  danger b_start b_end")
    print("-" * 76)
    try:
        for gate_state in ("open", "closed"):
            for deadline in DEADLINES:
                observation, _ = env.reset(
                    seed=args.seed,
                    options={"deadline": deadline, "gate_state": gate_state},
                )
                for episode in range(1, EPISODES_PER_TRIAL + 1):
                    action, _ = model.predict(observation, deterministic=True)
                    observation, _, terminated, truncated, info = env.step(int(action))
                    completed = (
                        info["subepisodes"][-1]
                        if terminated or truncated
                        else info["completed_subepisode"]
                    )
                    record = {"gate_state": gate_state, **completed}
                    records.append(record)
                    print(
                        f"{gate_state:6s} {deadline:3d} {episode:2d} "
                        f"{str(record['success']):>7s} {record['return']:6.3f} "
                        f"{record['steps']:5d} {record['route']:7s} "
                        f"{record['danger_visits']:6d} "
                        f"{record['belief_start']:7.2f} {record['belief_end']:5.2f}"
                    )
    finally:
        env.close()
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--prior-open", type=float, default=0.5)
    parser.add_argument("--checkpoint-frequency", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    if args.timesteps <= 0 or args.n_steps <= 0:
        parser.error("timesteps and n-steps must be positive")
    if args.batch_size <= 0 or args.batch_size > args.n_steps:
        parser.error("batch-size must be in [1, n-steps]")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Measuring 18 frozen options in MuJoCo...")
    option_table = build_option_table()
    with (output_dir / "option_table.json").open("w", encoding="utf-8") as handle:
        json.dump(option_table, handle, indent=2)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "deadlines": DEADLINES,
                "episodes_per_trial": EPISODES_PER_TRIAL,
                "routes": ROUTE_NAMES,
                "gate_sensor_range": GATE_SENSOR_RANGE,
                "reward": {
                    "step": -0.001,
                    "goal": 1.0,
                    "collision_event": -0.05,
                    "timeout": -1.0,
                    "danger_once": -0.2,
                    "safety_cost_used": False,
                },
            },
            handle,
            indent=2,
        )

    checked_env = ContinuousBeliefTrial(
        option_table,
        seed=args.seed,
        prior_open=args.prior_open,
        training=False,
    )
    check_env(checked_env, warn=True)
    checked_env.close()
    raw_env = ContinuousBeliefTrial(
        option_table,
        seed=args.seed,
        prior_open=args.prior_open,
        training=True,
    )
    train_env: gym.Env = Monitor(
        raw_env,
        filename=str(output_dir / "train.monitor.csv"),
        info_keywords=("deadline", "gate_state", "belief_open", "trial_return"),
    )
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
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        seed=args.seed,
        verbose=1,
        device=args.device,
    )
    callback = CheckpointCallback(
        save_freq=max(args.checkpoint_frequency, 1),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="continuous_hierarchical_ppo",
    )
    try:
        print(
            f"\nTraining continuous hierarchical PPO for {args.timesteps} "
            "route decisions\n"
            f"  deadlines={DEADLINES}, routes={ROUTE_NAMES}\n"
            "  one trial = 3 episodes with a fixed gate and deadline"
        )
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(str(output_dir / "final_model"))
    finally:
        train_env.close()

    results = evaluate(model, args, option_table)
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()

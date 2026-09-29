"""Baseline 0: feed-forward PPO observing only normalized D-t."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for path in (ROOT, ROOT / "Minigrid", HERE.parent, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tidyup_plan2.envs import ThreeRouteHiddenGateEnvV4

DEADLINES = (15, 17, 19)
MAX_DEADLINE = max(DEADLINES)
DEFAULT_OUTPUT = HERE / "artifacts"


class TimeOnlyObservation(gym.ObservationWrapper):
    """Expose only D-t as one normalized scalar."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.asarray([0.0], dtype=np.float32),
            high=np.asarray([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, observation: dict[str, Any]) -> np.ndarray:
        remaining = float(np.asarray(observation["time"])[0])
        return np.asarray([remaining / MAX_DEADLINE], dtype=np.float32)


class RandomTaskReset(gym.Wrapper):
    """Sample exactly D={15,17,19}; base env samples the gate 50/50."""

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if "deadline" not in options:
            # Seed the base environment first, then use its RNG for reproducibility.
            if seed is not None:
                self.env.unwrapped.np_random = np.random.default_rng(seed)
            options["deadline"] = int(self.env.unwrapped.np_random.choice(DEADLINES))
        options.setdefault("new_trial", True)
        return self.env.reset(seed=seed, options=options)


def make_base(args: argparse.Namespace) -> gym.Env:
    return ThreeRouteHiddenGateEnvV4(
        min_deadline=min(DEADLINES),
        max_deadline=max(DEADLINES),
        step_penalty=args.step_penalty,
        progress_scale=0.0,
    )


def make_training_env(args: argparse.Namespace, output: Path) -> gym.Env:
    env: gym.Env = RandomTaskReset(make_base(args))
    env = TimeOnlyObservation(env)
    return Monitor(
        env,
        filename=str(output / "train.monitor.csv"),
        info_keywords=("deadline", "gate_state", "success", "timeout", "collision"),
    )


def classify_route(positions: list[tuple[int, int]]) -> str:
    if ThreeRouteHiddenGateEnvV4.GATE_POSITION in positions:
        return "direct"
    if any(p in ThreeRouteHiddenGateEnvV4.DANGER_POSITIONS for p in positions):
        return "medium"
    if any(p in {(3, 5), (3, 6), (4, 6), (5, 6)} for p in positions):
        return "detour"
    return "other"


def action_probabilities(model: PPO, observation: np.ndarray) -> list[float]:
    tensor, _ = model.policy.obs_to_tensor(observation)
    distribution = model.policy.get_distribution(tensor)
    return distribution.distribution.probs.detach().cpu().numpy()[0].tolist()


def evaluate(model: PPO, args: argparse.Namespace) -> list[dict[str, Any]]:
    env = TimeOnlyObservation(make_base(args))
    results: list[dict[str, Any]] = []
    try:
        for gate in ("open", "closed"):
            for deadline in DEADLINES:
                for trial in range(args.eval_trials_per_case):
                    observation, _ = env.reset(
                        seed=args.eval_seed + trial,
                        options={"new_trial": True, "gate_state": gate, "deadline": deadline},
                    )
                    initial_probs = action_probabilities(model, observation)
                    positions = [tuple(map(int, env.unwrapped.agent_pos))]
                    actions: list[int] = []
                    total = 0.0
                    done = False
                    final_info: dict[str, Any] = {}
                    while not done:
                        action, _ = model.predict(observation, deterministic=True)
                        action = int(action)
                        observation, reward, terminated, truncated, final_info = env.step(action)
                        done = bool(terminated or truncated)
                        total += float(reward)
                        actions.append(action)
                        positions.append(tuple(map(int, env.unwrapped.agent_pos)))
                    results.append({
                        "gate_state": gate,
                        "deadline": deadline,
                        "trial": trial,
                        "success": bool(final_info.get("success", False)),
                        "return": total,
                        "steps": int(env.unwrapped.step_count),
                        "route": classify_route(positions),
                        "actions": actions,
                        "positions": positions,
                        "initial_action_probabilities": initial_probs,
                    })
    finally:
        env.close()
    return results


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for gate in ("open", "closed"):
        summary[gate] = {}
        for deadline in DEADLINES:
            subset = [r for r in results if r["gate_state"] == gate and r["deadline"] == deadline]
            summary[gate][str(deadline)] = {
                "success_rate": float(np.mean([r["success"] for r in subset])),
                "mean_return": float(np.mean([r["return"] for r in subset])),
                "mean_steps": float(np.mean([r["steps"] for r in subset])),
                "route_rates": {
                    name: float(np.mean([r["route"] == name for r in subset]))
                    for name in ("direct", "medium", "detour", "other")
                },
                "deterministic_actions": subset[0]["actions"],
            }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--eval-trials-per-case", type=int, default=20)
    parser.add_argument("--checkpoint-frequency", type=int, default=50_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2))
    env = make_training_env(args, output)
    callback = CheckpointCallback(
        save_freq=args.checkpoint_frequency,
        save_path=str(output / "checkpoints"),
        name_prefix="time_only_ppo",
    )
    model = PPO(
        "MlpPolicy", env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.ppo_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(output / "final_model")
    env.close()
    results = evaluate(model, args)
    summary = aggregate(results)
    (output / "evaluation.json").write_text(json.dumps(results, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

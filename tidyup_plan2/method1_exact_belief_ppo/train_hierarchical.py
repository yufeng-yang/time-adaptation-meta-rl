"""Train a hierarchical exact-belief route selector.

The low level consists of the three navigation options already available in
baseline2 (direct, medium/danger, and safe detour).  The learned high-level PPO
policy makes exactly one decision per base episode: which option to execute.
One trial is still three episodes with a fixed deadline and gate state, so the
return of all three decisions is optimized jointly with gamma=1.
"""

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
PACKAGE_DIR = SCRIPT_DIR.parent
WORKSPACE_ROOT = PACKAGE_DIR.parent
LOCAL_MINIGRID_DIR = WORKSPACE_ROOT / "Minigrid"
for path in (WORKSPACE_ROOT, LOCAL_MINIGRID_DIR, PACKAGE_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tidyup_plan2.envs import ThreeRouteHiddenGateEnvV4
from tidyup_plan2.method1_exact_belief_ppo.train import (
    CompactViewThreeRouteEnv,
    ExactGateBelief,
)


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts_hierarchical"
DEADLINES = (15, 17, 19)
EPISODES_PER_TRIAL = 3
ROUTE_NAMES = ("direct", "medium", "detour")
ROUTE_ACTIONS = (
    ThreeRouteHiddenGateEnvV4.DIRECT_SAFE_ACTIONS,
    ThreeRouteHiddenGateEnvV4.MEDIUM_PATH_ACTIONS,
    ThreeRouteHiddenGateEnvV4.DETOUR_SAFE_ACTIONS,
)


class HierarchicalExactBeliefTrial(gym.Env[np.ndarray, int]):
    """Three-step SMDP whose actions execute complete navigation options."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        base_env: gym.Env,
        *,
        seed: int,
        prior_open: float,
        training: bool,
    ) -> None:
        super().__init__()
        self.base_env = base_env
        self.action_space = spaces.Discrete(len(ROUTE_NAMES))
        # deadline one-hot + exact gate belief + episode-index one-hot
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self._rng = np.random.default_rng(seed)
        self.filter = ExactGateBelief(prior_open)
        self.training = bool(training)
        self._deadline = DEADLINES[0]
        self._gate_state = "open"
        self._episode_index = 0
        self._trial_return = 0.0
        self._subepisodes: list[dict[str, Any]] = []

    @property
    def unwrapped(self):
        return self.base_env.unwrapped

    def _high_level_observation(self) -> np.ndarray:
        deadline = np.zeros(len(DEADLINES), dtype=np.float32)
        deadline[DEADLINES.index(self._deadline)] = 1.0
        episode = np.zeros(EPISODES_PER_TRIAL, dtype=np.float32)
        episode[self._episode_index] = 1.0
        return np.concatenate(
            [
                deadline,
                np.asarray([self.filter.probability_open], dtype=np.float32),
                episode,
            ]
        ).astype(np.float32)

    def _update_belief(self, observation: dict[str, Any]) -> None:
        # Match method1's legal sensor: the route must have been entered before
        # the enlarged local view is allowed to reveal the physical gate bit.
        if int(self.base_env.unwrapped.agent_pos[0]) >= 4:
            self.filter.update(observation)

    def _reset_base(self, *, first: bool, seed: int | None = None) -> None:
        options: dict[str, Any] = {
            "new_trial": first,
            "deadline": self._deadline,
        }
        if first:
            options["gate_state"] = self._gate_state
        observation, _ = self.base_env.reset(
            seed=seed if first else None,
            options=options,
        )
        self._update_belief(observation)

    def _info(self, **extra: Any) -> dict[str, Any]:
        info: dict[str, Any] = {
            "deadline": int(self._deadline),
            "gate_state": self._gate_state,
            "belief_open": float(self.filter.probability_open),
            "subepisode_index": int(self._episode_index + 1),
            "episodes_per_trial": EPISODES_PER_TRIAL,
            "trial_return": float(self._trial_return),
        }
        info.update(extra)
        return info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = dict(options or {})
        requested_deadline = options.get("deadline")
        requested_gate = options.get("gate_state")
        self._deadline = int(
            requested_deadline
            if requested_deadline is not None
            else self._rng.choice(DEADLINES)
        )
        self._gate_state = str(
            requested_gate
            if requested_gate is not None
            else self._rng.choice(("open", "closed"))
        )
        if self._deadline not in DEADLINES:
            raise ValueError(f"deadline must be one of {DEADLINES}")
        if self._gate_state not in ("open", "closed"):
            raise ValueError("gate_state must be open or closed")

        self._episode_index = 0
        self._trial_return = 0.0
        self._subepisodes = []
        self.filter.reset()
        self._reset_base(first=True, seed=seed)
        return self._high_level_observation(), self._info()

    def step(self, action: int):
        route_index = int(action)
        if not self.action_space.contains(route_index):
            raise ValueError(f"invalid route option {route_index}")

        belief_start = float(self.filter.probability_open)
        option_return = 0.0
        primitive_actions: list[int] = []
        positions = [tuple(map(int, self.base_env.unwrapped.agent_pos))]
        final_info: dict[str, Any] = {}
        base_done = False

        for primitive_action in ROUTE_ACTIONS[route_index]:
            observation, reward, terminated, truncated, final_info = self.base_env.step(
                int(primitive_action)
            )
            primitive_actions.append(int(primitive_action))
            option_return += float(reward)
            positions.append(tuple(map(int, self.base_env.unwrapped.agent_pos)))
            self._update_belief(observation)
            base_done = bool(terminated or truncated)
            if base_done:
                break

        # A blocked option (most importantly direct with a closed gate) may
        # exhaust its nominal action sequence before the deadline.  Treat the
        # option as committed and consume the remaining budget with forward
        # no-ops instead of handing control back to the high level mid-episode.
        while not base_done:
            observation, reward, terminated, truncated, final_info = self.base_env.step(2)
            primitive_actions.append(2)
            option_return += float(reward)
            positions.append(tuple(map(int, self.base_env.unwrapped.agent_pos)))
            self._update_belief(observation)
            base_done = bool(terminated or truncated)

        if not base_done:
            raise RuntimeError(
                f"route option {ROUTE_NAMES[route_index]} did not end the base episode"
            )

        self._trial_return += option_return
        summary = {
            "episode": int(self._episode_index + 1),
            "deadline": int(self._deadline),
            "route": ROUTE_NAMES[route_index],
            "success": bool(final_info.get("success", False)),
            "timeout": bool(final_info.get("timeout", False)),
            "collision": bool(final_info.get("collision", False)),
            "return": float(option_return),
            "steps": int(self.base_env.unwrapped.step_count),
            "belief_start": belief_start,
            "belief_end": float(self.filter.probability_open),
            "primitive_actions": primitive_actions,
            "positions": positions,
        }
        self._subepisodes.append(summary)

        trial_done = self._episode_index + 1 >= EPISODES_PER_TRIAL
        if trial_done:
            info = self._info(
                subepisode_done=True,
                subepisodes=list(self._subepisodes),
                selected_route=ROUTE_NAMES[route_index],
                success=summary["success"],
            )
            return self._high_level_observation(), option_return, True, False, info

        self._episode_index += 1
        self._reset_base(first=False)
        info = self._info(
            subepisode_done=True,
            completed_subepisode=summary,
            selected_route=ROUTE_NAMES[route_index],
            success=summary["success"],
        )
        return self._high_level_observation(), option_return, False, False, info

    def render(self):
        return self.base_env.render()

    def close(self) -> None:
        self.base_env.close()


def make_env(args: argparse.Namespace, *, training: bool) -> HierarchicalExactBeliefTrial:
    base_env = CompactViewThreeRouteEnv(
        min_deadline=min(DEADLINES),
        max_deadline=max(DEADLINES),
        step_penalty=args.step_penalty,
        progress_scale=args.progress_scale,
    )
    return HierarchicalExactBeliefTrial(
        base_env,
        seed=args.seed,
        prior_open=args.prior_open,
        training=training,
    )


def evaluate(model: PPO, args: argparse.Namespace) -> list[dict[str, Any]]:
    env = make_env(args, training=False)
    records: list[dict[str, Any]] = []
    print("\ngate   D ep success return steps route   b_start b_end")
    print("-" * 66)
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
                    record = {
                        "gate_state": gate_state,
                        **completed,
                    }
                    records.append(record)
                    print(
                        f"{gate_state:6s} {deadline:2d} {episode:2d} "
                        f"{str(record['success']):>7s} {record['return']:6.2f} "
                        f"{record['steps']:5d} {record['route']:7s} "
                        f"{record['belief_start']:7.2f} {record['belief_end']:5.2f}"
                    )
    finally:
        env.close()
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--progress-scale", type=float, default=0.0)
    parser.add_argument("--prior-open", type=float, default=0.5)
    parser.add_argument("--checkpoint-frequency", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    if args.timesteps <= 0 or args.n_steps <= 0:
        parser.error("timesteps and n-steps must be positive")
    if args.batch_size <= 0 or args.batch_size > args.n_steps:
        parser.error("batch-size must be in [1, n-steps]")
    if not 0.0 <= args.prior_open <= 1.0:
        parser.error("prior-open must be in [0, 1]")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "episodes_per_trial": EPISODES_PER_TRIAL,
                "routes": list(ROUTE_NAMES),
                "high_level_observation": (
                    "deadline one-hot + belief_open + episode-index one-hot"
                ),
            },
            handle,
            indent=2,
        )

    check_env(make_env(args, training=False), warn=True)
    raw_env = make_env(args, training=True)
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
        name_prefix="hierarchical_exact_belief_ppo",
    )
    try:
        print(
            f"\nTraining hierarchical exact-belief PPO for {args.timesteps} "
            "high-level decisions\n"
            "  one trial = 3 route choices; one choice executes one full option\n"
            "  high-level input = deadline, exact belief, and episode index\n"
            "  low-level options = direct / medium / detour expert navigation"
        )
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(str(output_dir / "final_model"))
    finally:
        train_env.close()

    results = evaluate(model, args)
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()

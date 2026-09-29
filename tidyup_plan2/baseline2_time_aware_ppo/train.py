"""Train the V4 D-only PPO baseline with a three-route reverse curriculum.

The policy observes the local semantic map, heading, and remaining time D-t.
Open and closed doors deliberately share one observation channel.  During the
reverse curriculum it learns the three route-defining anchors jointly:

* D=15, gate=open   -> 14-action direct route;
* D=17, gate=closed -> 16-action medium/danger route;
* D=19, gate=closed -> 18-action safe detour.

After all three routes work from the original start, training switches to the
real baseline distribution: D is uniform over {15, 17, 19} and the gate is
random.  Consequently D=15 is expected to fail when the hidden gate is closed.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch
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


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts"
EVAL_DEADLINES = (15, 17, 19)
TASK_NAMES = ("d15_direct", "d17_medium", "d19_detour")
TASKS = {
    "d15_direct": {
        "deadline": 15,
        "gate_state": "open",
        "actions": ThreeRouteHiddenGateEnvV4.DIRECT_SAFE_ACTIONS,
        "prefixes": (12, 12, 12, 12, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
        "branch_entry": (4, 4),
        "branch_action": 2,
    },
    "d17_medium": {
        "deadline": 17,
        "gate_state": "closed",
        "actions": ThreeRouteHiddenGateEnvV4.MEDIUM_PATH_ACTIONS,
        "prefixes": (14, 14, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
        "branch_entry": (3, 3),
        "branch_action": 0,
    },
    "d19_detour": {
        "deadline": 19,
        "gate_state": "closed",
        "actions": ThreeRouteHiddenGateEnvV4.DETOUR_SAFE_ACTIONS,
        "prefixes": (16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
        "branch_entry": (3, 5),
        "branch_action": 1,
    },
}
N_CURRICULUM_STAGES = len(TASKS["d15_direct"]["prefixes"])

CELL_EMPTY = 0
CELL_WALL = 1
CELL_GOAL = 2
CELL_DOOR_HIDDEN = 4
CELL_AGENT = 5
N_CELL_TYPES = 6
VIEW_SIZE = 5
OBJECT_WALL = 2
OBJECT_DOOR = 4
OBJECT_GOAL = 8


class DOnlyObservation(gym.ObservationWrapper):
    """Flatten the local sensor and hide the physical door state."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        sensor_dim = VIEW_SIZE * VIEW_SIZE * N_CELL_TYPES
        low = np.zeros(sensor_dim + 5, dtype=np.float32)
        high = np.ones(sensor_dim + 5, dtype=np.float32)
        high[-1] = np.inf
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        image = np.asarray(observation["image"], dtype=np.uint8)
        objects = image[..., 0]
        cells = np.full((VIEW_SIZE, VIEW_SIZE), CELL_EMPTY, dtype=np.int64)
        cells[objects == OBJECT_WALL] = CELL_WALL
        cells[objects == OBJECT_GOAL] = CELL_GOAL
        cells[objects == OBJECT_DOOR] = CELL_DOOR_HIDDEN
        cells[VIEW_SIZE // 2, VIEW_SIZE // 2] = CELL_AGENT
        local_sensor = np.eye(N_CELL_TYPES, dtype=np.float32)[cells].reshape(-1)

        direction = np.zeros(4, dtype=np.float32)
        direction[int(observation["direction"])] = 1.0
        remaining_time = np.asarray([observation["time"][0]], dtype=np.float32)
        return np.concatenate([local_sensor, direction, remaining_time]).astype(
            np.float32
        )


class ThreeRouteReverseCurriculum(gym.Wrapper):
    """Joint reverse curriculum with success-gated stage transitions."""

    def __init__(
        self,
        env: gym.Env,
        *,
        seed: int,
        success_threshold: float,
        window_size: int,
        min_episodes_per_task: int,
        start_final: bool,
        rehearsal_probability: float,
        rehearsal_prefix: int,
        wrong_route_penalty: float,
        branch_choice_bonus: float,
        branch_drill_probability: float,
    ) -> None:
        super().__init__(env)
        self._rng = np.random.default_rng(seed)
        self.success_threshold = float(success_threshold)
        self.window_size = int(window_size)
        self.min_episodes_per_task = int(min_episodes_per_task)
        self.stage = N_CURRICULUM_STAGES if start_final else 0
        self.rehearsal_probability = float(rehearsal_probability)
        self.rehearsal_prefix = int(rehearsal_prefix)
        self.wrong_route_penalty = float(wrong_route_penalty)
        self.branch_choice_bonus = float(branch_choice_bonus)
        self.branch_drill_probability = float(branch_drill_probability)
        self.current_task = ""
        self.route_target: str | None = None
        self.branch_drill = False
        self.current_prefix = 0
        self.stage_episode_counts = {name: 0 for name in TASK_NAMES}
        self.stage_successes = {
            name: deque(maxlen=self.window_size) for name in TASK_NAMES
        }
        self.stage_history: list[dict] = []

    @property
    def in_final_distribution(self) -> bool:
        return self.stage >= N_CURRICULUM_STAGES

    def _rates(self) -> dict[str, float]:
        return {
            name: (
                float(np.mean(self.stage_successes[name]))
                if self.stage_successes[name]
                else 0.0
            )
            for name in TASK_NAMES
        }

    def _maybe_advance(self) -> None:
        if self.in_final_distribution:
            self.branch_drill = False
            return
        enough_data = all(
            self.stage_episode_counts[name] >= self.min_episodes_per_task
            for name in TASK_NAMES
        )
        rates = self._rates()
        if enough_data and all(
            rates[name] >= self.success_threshold for name in TASK_NAMES
        ):
            self.stage_history.append(
                {
                    "completed_stage": self.stage,
                    "episode_counts": dict(self.stage_episode_counts),
                    "success_rates": rates,
                }
            )
            self.stage += 1
            self.stage_episode_counts = {name: 0 for name in TASK_NAMES}
            self.stage_successes = {
                name: deque(maxlen=self.window_size) for name in TASK_NAMES
            }
            if self.in_final_distribution:
                print("\nCurriculum complete: switched to real random-gate distribution")
            else:
                prefixes = {
                    name: TASKS[name]["prefixes"][self.stage]
                    for name in TASK_NAMES
                }
                print(f"\nCurriculum advanced to stage {self.stage}: {prefixes}")

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = dict(options or {})

        if self.in_final_distribution:
            if self._rng.random() < self.rehearsal_probability:
                self.current_task = TASK_NAMES[
                    int(self._rng.integers(len(TASK_NAMES)))
                ]
                task = TASKS[self.current_task]
                self.route_target = self.current_task
                options.update(
                    {
                        "deadline": task["deadline"],
                        "gate_state": task["gate_state"],
                        "new_trial": True,
                    }
                )
                observation, info = self.env.reset(seed=seed, options=options)
                self.current_prefix = min(
                    self.rehearsal_prefix, len(task["actions"]) - 1
                )
                for action in task["actions"][: self.current_prefix]:
                    observation, _, terminated, truncated, info = self.env.step(
                        action
                    )
                    if terminated or truncated:
                        raise RuntimeError(
                            f"rehearsal prefix terminated: {self.current_task}"
                        )
                base = self.env.unwrapped
                base.episode_return = 0.0
                base.last_progress_reward = 0.0
                info = dict(info)
                info["episode_return"] = 0.0
                info["progress_reward"] = 0.0
                self.current_task = f"rehearsal_{self.current_task}"
                return observation, self._add_info(info)

            deadline = int(self._rng.choice(EVAL_DEADLINES))
            options.update({"deadline": deadline, "new_trial": True})
            options.pop("gate_state", None)
            self.current_task = f"final_d{deadline}"
            self.route_target = None
            self.current_prefix = 0
            observation, info = self.env.reset(seed=seed, options=options)
            return observation, self._add_info(info)

        self.current_task = TASK_NAMES[int(self._rng.integers(len(TASK_NAMES)))]
        task = TASKS[self.current_task]
        self.route_target = self.current_task
        options.update(
            {
                "deadline": task["deadline"],
                "gate_state": task["gate_state"],
                "new_trial": True,
            }
        )
        observation, info = self.env.reset(seed=seed, options=options)
        self.branch_drill = self._rng.random() < self.branch_drill_probability
        self.current_prefix = int(task["prefixes"][self.stage])
        for action in task["actions"][: self.current_prefix]:
            observation, _, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                raise RuntimeError(
                    f"prefix terminated: task={self.current_task}, "
                    f"stage={self.stage}, prefix={self.current_prefix}"
                )

        base = self.env.unwrapped
        base.episode_return = 0.0
        base.last_progress_reward = 0.0
        info = dict(info)
        info["episode_return"] = 0.0
        info["progress_reward"] = 0.0
        return observation, self._add_info(info)

    def step(self, action):
        previous_position = tuple(map(int, self.env.unwrapped.agent_pos))
        previous_direction = int(self.env.unwrapped.agent_dir)
        observation, reward, terminated, truncated, info = self.env.step(action)
        route_violation = False
        if self.branch_drill and self.route_target is not None:
            expected_action = int(
                TASKS[self.route_target]["actions"][self.current_prefix]
            )
            if int(action) == expected_action:
                reward += self.branch_choice_bonus
                terminated = True
                truncated = False
                info = dict(info)
                info.update({"success": True, "timeout": False})
            else:
                reward -= self.wrong_route_penalty
                terminated = True
                truncated = False
                route_violation = True
                info = dict(info)
                info.update({"success": False, "timeout": False})
        elif self.route_target is not None:
            expected_action = int(TASKS[self.route_target]["branch_action"])
            choosing_branch = previous_position == (3, 4) and previous_direction == 0
            if choosing_branch:
                if int(action) == expected_action:
                    reward += self.branch_choice_bonus
                else:
                    reward -= self.wrong_route_penalty
                    terminated = True
                    truncated = False
                    route_violation = True
                    info = dict(info)
                    info.update({"success": False, "timeout": False})
            position = tuple(map(int, self.env.unwrapped.agent_pos))
            branch_entries = {
                tuple(task["branch_entry"]) for task in TASKS.values()
            }
            expected_entry = tuple(TASKS[self.route_target]["branch_entry"])
            if (
                not route_violation
                and position in branch_entries
                and position != expected_entry
            ):
                reward -= self.wrong_route_penalty
                terminated = True
                truncated = False
                route_violation = True
                info = dict(info)
                info.update({"success": False, "timeout": False})
        if (
            (terminated or truncated)
            and not self.in_final_distribution
            and not self.branch_drill
        ):
            success = bool(info.get("success", False))
            self.stage_episode_counts[self.current_task] += 1
            self.stage_successes[self.current_task].append(float(success))
            self._maybe_advance()
        info = self._add_info(info)
        info["route_violation"] = route_violation
        return observation, reward, terminated, truncated, info

    def _add_info(self, info: dict) -> dict:
        result = dict(info)
        result.update(
            {
                "curriculum_stage": self.stage,
                "curriculum_task": self.current_task,
                "curriculum_prefix": self.current_prefix,
                "branch_drill": self.branch_drill,
            }
        )
        return result

    def summary(self) -> dict:
        return {
            "stage": self.stage,
            "in_final_distribution": self.in_final_distribution,
            "current_stage_episode_counts": dict(self.stage_episode_counts),
            "current_stage_success_rates": self._rates(),
            "completed_stages": self.stage_history,
        }


def make_base_env(args: argparse.Namespace) -> gym.Env:
    return ThreeRouteHiddenGateEnvV4(
        min_deadline=min(EVAL_DEADLINES),
        max_deadline=max(EVAL_DEADLINES),
        step_penalty=args.step_penalty,
        progress_scale=args.progress_scale,
    )


def make_training_env(args: argparse.Namespace, output_dir: Path):
    base_env = make_base_env(args)
    curriculum = ThreeRouteReverseCurriculum(
        base_env,
        seed=args.seed,
        success_threshold=args.success_threshold,
        window_size=args.window_size,
        min_episodes_per_task=args.min_episodes_per_task,
        start_final=args.start_final,
        rehearsal_probability=args.rehearsal_probability,
        rehearsal_prefix=args.rehearsal_prefix,
        wrong_route_penalty=args.wrong_route_penalty,
        branch_choice_bonus=args.branch_choice_bonus,
        branch_drill_probability=args.branch_drill_probability,
    )
    env: gym.Env = DOnlyObservation(curriculum)
    env = Monitor(
        env,
        filename=str(output_dir / "train.monitor.csv"),
        info_keywords=(
            "curriculum_stage",
            "curriculum_task",
            "curriculum_prefix",
            "branch_drill",
            "deadline",
            "gate_state",
            "success",
            "timeout",
            "collision",
            "route_violation",
        ),
    )
    return env, curriculum


def make_evaluation_env(args: argparse.Namespace) -> gym.Env:
    return DOnlyObservation(make_base_env(args))


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


def initial_action_probabilities(model: PPO, observation) -> np.ndarray:
    observation_tensor, _ = model.policy.obs_to_tensor(observation)
    distribution = model.policy.get_distribution(observation_tensor)
    return distribution.distribution.probs.detach().cpu().numpy()[0]


def collect_expert_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    env = make_evaluation_env(args)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    labels: dict[bytes, set[int]] = {}
    try:
        for name in TASK_NAMES:
            task = TASKS[name]
            observation, _ = env.reset(
                options={
                    "deadline": task["deadline"],
                    "gate_state": task["gate_state"],
                    "new_trial": True,
                }
            )
            for action in task["actions"]:
                sample = np.asarray(observation, dtype=np.float32)
                observations.append(sample)
                actions.append(int(action))
                labels.setdefault(sample.tobytes(), set()).add(int(action))
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            if not info["success"]:
                raise RuntimeError(f"expert route failed: {name}")
    finally:
        env.close()
    conflicts = [value for value in labels.values() if len(value) > 1]
    if conflicts:
        raise RuntimeError("D-only expert dataset has conflicting action labels")
    return np.stack(observations), np.asarray(actions, dtype=np.int64)


def behavior_clone(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    observation_tensor = torch.as_tensor(observations, device=model.device)
    action_tensor = torch.as_tensor(actions, device=model.device)
    optimizer = torch.optim.Adam(
        model.policy.parameters(), lr=args.bc_learning_rate
    )
    model.policy.train()
    for epoch in range(1, args.bc_epochs + 1):
        distribution = model.policy.get_distribution(observation_tensor)
        loss = -distribution.log_prob(action_tensor).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            probabilities = model.policy.get_distribution(
                observation_tensor
            ).distribution.probs
            accuracy = float(
                (probabilities.argmax(dim=1) == action_tensor).float().mean().item()
            )
            mean_nll = float(loss.item())
        if epoch == 1 or epoch % 200 == 0:
            print(
                f"BC epoch={epoch} loss={mean_nll:.6f} accuracy={accuracy:.3f}"
            )
        if accuracy == 1.0 and mean_nll < args.bc_target_loss:
            break
    return {
        "epochs": epoch,
        "loss": mean_nll,
        "accuracy": accuracy,
        "samples": int(len(actions)),
    }


def evaluate(model: PPO, args: argparse.Namespace) -> list[dict]:
    env = make_evaluation_env(args)
    results: list[dict] = []
    print(
        "\ngate   D   success timeout return steps route   "
        "P(left) P(right) P(forward)"
    )
    print("-" * 82)
    try:
        for gate_state in ("open", "closed"):
            for deadline in EVAL_DEADLINES:
                observation, _ = env.reset(
                    seed=args.seed,
                    options={
                        "new_trial": True,
                        "deadline": deadline,
                        "gate_state": gate_state,
                    },
                )
                probabilities = initial_action_probabilities(model, observation)
                base = env.unwrapped
                positions = [tuple(map(int, base.agent_pos))]
                actions: list[int] = []
                episode_return = 0.0
                terminated = truncated = False
                final_info: dict = {}
                while not (terminated or truncated):
                    action, _ = model.predict(observation, deterministic=True)
                    action = int(action)
                    actions.append(action)
                    observation, reward, terminated, truncated, final_info = env.step(
                        action
                    )
                    episode_return += float(reward)
                    positions.append(tuple(map(int, base.agent_pos)))

                record = {
                    "gate_state": gate_state,
                    "deadline": deadline,
                    "success": bool(final_info["success"]),
                    "timeout": bool(final_info["timeout"]),
                    "collision": bool(final_info["collision"]),
                    "return": episode_return,
                    "steps": int(base.step_count),
                    "route": classify_route(positions),
                    "actions": actions,
                    "positions": positions,
                    "initial_action_probabilities": probabilities.tolist(),
                }
                results.append(record)
                print(
                    f"{gate_state:6s} {deadline:2d} "
                    f"{str(record['success']):>7s} "
                    f"{str(record['timeout']):>7s} "
                    f"{episode_return:6.2f} {record['steps']:5d} "
                    f"{record['route']:7s} {probabilities[0]:7.3f} "
                    f"{probabilities[1]:8.3f} {probabilities[2]:10.3f}"
                )
    finally:
        env.close()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--bc-epochs", type=int, default=0)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bc-target-loss", type=float, default=0.01)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--progress-scale", type=float, default=0.0)
    parser.add_argument("--success-threshold", type=float, default=0.75)
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--min-episodes-per-task", type=int, default=40)
    parser.add_argument(
        "--start-final",
        action="store_true",
        help="skip route discovery and start in final-distribution training",
    )
    parser.add_argument(
        "--rehearsal-probability",
        type=float,
        default=0.0,
        help="final-phase probability of rehearsing all three routes near the fork",
    )
    parser.add_argument("--rehearsal-prefix", type=int, default=2)
    parser.add_argument("--wrong-route-penalty", type=float, default=1.0)
    parser.add_argument("--branch-choice-bonus", type=float, default=0.5)
    parser.add_argument("--branch-drill-probability", type=float, default=0.3)
    parser.add_argument(
        "--initial-model",
        default="",
        help="optional PPO model path to resume (with or without .zip)",
    )
    parser.add_argument("--checkpoint-frequency", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    if args.timesteps <= 0 or args.n_steps <= 0:
        parser.error("timesteps and n-steps must be positive")
    if args.batch_size <= 0 or args.batch_size > args.n_steps:
        parser.error("batch-size must be in [1, n-steps]")
    if not 0.0 < args.success_threshold <= 1.0:
        parser.error("success-threshold must be in (0, 1]")
    if args.window_size <= 0 or args.min_episodes_per_task <= 0:
        parser.error("window sizes must be positive")
    if not 0.0 <= args.rehearsal_probability <= 1.0:
        parser.error("rehearsal-probability must be in [0, 1]")
    if args.rehearsal_prefix < 0:
        parser.error("rehearsal-prefix must be non-negative")
    if args.wrong_route_penalty < 0:
        parser.error("wrong-route-penalty must be non-negative")
    if args.branch_choice_bonus < 0:
        parser.error("branch-choice-bonus must be non-negative")
    if not 0.0 <= args.branch_drill_probability <= 1.0:
        parser.error("branch-drill-probability must be in [0, 1]")
    if args.bc_epochs < 0:
        parser.error("bc-epochs must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    check_env(DOnlyObservation(make_base_env(args)), warn=True)
    train_env, curriculum = make_training_env(args, output_dir)
    if args.initial_model:
        model = PPO.load(args.initial_model, env=train_env, device=args.device)
        model.verbose = 1
    else:
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
    if args.bc_epochs > 0 and not args.initial_model:
        observations, actions = collect_expert_dataset(args)
        bc_metrics = behavior_clone(model, observations, actions, args)
        model.save(str(output_dir / "bc_model"))
        bc_evaluation = evaluate(model, args)
        with (output_dir / "bc_evaluation.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {"metrics": bc_metrics, "evaluation": bc_evaluation},
                handle,
                indent=2,
            )
    callback = CheckpointCallback(
        save_freq=max(args.checkpoint_frequency, 1),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="time_aware_ppo",
    )
    try:
        print(
            f"\nTraining three-route D-only PPO for {args.timesteps} steps\n"
            "  D=15/open -> direct; D=17/closed -> medium; "
            "D=19/closed -> detour\n"
            "  gate state hidden; curriculum stages advance by per-route success\n"
            "  final phase uses uniform D and a random hidden gate"
        )
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(str(output_dir / "final_model"))
    finally:
        train_env.close()

    with (output_dir / "curriculum_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(curriculum.summary(), handle, indent=2)
    results = evaluate(model, args)
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()

"""Train a three-episode exact-gate-belief PPO controller.

The physical gate is hidden until it enters the agent's local observation.
Within a trial, the gate and deadline stay fixed for all three episodes.  A
deterministic Bayesian filter maintains P(gate=open), while PPO learns control
from the local state, D-t, the exact belief, and the number of episodes left.

Training follows baseline2's reverse curriculum over the direct, medium, and
detour routes before switching to the true random-gate task distribution.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
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


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts_route_exploration_two_pulse"
DEADLINES = (15, 17, 19)
EPISODES_PER_TRIAL = 3
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
# First stage where all three reverse-curriculum prefixes place the agent at
# (or before) the common route fork.  Exploration must be restored here rather
# than after the whole curriculum, otherwise the policy cannot pass this stage.
ROUTE_CHOICE_STAGE = next(
    stage
    for stage in range(N_CURRICULUM_STAGES)
    if all(TASKS[name]["prefixes"][stage] <= 2 for name in TASK_NAMES)
)

CELL_EMPTY = 0
CELL_WALL = 1
CELL_GOAL = 2
CELL_DOOR_HIDDEN = 4
CELL_AGENT = 5
N_CELL_TYPES = 6
# V4's medium corridor is three cells away from the gate while its detour
# corridor is two cells away.  A centered 7x7 local view makes the gate legally
# observable from both alternative routes without changing route lengths,
# rewards, or deadlines.
VIEW_SIZE = 7
OBJECT_WALL = 2
OBJECT_DOOR = 4
OBJECT_GOAL = 8
DOOR_OPEN = 0


class CompactViewThreeRouteEnv(ThreeRouteHiddenGateEnvV4):
    """Method-1-only V4 variant whose three routes share gate visibility."""

    LOCAL_VIEW_SIZE = VIEW_SIZE


class ExactGateBelief:
    """Exact posterior for a deterministic locally observed binary gate."""

    def __init__(self, prior_open: float = 0.5) -> None:
        if not 0.0 <= prior_open <= 1.0:
            raise ValueError("prior_open must be in [0, 1]")
        self.prior_open = float(prior_open)
        self.probability_open = float(prior_open)

    def reset(self) -> float:
        self.probability_open = self.prior_open
        return self.probability_open

    def update(self, observation: dict[str, Any]) -> float:
        image = np.asarray(observation["image"], dtype=np.uint8)
        door_mask = image[..., 0] == OBJECT_DOOR
        if not np.any(door_mask):
            return self.probability_open
        states = image[..., 2][door_mask]
        observed_open = states == DOOR_OPEN
        if np.all(observed_open):
            self.probability_open = 1.0
        elif np.all(~observed_open):
            self.probability_open = 0.0
        else:
            raise RuntimeError("inconsistent door states in one local observation")
        return self.probability_open


class ExactBeliefTrialCurriculum(gym.Env[np.ndarray, int]):
    """Join three V4 episodes and expose an exact gate belief to PPO."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        base_env: gym.Env,
        *,
        seed: int,
        prior_open: float,
        success_threshold: float,
        window_size: int,
        min_episodes_per_task: int,
        start_final: bool,
        rehearsal_probability: float,
        rehearsal_prefix: int,
        wrong_route_penalty: float,
        branch_choice_bonus: float,
        branch_drill_probability: float,
        training: bool,
    ) -> None:
        super().__init__()
        self.base_env = base_env
        self.action_space = base_env.action_space
        self._rng = np.random.default_rng(seed)
        self.filter = ExactGateBelief(prior_open)
        self.training = bool(training)
        self.success_threshold = float(success_threshold)
        self.window_size = int(window_size)
        self.min_episodes_per_task = int(min_episodes_per_task)
        self.stage = N_CURRICULUM_STAGES if start_final or not training else 0
        self.rehearsal_probability = float(rehearsal_probability)
        self.rehearsal_prefix = int(rehearsal_prefix)
        self.wrong_route_penalty = float(wrong_route_penalty)
        self.branch_choice_bonus = float(branch_choice_bonus)
        self.branch_drill_probability = float(branch_drill_probability)

        sensor_dim = VIEW_SIZE * VIEW_SIZE * N_CELL_TYPES
        # semantic map + direction + D-t + belief + normalized episodes left
        low = np.zeros(sensor_dim + 7, dtype=np.float32)
        high = np.ones(sensor_dim + 7, dtype=np.float32)
        high[sensor_dim + 4] = np.inf
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.stage_episode_counts = {name: 0 for name in TASK_NAMES}
        self.stage_successes = {
            name: deque(maxlen=self.window_size) for name in TASK_NAMES
        }
        self.stage_history: list[dict[str, Any]] = []
        self._episode_index = 0
        self._deadline = DEADLINES[0]
        self._gate_state = "open"
        self._task_name = ""
        self._route_target: str | None = None
        self._prefix = 0
        self._branch_drill = False
        self._trial_return = 0.0
        self._subepisode_return = 0.0
        self._subepisodes: list[dict[str, Any]] = []
        self._positions: list[tuple[int, int]] = []

    @property
    def unwrapped(self):
        return self.base_env.unwrapped

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
            return
        enough = all(
            self.stage_episode_counts[name] >= self.min_episodes_per_task
            for name in TASK_NAMES
        )
        rates = self._rates()
        if enough and all(rates[name] >= self.success_threshold for name in TASK_NAMES):
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
                print("\nCurriculum complete: exact-belief random trials enabled")
            else:
                prefixes = {
                    name: TASKS[name]["prefixes"][self.stage] for name in TASK_NAMES
                }
                print(f"\nCurriculum advanced to stage {self.stage}: {prefixes}")

    def _semantic_observation(self, observation: dict[str, Any]) -> np.ndarray:
        image = np.asarray(observation["image"], dtype=np.uint8)
        objects = image[..., 0]
        cells = np.full((VIEW_SIZE, VIEW_SIZE), CELL_EMPTY, dtype=np.int64)
        cells[objects == OBJECT_WALL] = CELL_WALL
        cells[objects == OBJECT_GOAL] = CELL_GOAL
        # The local map deliberately hides door state; belief is the only gate bit.
        cells[objects == OBJECT_DOOR] = CELL_DOOR_HIDDEN
        cells[VIEW_SIZE // 2, VIEW_SIZE // 2] = CELL_AGENT
        local_sensor = np.eye(N_CELL_TYPES, dtype=np.float32)[cells].reshape(-1)
        direction = np.zeros(4, dtype=np.float32)
        direction[int(observation["direction"])] = 1.0
        episodes_left = EPISODES_PER_TRIAL - self._episode_index
        auxiliary = np.asarray(
            [
                float(observation["time"][0]),
                self.filter.probability_open,
                episodes_left / EPISODES_PER_TRIAL,
            ],
            dtype=np.float32,
        )
        return np.concatenate([local_sensor, direction, auxiliary]).astype(np.float32)

    def _update_belief(self, observation: dict[str, Any]) -> float:
        """Consume the gate sensor only after a route has been entered.

        With the 7x7 view, the raw MiniGrid image contains the door at the fork
        (x=3).  The policy map always masks door state, and the explicit gate
        sensor becomes readable only at x>=4.  Thus episode 1 must commit to a
        route under the prior, while either alternative route still reveals the
        gate for later episodes.
        """
        agent_x = int(self.base_env.unwrapped.agent_pos[0])
        if agent_x >= 4:
            return self.filter.update(observation)
        return self.filter.probability_open

    def _select_trial(self, options: dict[str, Any]) -> None:
        requested_deadline = options.get("deadline")
        requested_gate = options.get("gate_state")
        if requested_deadline is not None or requested_gate is not None or not self.training:
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
            self._task_name = f"eval_d{self._deadline}_{self._gate_state}"
            self._route_target = None
            self._prefix = 0
            self._branch_drill = False
            return

        if not self.in_final_distribution:
            self._task_name = TASK_NAMES[int(self._rng.integers(len(TASK_NAMES)))]
            task = TASKS[self._task_name]
            self._deadline = int(task["deadline"])
            self._gate_state = str(task["gate_state"])
            self._route_target = self._task_name
            self._prefix = int(task["prefixes"][self.stage])
            self._branch_drill = self._rng.random() < self.branch_drill_probability
            return

        if self._rng.random() < self.rehearsal_probability:
            self._task_name = TASK_NAMES[int(self._rng.integers(len(TASK_NAMES)))]
            task = TASKS[self._task_name]
            self._deadline = int(task["deadline"])
            self._gate_state = str(task["gate_state"])
            self._route_target = self._task_name
            self._prefix = min(self.rehearsal_prefix, len(task["actions"]) - 1)
            self._branch_drill = False
        else:
            self._deadline = int(self._rng.choice(DEADLINES))
            self._gate_state = str(self._rng.choice(("open", "closed")))
            self._task_name = f"final_d{self._deadline}_{self._gate_state}"
            self._route_target = None
            self._prefix = 0
            self._branch_drill = False

    def _base_reset(self, *, first: bool, seed: int | None = None):
        options: dict[str, Any] = {
            "new_trial": first,
            "deadline": self._deadline,
        }
        if first:
            options["gate_state"] = self._gate_state
        observation, info = self.base_env.reset(seed=seed if first else None, options=options)
        self._update_belief(observation)

        if self._route_target is not None and self._prefix > 0:
            actions = TASKS[self._route_target]["actions"]
            for action in actions[: self._prefix]:
                observation, _, terminated, truncated, info = self.base_env.step(action)
                self._update_belief(observation)
                if terminated or truncated:
                    raise RuntimeError(
                        f"curriculum prefix terminated: {self._route_target}, "
                        f"stage={self.stage}, prefix={self._prefix}"
                    )
            base = self.base_env.unwrapped
            base.episode_return = 0.0
            base.last_progress_reward = 0.0
            info = dict(info)
            info["episode_return"] = 0.0
            info["progress_reward"] = 0.0

        self._positions = [tuple(map(int, self.base_env.unwrapped.agent_pos))]
        return observation, info

    def _augment_info(self, info: dict[str, Any], **extra) -> dict[str, Any]:
        result = dict(info)
        result.update(
            {
                "belief_open": float(self.filter.probability_open),
                "subepisode_index": int(self._episode_index + 1),
                "episodes_per_trial": EPISODES_PER_TRIAL,
                "trial_return": float(self._trial_return),
                "curriculum_stage": int(self.stage),
                "curriculum_task": self._task_name,
                "curriculum_prefix": int(self._prefix),
                "branch_drill": bool(self._branch_drill),
            }
        )
        result.update(extra)
        return result

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = dict(options or {})
        self._select_trial(options)
        self._episode_index = 0
        self._trial_return = 0.0
        self._subepisode_return = 0.0
        self._subepisodes = []
        self.filter.reset()
        observation, info = self._base_reset(first=True, seed=seed)
        return self._semantic_observation(observation), self._augment_info(info)

    def step(self, action: int):
        previous_position = tuple(map(int, self.base_env.unwrapped.agent_pos))
        previous_direction = int(self.base_env.unwrapped.agent_dir)
        observation, reward, terminated, truncated, info = self.base_env.step(int(action))
        reward = float(reward)
        route_violation = False

        if self._branch_drill and self._route_target is not None:
            expected = int(TASKS[self._route_target]["actions"][self._prefix])
            if int(action) == expected:
                reward += self.branch_choice_bonus
                info = dict(info)
                info.update({"success": True, "timeout": False})
            else:
                reward -= self.wrong_route_penalty
                route_violation = True
                info = dict(info)
                info.update({"success": False, "timeout": False})
            terminated, truncated = True, False
        elif self._route_target is not None:
            choosing_branch = previous_position == (3, 4) and previous_direction == 0
            if choosing_branch:
                expected = int(TASKS[self._route_target]["branch_action"])
                if int(action) == expected:
                    reward += self.branch_choice_bonus
                else:
                    reward -= self.wrong_route_penalty
                    route_violation = True
                    terminated, truncated = True, False
                    info = dict(info)
                    info.update({"success": False, "timeout": False})
            position = tuple(map(int, self.base_env.unwrapped.agent_pos))
            entries = {tuple(task["branch_entry"]) for task in TASKS.values()}
            expected_entry = tuple(TASKS[self._route_target]["branch_entry"])
            if not route_violation and position in entries and position != expected_entry:
                reward -= self.wrong_route_penalty
                route_violation = True
                terminated, truncated = True, False
                info = dict(info)
                info.update({"success": False, "timeout": False})

        self._update_belief(observation)
        self._positions.append(tuple(map(int, self.base_env.unwrapped.agent_pos)))
        self._trial_return += reward
        self._subepisode_return += reward
        base_done = bool(terminated or truncated)
        if not base_done:
            return (
                self._semantic_observation(observation),
                reward,
                False,
                False,
                self._augment_info(info, route_violation=route_violation),
            )

        success = bool(info.get("success", False))
        summary = {
            "episode": int(self._episode_index + 1),
            "deadline": int(self._deadline),
            "return": float(self._subepisode_return),
            "success": success,
            "timeout": bool(info.get("timeout", False)),
            "collision": bool(info.get("collision", False)),
            "steps": int(self.base_env.unwrapped.step_count),
            "route": classify_route(self._positions),
            "belief_open": float(self.filter.probability_open),
        }
        self._subepisodes.append(summary)

        if self.training and not self.in_final_distribution and not self._branch_drill:
            assert self._route_target is not None
            self.stage_episode_counts[self._route_target] += 1
            self.stage_successes[self._route_target].append(float(success))

        if self._episode_index + 1 >= EPISODES_PER_TRIAL:
            if self.training:
                self._maybe_advance()
            final_info = self._augment_info(
                info,
                subepisode_done=True,
                subepisode_return=float(self._subepisode_return),
                subepisodes=list(self._subepisodes),
                route_violation=route_violation,
            )
            return self._semantic_observation(observation), reward, True, False, final_info

        self._episode_index += 1
        self._subepisode_return = 0.0
        next_observation, reset_info = self._base_reset(first=False)
        next_info = self._augment_info(
            reset_info,
            subepisode_done=True,
            completed_subepisode=summary,
            route_violation=route_violation,
        )
        return self._semantic_observation(next_observation), reward, False, False, next_info

    def summary(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "in_final_distribution": self.in_final_distribution,
            "current_stage_episode_counts": dict(self.stage_episode_counts),
            "current_stage_success_rates": self._rates(),
            "completed_stages": self.stage_history,
        }

    def render(self):
        return self.base_env.render()

    def close(self) -> None:
        self.base_env.close()


def classify_route(positions: list[tuple[int, int]]) -> str:
    if ThreeRouteHiddenGateEnvV4.GATE_POSITION in positions:
        return "direct"
    if any(position in ThreeRouteHiddenGateEnvV4.DANGER_POSITIONS for position in positions):
        return "medium"
    if ThreeRouteHiddenGateEnvV4.GOAL_POSITION in positions:
        return "detour"
    return "other"


def make_base_env(args: argparse.Namespace) -> gym.Env:
    return CompactViewThreeRouteEnv(
        min_deadline=min(DEADLINES),
        max_deadline=max(DEADLINES),
        step_penalty=args.step_penalty,
        progress_scale=args.progress_scale,
    )


def make_trial_env(args: argparse.Namespace, *, training: bool) -> ExactBeliefTrialCurriculum:
    return ExactBeliefTrialCurriculum(
        make_base_env(args),
        seed=args.seed,
        prior_open=args.prior_open,
        success_threshold=args.success_threshold,
        window_size=args.window_size,
        min_episodes_per_task=args.min_episodes_per_task,
        start_final=args.start_final,
        rehearsal_probability=args.rehearsal_probability,
        rehearsal_prefix=args.rehearsal_prefix,
        wrong_route_penalty=args.wrong_route_penalty,
        branch_choice_bonus=args.branch_choice_bonus,
        branch_drill_probability=args.branch_drill_probability,
        training=training,
    )


def make_training_env(args: argparse.Namespace, output_dir: Path):
    curriculum = make_trial_env(args, training=True)
    env: gym.Env = Monitor(
        curriculum,
        filename=str(output_dir / "train.monitor.csv"),
        info_keywords=(
            "curriculum_stage",
            "curriculum_task",
            "curriculum_prefix",
            "deadline",
            "gate_state",
            "belief_open",
            "trial_return",
        ),
    )
    return env, curriculum


def collect_expert_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Collect baseline2's anchor navigation, but leave route choice to RL.

    The sample at the common fork is deliberately omitted by default.  BC
    therefore teaches how to traverse all three corridors without hard-coding
    D=15/17/19 to direct/medium/detour at the one state where that choice is
    made.
    """
    env = make_trial_env(args, training=False)
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
                }
            )
            for episode in range(1, EPISODES_PER_TRIAL + 1):
                completed = None
                for action in task["actions"]:
                    sample = np.asarray(observation, dtype=np.float32)
                    base = env.base_env.unwrapped
                    at_route_fork = (
                        tuple(map(int, base.agent_pos)) == (3, 4)
                        and int(base.agent_dir) == 0
                    )
                    if args.bc_include_branch_decisions or not at_route_fork:
                        observations.append(sample)
                        actions.append(int(action))
                        labels.setdefault(sample.tobytes(), set()).add(int(action))
                    observation, _, terminated, truncated, info = env.step(int(action))
                    if info.get("subepisode_done"):
                        completed = (
                            info["subepisodes"][-1]
                            if terminated or truncated
                            else info["completed_subepisode"]
                        )
                        break
                if completed is None or not completed["success"]:
                    raise RuntimeError(
                        f"expert route failed: task={name}, episode={episode}"
                    )
    finally:
        env.close()
    conflicts = [values for values in labels.values() if len(values) > 1]
    if conflicts:
        raise RuntimeError("expert dataset contains conflicting action labels")
    return np.stack(observations), np.asarray(actions, dtype=np.int64)


class RouteExplorationCallback(BaseCallback):
    """Restore on-policy exploration after the route-anchoring curriculum.

    Strong BC makes the categorical action logits nearly deterministic.  Once
    reverse curriculum reaches the common route fork, soften the final policy
    layer a single time and temporarily raise entropy regularization.  PPO then
    samples alternative branch actions itself, so rewards remain attached to
    the actions that the policy actually selected.
    """

    def __init__(self, curriculum: ExactBeliefTrialCurriculum, args: argparse.Namespace):
        super().__init__(verbose=0)
        self.curriculum = curriculum
        self.args = args
        self.curriculum_pulse_timestep: int | None = None
        self.final_pulse_timestep: int | None = None
        self.active_schedule_start: int | None = None

    def _on_step(self) -> bool:
        return True

    def _apply_pulse(self, factor: float, label: str) -> None:
        with torch.no_grad():
            self.model.policy.action_net.weight.mul_(factor)
            if self.model.policy.action_net.bias is not None:
                self.model.policy.action_net.bias.mul_(factor)
        self.active_schedule_start = int(self.num_timesteps)
        print(
            f"\nRoute exploration pulse ({label}): softened policy logits by "
            f"{factor:.3f}, entropy={self.args.route_exploration_ent_coef:.4f}"
        )

    def _on_rollout_start(self) -> None:
        route_choice_active = (
            self.curriculum.in_final_distribution
            or self.curriculum.stage >= ROUTE_CHOICE_STAGE
        )
        if not route_choice_active:
            self.model.ent_coef = self.args.ent_coef
            return

        if self.curriculum_pulse_timestep is None and not self.curriculum.in_final_distribution:
            self.curriculum_pulse_timestep = int(self.num_timesteps)
            self._apply_pulse(
                float(self.args.policy_logit_soften_factor), "route curriculum"
            )

        if self.curriculum.in_final_distribution and self.final_pulse_timestep is None:
            self.final_pulse_timestep = int(self.num_timesteps)
            self._apply_pulse(
                float(self.args.final_policy_logit_soften_factor),
                "random gate-belief trials",
            )

        assert self.active_schedule_start is not None
        elapsed = max(0, int(self.num_timesteps) - self.active_schedule_start)
        fraction = min(1.0, elapsed / max(1, self.args.route_exploration_steps))
        initial = float(self.args.route_exploration_ent_coef)
        final = float(self.args.ent_coef)
        self.model.ent_coef = initial + fraction * (final - initial)

    def summary(self) -> dict[str, Any]:
        return {
            "curriculum_pulse_timestep": self.curriculum_pulse_timestep,
            "final_distribution_pulse_timestep": self.final_pulse_timestep,
            "activation_curriculum_stage": ROUTE_CHOICE_STAGE,
            "bc_branch_decisions_included": bool(
                self.args.bc_include_branch_decisions
            ),
            "policy_logit_soften_factor": float(
                self.args.policy_logit_soften_factor
            ),
            "final_policy_logit_soften_factor": float(
                self.args.final_policy_logit_soften_factor
            ),
            "initial_entropy_coefficient": float(
                self.args.route_exploration_ent_coef
            ),
            "final_entropy_coefficient": float(self.args.ent_coef),
            "anneal_steps": int(self.args.route_exploration_steps),
        }


def behavior_clone(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
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
            print(f"BC epoch={epoch} loss={mean_nll:.6f} accuracy={accuracy:.3f}")
        if accuracy == 1.0 and mean_nll < args.bc_target_loss:
            break
    return {
        "epochs": epoch,
        "loss": mean_nll,
        "accuracy": accuracy,
        "samples": int(len(actions)),
        "expert_tasks": list(TASK_NAMES),
    }


def evaluate(model: PPO, args: argparse.Namespace) -> list[dict[str, Any]]:
    env = make_trial_env(args, training=False)
    results: list[dict[str, Any]] = []
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
                    belief_start = float(observation[-2])
                    positions = [tuple(map(int, env.unwrapped.agent_pos))]
                    actions: list[int] = []
                    episode_return = 0.0
                    completed = None
                    while completed is None:
                        action, _ = model.predict(observation, deterministic=True)
                        action = int(action)
                        actions.append(action)
                        observation, reward, terminated, truncated, info = env.step(action)
                        episode_return += float(reward)
                        positions.append(tuple(map(int, env.unwrapped.agent_pos)))
                        if info.get("subepisode_done"):
                            completed = info.get("completed_subepisode")
                            if terminated or truncated:
                                completed = info["subepisodes"][-1]
                    record = {
                        "gate_state": gate_state,
                        "deadline": deadline,
                        "episode": episode,
                        "success": bool(completed["success"]),
                        "return": episode_return,
                        "steps": int(completed["steps"]),
                        "route": completed["route"],
                        "belief_start": belief_start,
                        "belief_end": float(completed["belief_open"]),
                        "actions": actions,
                        "positions": positions,
                    }
                    results.append(record)
                    print(
                        f"{gate_state:6s} {deadline:2d} {episode:2d} "
                        f"{str(record['success']):>7s} {episode_return:6.2f} "
                        f"{record['steps']:5d} {record['route']:7s} "
                        f"{belief_start:7.2f} {record['belief_end']:5.2f}"
                    )
    finally:
        env.close()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--bc-epochs", type=int, default=3000)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bc-target-loss", type=float, default=0.01)
    parser.add_argument(
        "--bc-include-branch-decisions",
        action="store_true",
        help="also clone the three route choices at the common fork",
    )
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--progress-scale", type=float, default=0.0)
    parser.add_argument("--prior-open", type=float, default=0.5)
    parser.add_argument("--success-threshold", type=float, default=0.75)
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--min-episodes-per-task", type=int, default=40)
    parser.add_argument("--start-final", action="store_true")
    parser.add_argument("--rehearsal-probability", type=float, default=0.1)
    parser.add_argument("--rehearsal-prefix", type=int, default=2)
    parser.add_argument("--wrong-route-penalty", type=float, default=1.0)
    parser.add_argument("--branch-choice-bonus", type=float, default=0.5)
    parser.add_argument("--branch-drill-probability", type=float, default=0.25)
    parser.add_argument("--route-exploration-ent-coef", type=float, default=0.05)
    parser.add_argument("--route-exploration-steps", type=int, default=120_000)
    parser.add_argument("--policy-logit-soften-factor", type=float, default=0.25)
    parser.add_argument("--final-policy-logit-soften-factor", type=float, default=0.5)
    parser.add_argument("--initial-model", default="")
    parser.add_argument("--checkpoint-frequency", type=int, default=100_000)
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
    if not 0.0 < args.success_threshold <= 1.0:
        parser.error("success-threshold must be in (0, 1]")
    if args.window_size <= 0 or args.min_episodes_per_task <= 0:
        parser.error("window sizes must be positive")
    if not 0.0 <= args.rehearsal_probability <= 1.0:
        parser.error("rehearsal-probability must be in [0, 1]")
    if args.rehearsal_prefix < 0:
        parser.error("rehearsal-prefix must be non-negative")
    if args.wrong_route_penalty < 0 or args.branch_choice_bonus < 0:
        parser.error("route shaping values must be non-negative")
    if not 0.0 <= args.branch_drill_probability <= 1.0:
        parser.error("branch-drill-probability must be in [0, 1]")
    if args.bc_epochs < 0:
        parser.error("bc-epochs must be non-negative")
    if args.route_exploration_ent_coef < 0 or args.ent_coef < 0:
        parser.error("entropy coefficients must be non-negative")
    if args.route_exploration_steps <= 0:
        parser.error("route-exploration-steps must be positive")
    if not 0.0 < args.policy_logit_soften_factor <= 1.0:
        parser.error("policy-logit-soften-factor must be in (0, 1]")
    if not 0.0 < args.final_policy_logit_soften_factor <= 1.0:
        parser.error("final-policy-logit-soften-factor must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({**vars(args), "episodes_per_trial": EPISODES_PER_TRIAL}, handle, indent=2)

    check_env(make_trial_env(args, training=False), warn=True)
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
        with (output_dir / "bc_evaluation.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {"metrics": bc_metrics, "evaluation": bc_evaluation},
                handle,
                indent=2,
            )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_frequency, 1),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="exact_belief_ppo",
    )
    route_exploration_callback = RouteExplorationCallback(curriculum, args)
    callback = CallbackList([checkpoint_callback, route_exploration_callback])
    try:
        print(
            f"\nTraining Method 1 exact-gate-belief PPO for {args.timesteps} steps\n"
            "  one trial = 3 episodes with fixed gate and fixed deadline\n"
            "  policy sees local state, D-t, P(gate=open), and episodes remaining\n"
            "  BC teaches corridor navigation but omits the common-fork choice\n"
            "  route learning follows baseline2's reverse curriculum, then "
            "restores on-policy exploration"
        )
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(str(output_dir / "final_model"))
    finally:
        train_env.close()

    with (output_dir / "curriculum_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **curriculum.summary(),
                "route_exploration": route_exploration_callback.summary(),
            },
            handle,
            indent=2,
        )
    results = evaluate(model, args)
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()

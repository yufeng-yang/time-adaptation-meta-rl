"""DREAM-inspired D=17 curriculum experiment.

This is intentionally kept separate from TEST.py, which remains the fixed
closed-gate sanity check.

Each five-episode trial uses one exploration episode followed by four
exploitation episodes.  The exploration episode's task reward is excluded
from the optimized objective; instead, the policy receives binary-belief
information gain.  Exploitation rewards use a two-level return:

    sum_e beta_trial**e * sum_t gamma_episode**t * reward[e, t]

Training starts with closed-only trials, continues with open-only trials, and
then consolidates on trials whose gate is sampled open/closed equally.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

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


DEADLINE = 17
EPISODES_PER_TRIAL = 5
PRIOR_OPEN = 0.5
DEFAULT_MODEL_PATH = (
    SCRIPT_DIR
    / "artifacts"
    / "test_dream_d17_closed_open_mixed"
    / "brl_dream_d17"
)


def binary_entropy_bits(probability: float) -> float:
    """Entropy of a Bernoulli belief, normalized so H(0.5) == 1."""
    probability = min(max(float(probability), 0.0), 1.0)
    if probability in (0.0, 1.0):
        return 0.0
    return -(
        probability * math.log2(probability)
        + (1.0 - probability) * math.log2(1.0 - probability)
    )


class DreamTrialObjective(gym.Wrapper):
    """One information-gathering episode, then exploitation-only return."""

    def __init__(
        self,
        env: gym.Env,
        *,
        gamma_episode: float,
        beta_trial: float,
        information_reward_scale: float,
    ) -> None:
        super().__init__(env)
        self.gamma_episode = float(gamma_episode)
        self.beta_trial = float(beta_trial)
        self.information_reward_scale = float(information_reward_scale)
        self._episode_index = 0
        self._step_index = 0
        self._previous_belief = PRIOR_OPEN
        self._information_return = 0.0
        self._exploitation_return = 0.0
        self._training_objective = 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._episode_index = 0
        self._step_index = 0
        self._previous_belief = float(info["belief_open"])
        self._information_return = 0.0
        self._exploitation_return = 0.0
        self._training_objective = 0.0
        return observation, self._augment_info(
            info,
            raw_reward=0.0,
            information_gain=0.0,
            exploitation_reward=0.0,
            training_reward=0.0,
        )

    def _augment_info(
        self,
        info: dict[str, Any],
        *,
        raw_reward: float,
        information_gain: float,
        exploitation_reward: float,
        training_reward: float,
    ) -> dict[str, Any]:
        result = dict(info)
        result.update(
            {
                "dream_phase": (
                    "exploration" if self._episode_index == 0 else "exploitation"
                ),
                "raw_reward": float(raw_reward),
                "information_gain": float(information_gain),
                "information_return": float(self._information_return),
                "exploitation_reward": float(exploitation_reward),
                "exploitation_return": float(self._exploitation_return),
                "training_reward": float(training_reward),
                "training_objective": float(self._training_objective),
            }
        )
        return result

    def step(self, action):
        observation, raw_reward, terminated, truncated, info = self.env.step(action)
        raw_reward = float(raw_reward)
        current_belief = float(info["belief_open"])

        information_gain = 0.0
        exploitation_reward = 0.0
        if self._episode_index == 0:
            information_gain = max(
                binary_entropy_bits(self._previous_belief)
                - binary_entropy_bits(current_belief),
                0.0,
            )
            training_reward = self.information_reward_scale * information_gain
            self._information_return += training_reward
        else:
            exploitation_index = self._episode_index - 1
            reward_weight = (
                self.beta_trial**exploitation_index
                * self.gamma_episode**self._step_index
            )
            exploitation_reward = reward_weight * raw_reward
            training_reward = exploitation_reward
            self._exploitation_return += exploitation_reward

        self._training_objective += training_reward
        self._previous_belief = current_belief
        result_info = self._augment_info(
            info,
            raw_reward=raw_reward,
            information_gain=information_gain,
            exploitation_reward=exploitation_reward,
            training_reward=training_reward,
        )

        if info.get("subepisode_done", False) and not (terminated or truncated):
            self._episode_index += 1
            self._step_index = 0
        else:
            self._step_index += 1

        return observation, float(training_reward), terminated, truncated, result_info


def make_env(
    args: argparse.Namespace,
    *,
    fixed_gate_state: str | None,
    monitor_name: str | None,
) -> gym.Env:
    base_env = ThreeRouteHiddenGateEnvV3(
        min_deadline=DEADLINE,
        max_deadline=DEADLINE,
        step_penalty=args.step_penalty,
        progress_scale=args.progress_scale,
        danger_penalty=args.danger_penalty,
        defer_danger_penalty=True,
    )
    env: gym.Env = ExactBeliefTrialEnv(
        base_env,
        deadlines=(DEADLINE,),
        episodes_per_trial=EPISODES_PER_TRIAL,
        prior_open=PRIOR_OPEN,
        fixed_gate_state=fixed_gate_state,
    )
    env = DreamTrialObjective(
        env,
        gamma_episode=args.gamma_episode,
        beta_trial=args.beta_trial,
        information_reward_scale=args.information_reward_scale,
    )
    if monitor_name is not None:
        artifact_dir = Path(args.model_path).parent
        artifact_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(
            env,
            filename=str(artifact_dir / monitor_name),
            info_keywords=(
                "gate_state",
                "belief_open",
                "trial_return",
                "information_return",
                "exploitation_return",
                "training_objective",
            ),
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
        # DreamTrialObjective has already encoded gamma_episode and beta.
        gamma=1.0,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        policy_kwargs={"net_arch": {"pi": [256, 128], "vf": [256, 128]}},
        seed=args.seed,
        verbose=1,
        device=args.device,
    )


def run_trial(model: PPO, args: argparse.Namespace, gate_state: str) -> dict:
    env = make_env(args, fixed_gate_state=gate_state, monitor_name=None)
    try:
        observation, initial_info = env.reset(seed=args.seed)
        terminated = truncated = False
        final_info: dict[str, Any] = {}
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, final_info = env.step(int(action))
    finally:
        env.close()

    subepisodes = final_info["subepisodes"]
    return {
        "gate_state": gate_state,
        "deadline": DEADLINE,
        "initial_belief_open": float(initial_info["belief_open"]),
        "final_belief_open": float(final_info["belief_open"]),
        "raw_trial_return": float(final_info["trial_return"]),
        "information_return": float(final_info["information_return"]),
        "exploitation_return": float(final_info["exploitation_return"]),
        "training_objective": float(final_info["training_objective"]),
        "total_successes": sum(int(item["success"]) for item in subepisodes),
        "exploitation_successes": sum(
            int(item["success"]) for item in subepisodes[1:]
        ),
        "subepisodes": subepisodes,
    }


def evaluate(model: PPO, args: argparse.Namespace) -> dict:
    results = {gate: run_trial(model, args, gate) for gate in ("open", "closed")}
    print("\nDREAM-inspired D=17 deterministic evaluation")
    for gate, result in results.items():
        print(
            f"  {gate}: belief "
            f"{result['initial_belief_open']:.1f}->"
            f"{result['final_belief_open']:.1f}, "
            f"exploration_success={result['subepisodes'][0]['success']}, "
            f"exploitation_successes={result['exploitation_successes']}/4, "
            f"info_return={result['information_return']:.3f}, "
            f"exploitation_return={result['exploitation_return']:.3f}"
        )
        for item in result["subepisodes"]:
            role = "explore" if item["episode_index"] == 0 else "exploit"
            print(
                f"    episode {item['episode_index'] + 1} ({role}): "
                f"success={item['success']} timeout={item['timeout']} "
                f"raw_return={item['return']:.3f} "
                f"belief={item['belief_open']:.1f}"
            )
    return results


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def train(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    artifact_dir = model_path.parent
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        artifact_dir / "config.json",
        {
            **vars(args),
            "deadline": DEADLINE,
            "episodes_per_trial": EPISODES_PER_TRIAL,
            "initial_belief_open": PRIOR_OPEN,
            "initialization": "random",
            "curriculum": [
                {"gate": "closed", "timesteps": args.closed_timesteps},
                {"gate": "open", "timesteps": args.open_timesteps},
                {"gate": "uniform_open_closed", "timesteps": args.mixed_timesteps},
            ],
            "episode_1_objective": "binary belief information gain only",
            "episodes_2_to_5_objective": (
                "sum_e beta_trial**e * sum_t gamma_episode**t * task_reward[e,t]"
            ),
            "ppo_internal_gamma": 1.0,
            "danger_penalty_timing": "episode_end",
        },
    )

    closed_env = make_env(
        args,
        fixed_gate_state="closed",
        monitor_name="closed_stage_monitor.csv",
    )
    check_env(closed_env, warn=True)
    model = build_model(closed_env, args)
    closed_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_frequency, 1),
        save_path=str(artifact_dir / "checkpoints" / "closed_stage"),
        name_prefix="brl_dream_closed",
    )
    try:
        print(
            "Stage 1/3: closed-only DREAM-inspired training\n"
            f"  timesteps={args.closed_timesteps}\n"
            f"  episode 1: information gain x {args.information_reward_scale}\n"
            "  episodes 2-5: learn belief=0 -> danger route"
        )
        model.learn(total_timesteps=args.closed_timesteps, callback=closed_callback)
        model.save(str(artifact_dir / "closed_stage_model"))
        save_json(
            artifact_dir / "closed_stage_evaluation.json",
            evaluate(model, args),
        )
    finally:
        closed_env.close()

    open_env = make_env(
        args,
        fixed_gate_state="open",
        monitor_name="open_stage_monitor.csv",
    )
    model.set_env(open_env)
    open_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_frequency, 1),
        save_path=str(artifact_dir / "checkpoints" / "open_stage"),
        name_prefix="brl_dream_open",
    )
    try:
        print(
            "\nStage 2/3: open-only DREAM-inspired training\n"
            f"  additional timesteps={args.open_timesteps}\n"
            f"  episode 1: information gain x {args.information_reward_scale}\n"
            "  episodes 2-5: learn belief=1 -> direct route"
        )
        model.learn(
            total_timesteps=args.open_timesteps,
            callback=open_callback,
            reset_num_timesteps=False,
        )
        model.save(str(artifact_dir / "open_stage_model"))
        save_json(
            artifact_dir / "open_stage_evaluation.json",
            evaluate(model, args),
        )
    finally:
        open_env.close()

    mixed_env = make_env(
        args,
        fixed_gate_state=None,
        monitor_name="mixed_stage_monitor.csv",
    )
    model.set_env(mixed_env)
    mixed_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_frequency, 1),
        save_path=str(artifact_dir / "checkpoints" / "mixed_stage"),
        name_prefix="brl_dream_mixed",
    )
    try:
        print(
            "\nStage 3/3: uniformly random open/closed consolidation\n"
            f"  additional timesteps={args.mixed_timesteps}\n"
            f"  gamma_episode={args.gamma_episode}\n"
            f"  beta_trial={args.beta_trial}\n"
            f"  gae_lambda={args.gae_lambda}"
        )
        model.learn(
            total_timesteps=args.mixed_timesteps,
            callback=mixed_callback,
            reset_num_timesteps=False,
        )
        model.save(str(model_path))
        print(f"Saved final model to {model_path.with_suffix('.zip')}")
    finally:
        mixed_env.close()

    result = evaluate(model, args)
    save_json(artifact_dir / "evaluation.json", result)


def evaluate_saved(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path)
    if not model_path.with_suffix(".zip").exists():
        raise FileNotFoundError(f"model not found: {model_path.with_suffix('.zip')}")
    model = PPO.load(str(model_path), device=args.device)
    result = evaluate(model, args)
    save_json(model_path.parent / "evaluation.json", result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DREAM-inspired closed/open/mixed curriculum at D=17"
    )
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--closed-timesteps", type=int, default=400_000)
    parser.add_argument("--open-timesteps", type=int, default=400_000)
    parser.add_argument("--mixed-timesteps", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--gamma-episode", type=float, default=0.99)
    parser.add_argument("--beta-trial", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--information-reward-scale", type=float, default=1.0)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--step-penalty", type=float, default=0.01)
    parser.add_argument("--progress-scale", type=float, default=0.02)
    parser.add_argument("--danger-penalty", type=float, default=0.1)
    parser.add_argument("--checkpoint-frequency", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    for name in (
        "closed_timesteps",
        "open_timesteps",
        "mixed_timesteps",
        "n_steps",
        "batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.batch_size > args.n_steps:
        parser.error("--batch-size must not exceed --n-steps")
    if not 0.0 <= args.gamma_episode <= 1.0:
        parser.error("--gamma-episode must be in [0, 1]")
    if not 0.0 <= args.beta_trial <= 1.0:
        parser.error("--beta-trial must be in [0, 1]")
    if args.information_reward_scale < 0.0:
        parser.error("--information-reward-scale must be non-negative")
    if args.danger_penalty < 0.0:
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

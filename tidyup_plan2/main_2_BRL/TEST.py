"""Train from scratch on exactly one task: closed gate with deadline 17.

The environment fixes the hidden gate to closed, but the policy is not told
that fact. Every new trial starts with P(open)=0.5. The exact filter changes
the belief only if the closed door enters the legal 5x5 sensor view.
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
DEFAULT_MODEL_PATH = (
    SCRIPT_DIR / "artifacts" / "test_closed_d17" / "brl_closed_d17"
)


class HierarchicalTrialReturn(gym.Wrapper):
    """Expose the exact two-level trial objective as PPO rewards.

    For raw environment rewards ``r[e, t]``, PPO receives

        beta_trial ** e * gamma_episode ** t * r[e, t].

    PPO must therefore use ``gamma=1``: discounting is already represented in
    the emitted rewards.  Raw returns remain available in the wrapped env's
    info, while ``hierarchical_trial_return`` records the optimized objective.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        gamma_episode: float,
        beta_trial: float,
    ) -> None:
        super().__init__(env)
        self.gamma_episode = float(gamma_episode)
        self.beta_trial = float(beta_trial)
        self._episode_index = 0
        self._step_index = 0
        self._hierarchical_trial_return = 0.0

    def reset(self, **kwargs):
        self._episode_index = 0
        self._step_index = 0
        self._hierarchical_trial_return = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, raw_reward, terminated, truncated, info = self.env.step(action)
        reward_weight = (
            self.beta_trial**self._episode_index
            * self.gamma_episode**self._step_index
        )
        weighted_reward = float(reward_weight * float(raw_reward))
        self._hierarchical_trial_return += weighted_reward

        info = dict(info)
        info.update(
            {
                "raw_reward": float(raw_reward),
                "reward_weight": float(reward_weight),
                "hierarchical_reward": weighted_reward,
                "hierarchical_trial_return": float(
                    self._hierarchical_trial_return
                ),
            }
        )

        if info.get("subepisode_done", False) and not (terminated or truncated):
            self._episode_index += 1
            self._step_index = 0
        else:
            self._step_index += 1

        return observation, weighted_reward, terminated, truncated, info


def make_env(args: argparse.Namespace, *, monitor: bool) -> gym.Env:
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
        episodes_per_trial=args.episodes_per_trial,
        prior_open=PRIOR_OPEN,
        fixed_gate_state=GATE_STATE,
    )
    env = HierarchicalTrialReturn(
        env,
        gamma_episode=args.gamma_episode,
        beta_trial=args.beta_trial,
    )
    if monitor:
        artifact_dir = Path(args.model_path).parent
        artifact_dir.mkdir(parents=True, exist_ok=True)
        env = Monitor(
            env,
            filename=str(artifact_dir / "trial_monitor.csv"),
            info_keywords=(
                "gate_state",
                "belief_open",
                "trial_return",
                "hierarchical_trial_return",
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
        # HierarchicalTrialReturn has already applied gamma_episode and beta.
        gamma=1.0,
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
    hierarchical_trial_return = 0.0
    try:
        observation, initial_info = env.reset(seed=args.seed)
        initial_belief = float(initial_info["belief_open"])
        terminated = truncated = False
        final_info: dict = {}
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, final_info = env.step(
                int(action)
            )
            hierarchical_trial_return += float(reward)
    finally:
        env.close()

    subepisodes = final_info["subepisodes"]
    result = {
        "gate_state": GATE_STATE,
        "deadline": DEADLINE,
        "initial_belief_open": initial_belief,
        "final_belief_open": float(final_info["belief_open"]),
        "trial_return": float(final_info["trial_return"]),
        "hierarchical_trial_return": hierarchical_trial_return,
        "gamma_episode": args.gamma_episode,
        "beta_trial": args.beta_trial,
        "successes": sum(int(item["success"]) for item in subepisodes),
        "episodes_per_trial": args.episodes_per_trial,
        "subepisodes": subepisodes,
    }
    print(
        "\nClosed D=17 deterministic evaluation\n"
        f"  initial belief P(open)={initial_belief:.1f}\n"
        f"  final belief P(open)={result['final_belief_open']:.1f}\n"
        f"  successes={result['successes']}/{args.episodes_per_trial}\n"
        f"  raw trial return={result['trial_return']:.3f}\n"
        f"  hierarchical trial return={hierarchical_trial_return:.3f}\n"
        f"  gamma_episode={args.gamma_episode}, beta_trial={args.beta_trial}"
    )
    for item in subepisodes:
        print(
            f"  episode {item['episode_index'] + 1}: "
            f"success={item['success']} "
            f"timeout={item['timeout']} "
            f"return={item['return']:.3f} "
            f"belief={item['belief_open']:.1f}"
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
            "initialization": "random",
            "policy_gate_access": False,
            "danger_penalty_timing": "episode_end",
            "return_objective": (
                "sum_e beta_trial**e * sum_t gamma_episode**t * reward[e,t]"
            ),
            "ppo_internal_gamma": 1.0,
        },
    )
    env = make_env(args, monitor=True)
    check_env(env, warn=True)
    model = build_model(env, args)
    callback = CheckpointCallback(
        save_freq=max(args.checkpoint_frequency, 1),
        save_path=str(model_path.parent / "checkpoints"),
        name_prefix="brl_closed_d17",
    )
    try:
        print(
            f"Training from scratch: gate={GATE_STATE}, D={DEADLINE}\n"
            f"  timesteps={args.timesteps}\n"
            f"  episodes/trial={args.episodes_per_trial}\n"
            f"  initial belief P(open)={PRIOR_OPEN}\n"
            f"  gamma_episode={args.gamma_episode}\n"
            f"  beta_trial={args.beta_trial}\n"
            "  PPO internal gamma=1.0 (discount already applied to rewards)\n"
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
        description="Train only closed-gate D=17 exact-belief PPO"
    )
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--episodes-per-trial", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument(
        "--gamma-episode",
        "--gamma",
        dest="gamma_episode",
        type=float,
        default=0.99,
        help="within-episode reward discount",
    )
    parser.add_argument(
        "--beta-trial",
        type=float,
        default=1.0,
        help="discount between subepisodes in a trial",
    )
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
    if args.episodes_per_trial <= 0:
        parser.error("--episodes-per-trial must be positive")
    if args.n_steps <= 0:
        parser.error("--n-steps must be positive")
    if args.batch_size <= 0 or args.batch_size > args.n_steps:
        parser.error("--batch-size must be in [1, n-steps]")
    if args.danger_penalty < 0:
        parser.error("--danger-penalty must be non-negative")
    if not 0.0 <= args.gamma_episode <= 1.0:
        parser.error("--gamma-episode must be in [0, 1]")
    if not 0.0 <= args.beta_trial <= 1.0:
        parser.error("--beta-trial must be in [0, 1]")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate_saved(args)


if __name__ == "__main__":
    main()

"""Baseline 0.1: behavior cloning warm-start, then time-only PPO."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

HERE = Path(__file__).resolve().parent
BASELINE0_DIR = HERE.parent
ROOT = BASELINE0_DIR.parent.parent
for path in (ROOT, ROOT / "Minigrid", BASELINE0_DIR, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tidyup_plan2.envs import ThreeRouteHiddenGateEnvV4
from tidyup_plan2.baseline0_time_only_ppo.train import (
    TimeOnlyObservation,
    aggregate,
    evaluate,
    make_base,
    make_training_env,
)

DEFAULT_OUTPUT = HERE / "artifacts"
EXPERT_TASKS = (
    ("d15_direct", 15, "open", ThreeRouteHiddenGateEnvV4.DIRECT_SAFE_ACTIONS),
    ("d17_medium", 17, "closed", ThreeRouteHiddenGateEnvV4.MEDIUM_PATH_ACTIONS),
    ("d19_detour", 19, "closed", ThreeRouteHiddenGateEnvV4.DETOUR_SAFE_ACTIONS),
)


def collect_expert_data(args: argparse.Namespace):
    env = TimeOnlyObservation(make_base(args))
    observations: list[np.ndarray] = []
    actions: list[int] = []
    labels: dict[float, list[int]] = defaultdict(list)
    routes: list[dict] = []
    try:
        for name, deadline, gate, expert_actions in EXPERT_TASKS:
            observation, _ = env.reset(options={
                "new_trial": True, "deadline": deadline, "gate_state": gate,
            })
            final_info = {}
            for action in expert_actions:
                sample = np.asarray(observation, dtype=np.float32)
                observations.append(sample)
                actions.append(int(action))
                remaining = round(float(sample[0] * 19), 6)
                labels[remaining].append(int(action))
                observation, _, terminated, truncated, final_info = env.step(action)
                if terminated or truncated:
                    break
            if not final_info.get("success", False):
                raise RuntimeError(f"expert route failed: {name}")
            routes.append({"name": name, "samples": len(expert_actions)})
    finally:
        env.close()

    conflicts = {
        str(remaining): dict(Counter(values))
        for remaining, values in sorted(labels.items())
        if len(set(values)) > 1
    }
    correct_majority = sum(max(Counter(values).values()) for values in labels.values())
    analysis = {
        "samples": len(actions),
        "unique_observations": len(labels),
        "conflicting_observations": len(conflicts),
        "conflicts": conflicts,
        "maximum_deterministic_training_accuracy": correct_majority / len(actions),
        "routes": routes,
    }
    return np.stack(observations), np.asarray(actions, np.int64), analysis


def behavior_clone(model: PPO, observations: np.ndarray, actions: np.ndarray, args):
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
    action_tensor = torch.as_tensor(actions, dtype=torch.long, device=model.device)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=args.bc_learning_rate)
    history: list[dict] = []
    model.policy.train()
    for epoch in range(1, args.bc_epochs + 1):
        distribution = model.policy.get_distribution(obs_tensor)
        loss = -distribution.log_prob(action_tensor).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % 100 == 0 or epoch == args.bc_epochs:
            with torch.no_grad():
                probabilities = model.policy.get_distribution(obs_tensor).distribution.probs
                accuracy = float((probabilities.argmax(1) == action_tensor).float().mean())
            history.append({
                "epoch": epoch,
                "loss": float(loss.detach()),
                "accuracy": accuracy,
            })
            print(f"BC epoch={epoch} loss={loss.item():.6f} accuracy={accuracy:.3f}")
    return {"epochs": args.bc_epochs, "final": history[-1], "history": history}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--bc-epochs", type=int, default=2_000)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
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
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2))

    env = make_training_env(args, output)
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
        verbose=0,
    )

    observations, actions, dataset_analysis = collect_expert_data(args)
    bc_metrics = behavior_clone(model, observations, actions, args)
    model.save(output / "bc_model")
    bc_results = evaluate(model, args)
    (output / "bc_summary.json").write_text(json.dumps({
        "dataset": dataset_analysis,
        "optimization": bc_metrics,
        "evaluation": aggregate(bc_results),
    }, indent=2))
    (output / "bc_evaluation.json").write_text(json.dumps(bc_results, indent=2))

    callback = CheckpointCallback(
        save_freq=args.checkpoint_frequency,
        save_path=str(output / "checkpoints"),
        name_prefix="time_only_bc_ppo",
    )
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(output / "final_model")
    env.close()

    final_results = evaluate(model, args)
    final_summary = aggregate(final_results)
    (output / "evaluation.json").write_text(json.dumps(final_results, indent=2))
    (output / "summary.json").write_text(json.dumps(final_summary, indent=2))
    print(json.dumps({
        "dataset": dataset_analysis,
        "bc": aggregate(bc_results),
        "ppo": final_summary,
    }, indent=2))


if __name__ == "__main__":
    main()

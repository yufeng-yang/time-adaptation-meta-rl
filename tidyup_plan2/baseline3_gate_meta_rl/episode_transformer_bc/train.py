"""From-scratch episode-level Meta-RL: position-free Transformer + annealed BC.

Each trial fixes gate and D for three episodes. Only completed episodes update
the cross-episode GRU; the controller never receives an unfinished summary.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tidyup_plan2.baseline3_gate_meta_rl import train_decay_bc as base

DEADLINES = (15, 17, 25)


class RemainingTimeObservation(base.TimeConditionedHiddenGateObservation):
    """Same local semantics, but time is (D-t)/fixed_scale without clipping."""

    def __init__(self, env, time_scale=25.):
        super().__init__(env, time_cap=time_scale)
        self.time_scale = float(time_scale)
        high = self.observation_space.high.copy()
        high[-1] = np.inf
        self.observation_space = base.gym.spaces.Box(
            low=self.observation_space.low.copy(), high=high, dtype=np.float32,
        )

    def observation(self, observation):
        encoded = super().observation(observation)
        encoded[-1] = float(np.asarray(observation["time"])[0]) / self.time_scale
        return encoded


def make_env(args):
    env = base.ThreeRouteHiddenGateEnvV4(
        min_deadline=min(DEADLINES), max_deadline=max(DEADLINES),
        step_penalty=args.step_penalty, progress_scale=0.,
    )
    # Missing flag in historical checkpoints means the original reward, never
    # silently reinterpret an old trained model under the new environment.
    if getattr(args, "discount_compensated_terminal_refund", False):
        from tidyup_plan2.baseline3_gate_meta_rl.episode_transformer_deadline_reach.reward import (
            DiscountCompensatedTerminalRefund,
        )
        env = DiscountCompensatedTerminalRefund(env, gamma=args.gamma)
    return RemainingTimeObservation(env, time_scale=args.time_scale)


class TransformerTrialActorCritic(base.TrialActorCritic):
    """No positional embeddings, causal masks, dropout, or extra time features.

Episodes are encoded individually, so there is no padding to mask. The only
time coordinate is the remaining-time feature already present in observations.
Bidirectional attention is safe: this summary is consumed by later episodes.
"""

    def __init__(self, hidden_dim=32, transformer_layers=2, attention_heads=4):
        super().__init__(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=attention_heads, dim_feedforward=128,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.trajectory_transformer = nn.TransformerEncoder(
            layer, num_layers=transformer_layers, norm=nn.LayerNorm(64),
            enable_nested_tensor=False,
        )
        # TransformerEncoder clones its prototype. Independently initialize
        # layers instead of starting every layer with identical parameters.
        for block in self.trajectory_transformer.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.zeros_(block.self_attn.in_proj_bias)
            for linear in (block.self_attn.out_proj, block.linear1, block.linear2):
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)

    def update_hidden(self, hidden, observations, actions, rewards, next_observations, dones):
        one_hot = F.one_hot(actions.long(), base.N_ACTIONS).float()
        extras = torch.stack([rewards.clamp(-2., 2.), dones.float()], dim=-1)
        transitions = torch.cat([observations, one_hot, extras, next_observations], dim=-1)
        encoded = self.transition_encoder(transitions)
        contextual = self.trajectory_transformer(encoded.unsqueeze(0)).squeeze(0)
        weights = torch.softmax(self.event_attention(contextual).squeeze(-1), dim=0)
        attended = (contextual * weights.unsqueeze(-1)).sum(dim=0)
        strongest = contextual.max(dim=0).values
        episode_code = self.episode_encoder(torch.cat([attended, strongest]))
        return self.meta_gru(episode_code, hidden)


def sample_deadline():
    """Uniform target tasks from step zero; no deadline curriculum."""
    return int(np.random.choice(DEADLINES))


def build_experts(env, args):
    return [
        base.collect_expert_trial(
            env, gate, deadline, args.seed + 100_000 + index * 100 + deadline, args,
        )
        for index, gate in enumerate(("open", "closed")) for deadline in DEADLINES
    ]


def reported_return(episode, args):
    if getattr(args, "report_discounted_return", False):
        return episode.discounted_return
    return episode.episode_return


def expert_records(trials, args=None):
    return [
        dict(gate_state=trial.gate_state, deadline=episode.deadline,
             episode=index, steps=episode.steps, return_=reported_return(episode, args),
             undiscounted_return=episode.episode_return,
             success=episode.success, route=episode.route, probed_gate=episode.probed_gate,
             actions=episode.actions.tolist())
        for trial in trials for index, episode in enumerate(trial.episodes, 1)
    ]


@torch.no_grad()
def evaluate(model, args, steps):
    # Evaluation must not alter rollout randomness or leave the model in eval.
    states = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    model.eval()
    env = make_env(args)
    records = []
    try:
        for gate in ("open", "closed"):
            for deadline in DEADLINES:
                for index in range(args.eval_trials_per_gate):
                    trial = base.collect_trial(
                        model, env, gate, args.eval_seed + index, steps, args,
                        torch.device(args.device), deterministic=True, trial_deadline=deadline,
                    )
                    for number, episode in enumerate(trial.episodes, 1):
                        records.append(dict(
                            gate_state=gate, deadline=deadline, trial=index,
                            seed=args.eval_seed + index, episode=number,
                            steps=episode.steps, success=episode.success,
                            return_=reported_return(episode, args), route=episode.route,
                            probed_gate=episode.probed_gate,
                            deadline_sequence=[e.deadline for e in trial.episodes],
                        ))
                        if getattr(args, "report_discounted_return", False):
                            records[-1]["undiscounted_return"] = episode.episode_return
                            records[-1]["discounted_return"] = episode.discounted_return
        aggregate = {gate: {} for gate in ("open", "closed")}
        for gate in aggregate:
            for deadline in DEADLINES:
                per_episode = {}
                for number in (1, 2, 3):
                    subset = [r for r in records if r["gate_state"] == gate
                              and r["deadline"] == deadline and r["episode"] == number]
                    per_episode[str(number)] = dict(
                        mean_return=float(np.mean([r["return_"] for r in subset])),
                        success_rate=float(np.mean([r["success"] for r in subset])),
                        mean_steps=float(np.mean([r["steps"] for r in subset])),
                        route_rates={name: float(np.mean([r["route"] == name for r in subset]))
                                     for name in ("direct", "medium", "detour", "other")},
                    )
                aggregate[gate][str(deadline)] = per_episode
        third = [r for r in records if r["episode"] == 3]
        return dict(total_steps=steps, aggregate=aggregate, episodes=records,
                    third_episode_mean_return=float(np.mean([r["return_"] for r in third])),
                    third_episode_success_rate=float(np.mean([r["success"] for r in third])))
    finally:
        env.close()
        model.train(was_training)
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.random.set_rng_state(states[2])
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint(path, model, optimizer, args, steps, updates, trial_count):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
        config=vars(args), total_steps=steps, updates=updates, trial_count=trial_count,
        python_rng=random.getstate(), numpy_rng=np.random.get_state(),
        torch_rng=torch.random.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    ), path)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def train(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Output is not empty; choose a fresh --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    model = TransformerTrialActorCritic(
        args.hidden_dim, args.transformer_layers, args.attention_heads,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    write_json(output / "config.json", vars(args))
    source_paths = [Path(__file__), Path(base.__file__)]
    if getattr(args, "discount_compensated_terminal_refund", False):
        from tidyup_plan2.baseline3_gate_meta_rl.episode_transformer_deadline_reach import reward
        source_paths.append(Path(reward.__file__))
    source_paths += [Path(path) for path in getattr(args, "training_source_paths", [])]
    write_json(output / "source_manifest.json", {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    })
    env = make_env(args)
    experts = build_experts(env, args)
    write_json(output / "expert_trials.json", expert_records(experts, args))
    print("Fresh initialization; tasks=open/closed x D=(15,17,25); "
          "Transformer=2-way self-attention, no position encoding; h=episode-level", flush=True)
    if getattr(args, "report_discounted_return", False):
        print("Reported return: " + args.return_definition, flush=True)
    print("closed D=25 BC: " + str([
        (e.steps, round(reported_return(e, args), 3), e.route, e.success)
        for trial in experts if trial.gate_state == "closed" and trial.episodes[0].deadline == 25
        for e in trial.episodes
    ]), flush=True)
    fields = ["update", "trial", "total_steps", "gate_state", "trial_return"]
    fields += [f"episode_{n}_{name}" for n in (1, 2, 3)
               for name in ("return", "success", "route", "deadline", "steps", "probe")]
    fields += ["loss", "actor_loss", "critic_loss", "bc_loss", "bc_weight", "entropy", "clip_fraction"]
    if getattr(args, "report_discounted_return", False):
        fields += [f"episode_{n}_undiscounted_return" for n in (1, 2, 3)]
    curve_fields = ["total_steps", "third_episode_mean_return", "third_episode_success_rate", "bc_weight"]
    steps = updates = trial_count = 0
    next_eval, next_checkpoint = args.eval_interval_steps, args.checkpoint_interval_steps
    recent = []
    started = time.monotonic()
    try:
        with (output / "train_trials.csv").open("w", newline="") as logs, \
             (output / "evaluation_curve.csv").open("w", newline="") as curves:
            writer = csv.DictWriter(logs, fieldnames=fields)
            curve_writer = csv.DictWriter(curves, fieldnames=curve_fields)
            writer.writeheader()
            curve_writer.writeheader()

            def record_evaluation():
                result = evaluate(model, args, steps)
                write_json(output / "evaluations" / f"steps_{steps}.json", result)
                write_json(output / "latest_evaluation.json", result)
                curve_writer.writerow({key: result[key] for key in curve_fields[:-1]}
                                      | {"bc_weight": base.bc_coefficient(steps, args)})
                curves.flush()
                print(f"EVAL steps={steps} ep3_return={result['third_episode_mean_return']:.4f} "
                      f"ep3_success={result['third_episode_success_rate']:.3f}", flush=True)
                return result

            record_evaluation()
            save_checkpoint(output / "initial_model.pt", model, optimizer, args, steps, updates, trial_count)
            while steps < args.total_steps:
                gates = ["open", "closed"] * (args.trials_per_update // 2)
                random.shuffle(gates)
                trials = []
                for gate in gates:
                    deadline = sample_deadline()
                    trial = base.collect_trial(
                        model, env, gate, args.seed + trial_count, steps, args, device,
                        trial_deadline=deadline,
                    )
                    trials.append(trial)
                    steps += sum(e.steps for e in trial.episodes)
                    trial_count += 1
                targets, advantages = base.build_meta_targets(trials, args.gamma, args.beta_trial)
                metrics = base.ppo_update(
                    model, optimizer, trials, targets, advantages, args, device, experts, steps,
                )
                updates += 1
                for offset, trial in enumerate(trials):
                    row = dict(update=updates, trial=trial_count-len(trials)+offset,
                               total_steps=steps, gate_state=trial.gate_state,
                               trial_return=sum(reported_return(e, args) for e in trial.episodes), **metrics)
                    for n, e in enumerate(trial.episodes, 1):
                        for key, value in dict(return_=reported_return(e, args), success=e.success,
                                               route=e.route, deadline=e.deadline, steps=e.steps,
                                               probe=e.probed_gate).items():
                            row[f"episode_{n}_{key.rstrip('_')}"] = value
                        if getattr(args, "report_discounted_return", False):
                            row[f"episode_{n}_undiscounted_return"] = e.episode_return
                    writer.writerow(row)
                    recent.append(reported_return(trial.episodes[-1], args))
                logs.flush()
                recent = recent[-50:]
                if updates == 1 or updates % args.log_interval == 0:
                    print(f"update={updates} trials={trial_count} steps={steps} "
                          f"ep3_return50={np.mean(recent):.3f} "
                          f"bc={metrics['bc_loss']:.3f}*{metrics['bc_weight']:.3f} "
                          f"elapsed={time.monotonic()-started:.1f}s", flush=True)
                if steps >= next_eval:
                    record_evaluation()
                    save_checkpoint(output / "latest_model.pt", model, optimizer, args, steps, updates, trial_count)
                    next_eval = (steps // args.eval_interval_steps + 1) * args.eval_interval_steps
                if steps >= next_checkpoint:
                    save_checkpoint(output / "checkpoints" / f"steps_{steps}.pt",
                                    model, optimizer, args, steps, updates, trial_count)
                    next_checkpoint = (steps // args.checkpoint_interval_steps + 1) * args.checkpoint_interval_steps
            result = record_evaluation()
            save_checkpoint(output / "final_model.pt", model, optimizer, args, steps, updates, trial_count)
            write_json(output / "summary.json", dict(
                total_steps=steps, updates=updates, trials=trial_count,
                elapsed_seconds=time.monotonic()-started, evaluation=result["aggregate"],
                third_episode_mean_return=result["third_episode_mean_return"],
                third_episode_success_rate=result["third_episode_success_rate"],
            ))
    finally:
        env.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    options = (
        ("total-steps", 1_000_000, int),
        ("trials-per-update", 4, int), ("hidden-dim", 32, int),
        ("transformer-layers", 2, int), ("attention-heads", 4, int),
        ("learning-rate", 3e-4, float), ("gamma", .99, float), ("beta-trial", 1., float),
        ("ppo-epochs", 6, int), ("clip-range", .2, float), ("entropy-coef", .02, float),
        ("value-coef", .5, float), ("max-grad-norm", 1., float),
        ("bc-initial-weight", 1., float), ("bc-final-weight", 0., float),
        ("bc-decay-steps", 600_000, int), ("step-penalty", .01, float),
        ("time-scale", 25., float), ("eval-interval-steps", 5_000, int),
        ("checkpoint-interval-steps", 25_000, int), ("log-interval", 20, int),
        ("eval-trials-per-gate", 3, int), ("eval-seed", 10_000, int),
        ("seed", 0, int), ("torch-threads", 1, int),
    )
    for name, default, kind in options:
        parser.add_argument("--" + name, default=default, type=kind)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=str(HERE / "runs/seed0_1m"))
    args = parser.parse_args(argv)
    if args.trials_per_update <= 0 or args.trials_per_update % 2:
        parser.error("trials-per-update must be positive and even")
    for name in ("total_steps", "hidden_dim", "transformer_layers", "attention_heads", "ppo_epochs",
                 "eval_interval_steps", "checkpoint_interval_steps", "log_interval",
                 "eval_trials_per_gate", "torch_threads", "time_scale"):
        if getattr(args, name) <= 0:
            parser.error(name + " must be positive")
    if 64 % args.attention_heads:
        parser.error("attention-heads must divide 64")
    if args.bc_decay_steps < 0:
        parser.error("BC schedule length must be nonnegative")
    if args.bc_initial_weight < 0 or args.bc_final_weight < 0:
        parser.error("BC weights must be nonnegative")
    if not 0 <= args.beta_trial <= 1:
        parser.error("beta-trial must be in [0,1]")
    args.task_deadlines = list(DEADLINES)
    args.curriculum_steps = 0
    args.time_normalization = "remaining_steps / fixed_time_scale; no clipping"
    args.episodes_per_trial = 3
    args.hidden_update = "completed_episode_only"
    args.positional_encoding = False
    args.causal_mask = False
    args.dropout = 0.
    args.from_scratch = True
    return args


if __name__ == "__main__":
    train(parse_args())

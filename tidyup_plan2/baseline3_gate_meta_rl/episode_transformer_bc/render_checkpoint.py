"""Render actual Transformer trial policies, without changing actor inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tidyup_plan2.baseline3_gate_meta_rl.episode_transformer_bc import train
from tidyup_plan2.baseline3_gate_meta_rl.continuation_h_comparison.render_step_checkpoint import (
    panel, save_gif, overview_frame,
)


@torch.no_grad()
def record_trial(model, args, gate, deadline, seed):
    env = train.make_env(args)
    hidden = model.initial_hidden(torch.device("cpu"))
    records, groups = [], []
    try:
        for index in range(3):
            options = dict(new_trial=index == 0, deadline=deadline)
            if index == 0:
                options["gate_state"] = gate
            observation, _ = env.reset(seed=seed if index == 0 else None, options=options)
            world = env.unwrapped
            assert world.gate_state == gate and world.deadline == deadline and world.step_count == 0
            initial_hidden = hidden.clone()
            if index:
                assert torch.equal(initial_hidden, previous_hidden)
            obs_items, nxt_items, actions, rewards, dones = [], [], [], [], []
            positions = [list(map(int, world.agent_pos))]
            evidence = [train.base.visible_gate_state(observation)]
            frames = [panel(world, observation, hidden, index+1, gate, deadline, 0., "reset",
                            "New trial: h=0" if index == 0 else "New episode: h inherited")]
            done = False
            while not done:
                logits, _ = model(torch.as_tensor(observation, dtype=torch.float32), hidden)
                action = int(logits.argmax())
                nxt, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                obs_items.append(observation.copy())
                nxt_items.append(nxt.copy())
                actions.append(action)
                rewards.append(float(reward))
                dones.append(float(done))
                observation = nxt
                positions.append(list(map(int, world.agent_pos)))
                evidence.append(train.base.visible_gate_state(observation))
                assert torch.equal(hidden, initial_hidden), "h changed inside an episode"
                route = train.base.classify_route([tuple(p) for p in positions])
                status = ("GOAL reached" if info.get("success", False) else "TIMEOUT") if done else "Running"
                if done:
                    status += f" | route={route}"
                displayed_return = (sum(args.gamma**t*r for t, r in enumerate(rewards))
                                    if getattr(args, "report_discounted_return", False) else sum(rewards))
                frames.append(panel(world, observation, hidden, index+1, gate, deadline,
                                    displayed_return, ("left", "right", "forward")[action], status))
            if index < 2:
                hidden = model.update_hidden(
                    hidden, torch.as_tensor(np.asarray(obs_items), dtype=torch.float32),
                    torch.as_tensor(actions, dtype=torch.long),
                    torch.as_tensor(rewards, dtype=torch.float32),
                    torch.as_tensor(np.asarray(nxt_items), dtype=torch.float32),
                    torch.as_tensor(dones, dtype=torch.float32),
                )
            previous_hidden = hidden.clone()
            records.append(dict(
                episode=index+1, success=bool(info.get("success", False)), steps=world.step_count,
                route=route, return_=displayed_return, positions=positions, actions=actions,
                rewards=rewards, gate_observations=evidence,
                probed_gate=any(e is not None for e in evidence),
                initial_hidden=initial_hidden.tolist(), terminal_hidden=hidden.tolist(),
            ))
            if getattr(args, "report_discounted_return", False):
                records[-1]["undiscounted_return"] = sum(rewards)
            groups.append(frames)
    finally:
        env.close()
    # Independently replay through the unmodified training rollout collector.
    env = train.make_env(args)
    try:
        trial = train.base.collect_trial(model, env, gate, seed, 0, args, torch.device("cpu"),
                                         deterministic=True, trial_deadline=deadline)
    finally:
        env.close()
    for actual, expected in zip(records, trial.episodes):
        assert actual["actions"] == expected.actions.tolist()
        for key in ("success", "steps", "route", "probed_gate"):
            assert actual[key] == getattr(expected, key)
        assert np.isclose(actual["return_"], train.reported_return(expected, args), atol=1e-6, rtol=0)
    return dict(gate_state=gate, deadline=deadline, seed=seed, episodes=records), groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=HERE / "release_0_80w/final_model.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--fps", type=float, default=5.)
    options = parser.parse_args()
    if not 0 < options.fps <= 30:
        parser.error("fps must be in (0,30]")
    torch.set_num_threads(1)
    checkpoint = options.checkpoint.resolve()
    value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**value["config"])
    args.device = "cpu"
    steps = value["total_steps"]
    model = train.TransformerTrialActorCritic(args.hidden_dim, args.transformer_layers, args.attention_heads)
    model.load_state_dict(value["model_state_dict"])
    model.eval()
    output = (options.output_dir or HERE / "runs" / f"renders_{steps}").resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Render output is not empty; choose a fresh --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    evaluation = train.evaluate(model, args, steps)
    reference = checkpoint.parent.parent / "evaluations" / f"steps_{steps}.json"
    matches_saved = None
    if reference.exists():
        saved = json.loads(reference.read_text())
        assert evaluation["episodes"] == saved["episodes"]
        matches_saved = True
    (output / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
    records, groups = [], []
    duration, pause = round(1000/options.fps), max(1, round(options.fps*.8))
    for gate in ("open", "closed"):
        for deadline in train.DEADLINES:
            record, frames = record_trial(model, args, gate, deadline, options.seed)
            for actual in record["episodes"]:
                expected = next(r for r in evaluation["episodes"] if r["gate_state"] == gate
                                and r["deadline"] == deadline and r["episode"] == actual["episode"]
                                and r["seed"] == options.seed)
                for key in ("success", "steps", "route", "probed_gate"):
                    assert actual[key] == expected[key]
                assert np.isclose(actual["return_"], expected["return_"], atol=1e-6, rtol=0)
            records.append(record)
            groups.append(frames)
            flattened = []
            for number, episode in enumerate(frames, 1):
                flattened.extend(episode)
                flattened.extend([episode[-1]]*pause)
                episode[-1].save(output / f"{gate}_D{deadline}_episode{number}.png")
            save_gif(output / f"{gate}_D{deadline}_trial.gif", flattened, duration)
    combined = []
    for episode in range(3):
        longest = max(len(g[episode]) for g in groups)
        for tick in range(longest):
            images = [g[episode][min(tick, len(g[episode])-1)] for g in groups]
            combined.append(overview_frame(
                images, f"Transformer checkpoint {steps:,} steps | Episode {episode+1}/3 | columns: D15 / D17 / D25",
            ))
        combined.extend([combined[-1]]*pause)
        combined[-1].save(output / f"overview_episode{episode+1}.png")
    save_gif(output / "overview_trial.gif", combined, duration)
    metadata = dict(
        checkpoint=str(checkpoint), checkpoint_steps=steps,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        deterministic=True, evaluation_seeds=list(range(args.eval_seed, args.eval_seed+args.eval_trials_per_gate)),
        validation=dict(matches_saved_54_records=matches_saved, matches_original_collector=True,
                        rendered_actions_match_collector=True, hidden_fixed_within_episode=True),
        hidden_update="Completed episodes 1/2 only; inherited across resets; new trials start at zero",
        viewer_only="Full map/true gate labels only in GIF, never added to actor input",
        trials=records,
    )
    (output / "trajectories.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata["validation"]))
    for record in records:
        print(record["gate_state"], record["deadline"],
              [(e["steps"], e["route"], round(e["return_"], 3), e["success"]) for e in record["episodes"]])
    print("Render outputs:", output)


if __name__ == "__main__":
    main()

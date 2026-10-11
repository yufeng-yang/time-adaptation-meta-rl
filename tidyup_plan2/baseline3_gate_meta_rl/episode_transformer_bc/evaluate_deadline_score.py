"""Post-hoc, time-neutral scoring of the unchanged ASAP policy and evaluations.

Do not refund rewards inside the environment: rewards are also inputs to the
policy's episode encoder. Score only AFTER collecting the original trajectories.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASES = [(gate, deadline) for gate in ("open", "closed") for deadline in (15, 17, 25)]
OPTIMAL = (3 - 1 + .8 + 1)/6


def deadline_score(raw_return, steps, step_penalty=.01):
    """Remove step costs; retain fixed goal/danger/collision/timeout amounts."""
    if steps < 0 or step_penalty < 0 or not math.isfinite(raw_return):
        raise ValueError("Invalid scoring inputs")
    return float(raw_return + step_penalty*steps)


def score_rewards(rewards, gamma=.99, step_penalty=.01):
    """Audit the same score from a discounted return, with no double refund."""
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0,1]")
    raw = math.fsum(float(r) for r in rewards)
    discounted = math.fsum(gamma**t*float(r) for t, r in enumerate(rewards))
    gamma_adjustment = raw - discounted
    step_adjustment = step_penalty*len(rewards)
    score = discounted + gamma_adjustment + step_adjustment
    assert math.isclose(score, deadline_score(raw, len(rewards), step_penalty), abs_tol=1e-10)
    return dict(original_raw_return=raw, original_discounted_return=discounted,
                discount_adjustment=gamma_adjustment, step_adjustment=step_adjustment,
                deadline_score=score)


def load_curve(release, step_penalty):
    points = {}
    with (release / "evaluation_curve.csv").open() as handle:
        for row in csv.DictReader(handle):
            step = int(row["total_steps"])
            path = release / "evaluations" / f"steps_{step}.json"
            result = json.loads(path.read_text())
            records = [r for r in result["episodes"] if r["episode"] == 3]
            assert len(records) == 18
            case_scores = {}
            for gate, deadline in CASES:
                subset = [r for r in records if r["gate_state"] == gate and r["deadline"] == deadline]
                assert len(subset) == 3 and {r["seed"] for r in subset} == {10000, 10001, 10002}
                case_scores[f"{gate}_D{deadline}"] = math.fsum(
                    deadline_score(r["return_"], r["steps"], step_penalty) for r in subset)/3
            raw_mean = math.fsum(r["return_"] for r in records)/18
            assert abs(raw_mean-float(row["third_episode_mean_return"])) < 1e-7
            assert abs(raw_mean-result["third_episode_mean_return"]) < 1e-7
            point = dict(total_steps=step, original_return=raw_mean,
                         deadline_score=math.fsum(case_scores.values())/6,
                         success_rate=result["third_episode_success_rate"],
                         bc_weight=float(row["bc_weight"]), case_scores=case_scores)
            if step in points:
                assert points[step] == point
            points[step] = point
    return [points[step] for step in sorted(points)]


def evaluate_model(release, step_penalty):
    """Fresh deterministic rollouts; original gamma/rewards and inherited h."""
    import torch
    from tidyup_plan2.baseline3_gate_meta_rl.episode_transformer_bc import train
    torch.set_num_threads(1)
    checkpoint_path = release / "final_model.pt"
    before = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**checkpoint["config"])
    args.device = "cpu"
    assert args.gamma == .99 and not getattr(args, "discount_compensated_terminal_refund", False)
    assert args.step_penalty == step_penalty
    model = train.TransformerTrialActorCritic(args.hidden_dim, args.transformer_layers, args.attention_heads)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    saved = json.loads((release / "latest_evaluation.json").read_text())
    lookup = {(r["gate_state"], r["deadline"], r["seed"], r["episode"]): r for r in saved["episodes"]}
    records = []
    env = train.make_env(args)
    try:
        with torch.no_grad():
            for gate, deadline in CASES:
                for trial_number in range(3):
                    seed = args.eval_seed + trial_number
                    trial = train.base.collect_trial(
                        model, env, gate, seed, checkpoint["total_steps"], args,
                        torch.device("cpu"), deterministic=True, trial_deadline=deadline)
                    for number, episode in enumerate(trial.episodes, 1):
                        expected = lookup[(gate, deadline, seed, number)]
                        assert (episode.success, episode.route, episode.steps, episode.probed_gate) == (
                            expected["success"], expected["route"], expected["steps"], expected["probed_gate"])
                        assert abs(episode.episode_return-expected["return_"]) < 1e-7
                        audit = score_rewards(episode.rewards, args.gamma, step_penalty)
                        assert abs(audit["deadline_score"]-deadline_score(
                            expected["return_"], expected["steps"], step_penalty)) < 1e-6
                        records.append(dict(gate_state=gate, deadline=deadline, seed=seed,
                                            episode=number, success=episode.success,
                                            steps=episode.steps, route=episode.route,
                                            actions=episode.actions.tolist(),
                                            rewards=episode.rewards.tolist(), **audit))
    finally:
        env.close()
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == before
    return dict(total_steps=checkpoint["total_steps"], checkpoint_sha256=before,
                original_gamma=args.gamma, original_trajectories_match=True, episodes=records)


def generate(release, output):
    config = json.loads((release / "original_training_config.json").read_text())
    points = load_curve(release, config["step_penalty"])
    fresh = evaluate_model(release, config["step_penalty"])
    assert fresh["total_steps"] == points[-1]["total_steps"]
    fresh_mean = math.fsum(r["deadline_score"] for r in fresh["episodes"] if r["episode"] == 3)/18
    assert abs(fresh_mean-points[-1]["deadline_score"]) < 1e-6
    output.mkdir(parents=True, exist_ok=True)
    (output / "final_model_evaluation.json").write_text(json.dumps(fresh, indent=2))
    fields = ["total_steps", "original_return", "deadline_score", "success_rate", "bc_weight"]
    fields += [f"{g}_D{d}" for g, d in CASES]
    with (output / "third_episode_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p in points:
            writer.writerow({k: p[k] for k in fields[:5]} | p["case_scores"])
    metadata = dict(
        policy="Original ASAP Transformer; gamma=.99; original rewards preserved in h",
        evaluation_metric="Post-hoc deadline score = undiscounted return + .01 * steps",
        formula="I_success - .2*I_danger - .05*N_collisions - I_timeout",
        gamma_note="Original curve was already undiscounted; no extra gamma correction added to it",
        smoothing=False, training_seeds=1, evaluation_seeds_per_case=3,
        curve_provenance="Rescored saved evaluations, not reruns of every historical checkpoint",
        final_provenance="Fresh original-reward rollouts match all 54 saved episode records",
        source_release=str(release), checkpoint_sha256=fresh["checkpoint_sha256"],
        source_curve_sha256=hashlib.sha256((release / "evaluation_curve.csv").read_bytes()).hexdigest(),
        optimal_six_case_score=OPTIMAL, final=points[-1],
    )
    (output / "summary.json").write_text(json.dumps(metadata, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14})
    x = [p["total_steps"]/10000 for p in points]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(x, [p["original_return"] for p in points], color=".55", linestyle="--",
            linewidth=1.4, label="Original raw return (includes step costs)")
    ax.plot(x, [p["deadline_score"] for p in points], color="#26755d", linewidth=1.9,
            label="Post-hoc deadline score (no step costs or discount)")
    ax.axhline(OPTIMAL, color=".4", linestyle=":", linewidth=1.1,
               label=f"Optimal deadline score = {OPTIMAL:.4f}")
    ax.axvline(60, color=".65", linestyle="--", linewidth=1.)
    ax.text(60.8, -.75, "BC weight = 0", fontsize=9, color=".35")
    ax.scatter(x[-1], points[-1]["deadline_score"], color="#26755d", s=25, zorder=4)
    ax.annotate(f"Final score: {points[-1]['deadline_score']:.3f}",
                xy=(x[-1], points[-1]["deadline_score"]), xytext=(-8, 17),
                textcoords="offset points", ha="right", fontsize=10)
    values = [p[key] for p in points for key in ("original_return", "deadline_score")]
    ax.set(title="Original ASAP Transformer: episode-3 evaluation",
           xlabel="Total environment steps (x 10,000)",
           ylabel="Episode-3 return / score (six-case mean)", xlim=(0, x[-1]+.7),
           ylim=(min(values)-.15, max(max(values), OPTIMAL)+.18))
    ax.set_xticks(range(0, 81, 10))
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.text(.5, .025, "Source: retained 0-800,058-step evaluations | 3 evaluation seeds per case | "
             "No smoothing | Same ASAP policy, two scoring rules", ha="center", fontsize=9, color=".35")
    fig.tight_layout(rect=(0, .055, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"third_episode_deadline_score.{extension}", dpi=180)
    plt.close(fig)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=HERE / "release_0_80w")
    parser.add_argument("--output-dir", type=Path, default=HERE / "extra_evaluation_0_80w")
    options = parser.parse_args()
    generate(options.release.resolve(), options.output_dir.resolve())

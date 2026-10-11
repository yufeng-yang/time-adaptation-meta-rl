"""Plot the actual episode-3 evaluations of the gamma=1 run, without smoothing."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OPTIMAL = (1 + 1 + 1 - 1 + .8 + 1) / 6


def plot(run):
    config = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    assert config["gamma"] == 1 and config["discount_compensated_terminal_refund"]
    assert (run / "final_model.pt").is_file()
    points = {}
    curve_path = run / "evaluation_curve.csv"
    with curve_path.open() as handle:
        for row in csv.DictReader(handle):
            step = int(row["total_steps"])
            result = json.loads((run / "evaluations" / f"steps_{step}.json").read_text())
            episodes = [e for e in result["episodes"] if e["episode"] == 3]
            cases = {}
            for gate in ("open", "closed"):
                for deadline in (15, 17, 25):
                    selected = [e for e in episodes if e["gate_state"] == gate
                                and e["deadline"] == deadline]
                    assert len(selected) == 3
                    assert {e["seed"] for e in selected} == {10000, 10001, 10002}
                    cases[f"{gate}_D{deadline}"] = float(np.mean([e["return_"] for e in selected]))
            mean = float(np.mean(list(cases.values())))
            assert abs(mean-float(row["third_episode_mean_return"])) < 1e-7
            assert abs(mean-result["third_episode_mean_return"]) < 1e-7
            point = dict(total_steps=step, mean_return=mean, case_returns=cases)
            if step in points:
                assert points[step] == point
            points[step] = point
    data = [points[step] for step in sorted(points)]
    assert data[-1]["total_steps"] == summary["total_steps"]
    optimum_points = [p for p in data if abs(p["mean_return"]-OPTIMAL) < 1e-6]
    first_late_drop = next(p for p in data if p["total_steps"] >= 100000
                           and p["mean_return"] < OPTIMAL-1e-6)
    output = run / "plots"
    output.mkdir(exist_ok=True)
    metadata = dict(
        metric="Episode-3 return; equal mean over six tasks, three evaluation seeds each",
        training_seed=config["seed"], gamma=1, smoothing=False,
        source=str(curve_path.resolve()),
        source_sha256=hashlib.sha256(curve_path.read_bytes()).hexdigest(),
        final_steps=summary["total_steps"], final_mean=data[-1]["mean_return"],
        optimal_mean=OPTIMAL, first_drop_after_100k=first_late_drop["total_steps"],
        last_optimal_evaluation=optimum_points[-1]["total_steps"],
        bc_zero_steps=config["bc_decay_steps"], points=data,
    )
    (output / "curve_data.json").write_text(json.dumps(metadata, indent=2))
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14})
    x = np.array([p["total_steps"]/10000 for p in data])
    y = np.array([p["mean_return"] for p in data])
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(x, y, color="#26755d", linewidth=1.8, label="Gamma=1 + terminal step refund", zorder=3)
    ax.axhline(OPTIMAL, color=".45", linestyle=":", linewidth=1.2,
               label=f"Optimal six-case mean = {OPTIMAL:.4f}")
    bc_x = config["bc_decay_steps"]/10000
    ax.axvline(bc_x, color=".4", linestyle="--", linewidth=1.1, label="BC weight reaches zero (600k)")
    ax.scatter([first_late_drop["total_steps"]/10000], [first_late_drop["mean_return"]],
               s=30, color="#26755d", zorder=4)
    ax.annotate("First late drop: 555k", xy=(first_late_drop["total_steps"]/10000,
                first_late_drop["mean_return"]), xytext=(40, -.65),
                arrowprops=dict(arrowstyle="->", color=".4"), fontsize=10)
    ax.scatter([x[-1]], [y[-1]], s=30, color="#26755d", zorder=4)
    ax.annotate(f"Final: {y[-1]:.3f}", xy=(x[-1], y[-1]), xytext=(-9, 16),
                textcoords="offset points", ha="right", fontsize=10)
    ax.set(title="Deadline-reach Transformer: final episode of each trial",
           xlabel="Total environment steps (x 10,000)",
           ylabel="Episode-3 return (six-case mean)", xlim=(0, x[-1]+.7),
           ylim=(min(y.min(), -1)-.15, OPTIMAL+.2))
    ax.set_xticks(np.arange(0, 81, 10))
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", fontsize=9, frameon=False)
    fig.text(.5, .025,
             "Source: seed0_80w/evaluation_curve.csv | 0-800,027 steps | "
             "3 evaluation seeds per case | No smoothing | One training seed",
             ha="center", fontsize=9, color=".35")
    fig.tight_layout(rect=(0, .055, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"third_episode_return.{extension}", dpi=180)
    plt.close(fig)
    print(json.dumps({k: v for k, v in metadata.items() if k != "points"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "runs/seed0_80w")
    plot(parser.parse_args().run.resolve())

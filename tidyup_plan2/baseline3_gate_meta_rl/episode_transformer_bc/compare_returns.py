"""Compare episode-3 returns with the retained 0--645k historical release.

Historical series are not extended. This is a descriptive comparison, not a
matched-task architecture ablation: deadlines, time scaling and curriculum differ.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tidyup_plan2.baseline3_gate_meta_rl.release_0_64p5w import reproduce

OPTIMAL = (3*.86 - 1.15 + .64 + .82) / 6


def load_series(run):
    reproduce.validate_files()
    historical = reproduce.data()["curves"]
    series = {key: [{"steps": p["steps"], "return_ep3": p["return_ep3"],
                     "source": str(reproduce.HERE / p["source"]), "task_cases": p["task_cases"]}
                    for p in points] for key, points in historical.items()}
    by_step = {}
    with (run / "evaluation_curve.csv").open() as handle:
        for row in csv.DictReader(handle):
            step = int(row["total_steps"])
            source = run / "evaluations" / f"steps_{step}.json"
            result = json.loads(source.read_text())
            records = [r for r in result["episodes"] if r["episode"] == 3]
            cases = sorted({(r["gate_state"], r["deadline"]) for r in records})
            assert cases == [(g, d) for g in ("closed", "open") for d in (15, 17, 25)]
            assert len(records) == 18
            means = []
            for gate, deadline in cases:
                subset = [r for r in records if r["gate_state"] == gate and r["deadline"] == deadline]
                assert len(subset) == 3 and {r["seed"] for r in subset} == {10000, 10001, 10002}
                means.append(sum(r["return_"] for r in subset)/len(subset))
            mean = sum(means)/len(means)
            assert abs(mean-float(row["third_episode_mean_return"])) < 1e-7
            assert abs(mean-result["third_episode_mean_return"]) < 1e-7
            point = dict(steps=step, return_ep3=mean, source=str(source), task_cases=cases)
            if step in by_step:
                assert by_step[step] == point
            by_step[step] = point
    series["transformer"] = [by_step[s] for s in sorted(by_step)]
    return series


def compare(run):
    if not (run / "summary.json").exists():
        raise RuntimeError("Training has not finished; no final comparison generated")
    summary = json.loads((run / "summary.json").read_text())
    assert (run / "final_model.pt").is_file()
    series = load_series(run)
    assert series["transformer"][-1]["steps"] == summary["total_steps"]
    output = run / "comparison"
    output.mkdir(exist_ok=True)
    with (output / "third_episode_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "total_steps", "return_ep3", "source"])
        writer.writeheader()
        for key, points in series.items():
            for p in points:
                writer.writerow(dict(method=key, total_steps=p["steps"], return_ep3=p["return_ep3"], source=p["source"]))
    metadata = dict(
        metric="Undiscounted episode-3 return; equal mean of six tasks, three evaluation seeds per task",
        smoothing=False, extrapolation=False, training_seeds_per_method=1,
        caveat="Different task sets, curriculum and time scaling; not an architecture-only ablation",
        series=series,
    )
    (output / "third_episode_curves.json").write_text(json.dumps(metadata, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14})
    styles = {
        "episode": ("#2864a4", "o", "Original episode-level h (pooling)"),
        "step": ("#c36526", "s", "Original step-level h (GRU)"),
        "transformer": ("#27835c", None, "New episode-level h (Transformer)"),
    }
    fig, ax = plt.subplots(figsize=(12, 6.6))
    for key, points in series.items():
        color, marker, label = styles[key]
        ax.plot([p["steps"]/10000 for p in points], [p["return_ep3"] for p in points],
                color=color, marker=marker, markersize=3.5, linewidth=1.8,
                label=label, zorder=4 if key == "transformer" else 3)
    ax.axhline(OPTIMAL, color=".45", linestyle=":", linewidth=1,
               label=f"Optimal six-case mean: {OPTIMAL:.3f}")
    ax.axvline(60, color=".5", linestyle="--", linewidth=1)
    ax.text(59.5, .68, "BC weight = 0", ha="right", va="bottom", fontsize=9, color=".3")
    ax.axvline(64.5, color=".7", linestyle=":", linewidth=1)
    ax.text(65.2, .68, "Original curves end", va="bottom", fontsize=9, color=".4")
    final = series["transformer"][-1]
    ax.annotate(f"Endpoint: {final['return_ep3']:.3f}",
                xy=(final["steps"]/10000, final["return_ep3"]),
                xytext=(-8, 17), textcoords="offset points", ha="right", va="bottom", fontsize=10,
                arrowprops=dict(arrowstyle="-", color=".4", linewidth=.8))
    ax.set(title="Final episode of each trial: three memory encoders",
           xlabel="Total environment steps (x 10,000)", ylabel="Episode-3 return (six-case mean)",
           xlim=(0, max(p[-1]["steps"] for p in series.values())/10000 + .7), ylim=(-2.05, .78))
    max_w = max(p[-1]["steps"] for p in series.values()) / 10000
    ax.set_xticks(range(0, int(max_w // 10)*10+1, 10))
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", bbox_to_anchor=(1., .10), fontsize=9.5, framealpha=.95)
    fig.text(.08, .025,
             "Source: saved deterministic evaluations; episode 3 only; 3 seeds per case; no smoothing or extrapolation.\n"
             "Original episode: D15/17/19, closed D19 -> D25 after 55w; original step: D15/17/19 throughout.\n"
             "New Transformer: open/closed D15/17/25 from step 0; no curriculum; time=(D-t)/25 (unclipped).\n"
             "Original runs used a long-deadline curriculum and time=min(D-t,20)/20. Single training seed per method.",
             fontsize=8.5)
    fig.tight_layout(rect=(0, .16, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(output / f"third_episode_three_methods.{ext}", dpi=180)
    plt.close(fig)
    new = series["transformer"]
    optimal = [p for p in new if abs(p["return_ep3"]-OPTIMAL) < 1e-6]
    result = dict(
        endpoints={key: {"steps": p[-1]["steps"], "return_ep3": p[-1]["return_ep3"]} for key, p in series.items()},
        first_optimal_transformer=optimal[0]["steps"] if optimal else None,
        last_optimal_transformer=optimal[-1]["steps"] if optimal else None,
        plot=str(output / "third_episode_three_methods.png"),
    )
    (output / "comparison_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "runs/seed0_1m")
    compare(parser.parse_args().run.resolve())

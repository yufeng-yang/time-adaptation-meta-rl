"""Three original trained policies, scored post-hoc without step costs/discount.

The BC marker requested at 500k is a display reference, not a changed training
event: the actual logged annealing schedules still reach zero at 600k.
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
from tidyup_plan2.baseline3_gate_meta_rl.episode_transformer_bc.evaluate_deadline_score import (
    deadline_score, load_curve,
)
from tidyup_plan2.baseline3_gate_meta_rl.release_0_64p5w import reproduce


def scored_evaluation(path, steps):
    result = json.loads(path.read_text())
    records = [r for r in result["episodes"] if r["episode"] == 3]
    cases = sorted({(r["gate_state"], r["deadline"]) for r in records})
    assert len(cases) == 6 and len(records) == 18
    scores = []
    for gate, deadline in cases:
        selected = [r for r in records if r["gate_state"] == gate and r["deadline"] == deadline]
        assert len(selected) == 3
        scores.append(math.fsum(deadline_score(r["return"], r["steps"])
                                for r in selected)/3)
    return dict(total_steps=steps, deadline_score=math.fsum(scores)/6,
                original_return=math.fsum(r["return"] for r in records)/18,
                task_cases=cases, source=str(path.relative_to(ROOT)),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load_series(include_continuations=True):
    reproduce.validate_files()
    series = {}
    for method, points in reproduce.data()["curves"].items():
        series[method] = []
        for point in points:
            path = reproduce.HERE / point["source"]
            scored = scored_evaluation(path, point["steps"])
            assert abs(scored["original_return"]-point["return_ep3"]) < 1e-7
            series[method].append(scored)
        if include_continuations:
            run = HERE.parent / "continuation_h_comparison" / "runs_80w" / method
            progress = json.loads((run / "progress.json").read_text())
            assert progress["status"] == "complete" and progress["total_steps"] >= 800000
            boundary = series[method][-1]
            paths = sorted(run.glob("evaluation_*.json"),
                           key=lambda p: int(p.stem.split("_")[-1]))
            for path in paths:
                steps = int(path.stem.split("_")[-1])
                scored = scored_evaluation(path, steps)
                assert scored["task_cases"] == boundary["task_cases"]
                if steps == boundary["total_steps"]:
                    assert abs(scored["deadline_score"]-boundary["deadline_score"]) < 1e-7
                    assert abs(scored["original_return"]-boundary["original_return"]) < 1e-7
                elif steps > boundary["total_steps"]:
                    series[method].append(scored)
            assert series[method][-1]["total_steps"] == progress["total_steps"]
    release = HERE / "release_0_80w"
    series["transformer"] = [dict(
        total_steps=p["total_steps"], deadline_score=p["deadline_score"],
        original_return=p["original_return"],
        task_cases=[(g, d) for g in ("open", "closed") for d in (15, 17, 25)],
        source=str((release / "evaluations" / f"steps_{p['total_steps']}.json").relative_to(ROOT)),
    ) for p in load_curve(release, .01)]
    return series


def generate(output, bc_marker=500., max_steps=None):
    series = load_series()
    end_steps = max(p[-1]["total_steps"] for p in series.values())
    view_end = end_steps if max_steps is None else max_steps
    if view_end <= 0 or view_end > end_steps:
        raise ValueError("Display cutoff must be within the available training range")
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        metric="Episode-3 deadline score; six tasks equally weighted; three evaluation seeds each",
        formula="Undiscounted original return + .01 * actual episode steps",
        gamma_note="Saved returns are already undiscounted; no additional gamma refund",
        smoothing=False, extrapolation=False, training_seeds_per_method=1,
        x_axis_unit="k steps (1k = 1,000 environment steps)",
        bc_display_marker_steps=bc_marker*1000, actual_bc_zero_steps=600000,
        bc_marker_note="User-requested display reference, not actual annealing endpoint",
        caveat="Different deadlines, curricula and time normalizations; descriptive comparison only",
        display_cutoff_steps=max_steps,
        cutoff_note="Display crop only; source evaluations remain unchanged",
        original_curves_removed=True, series=series,
    )
    (output / "three_methods_deadline_score.json").write_text(json.dumps(metadata, indent=2))
    with (output / "three_methods_deadline_score.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "total_steps", "deadline_score", "original_return"])
        writer.writeheader()
        for method, points in series.items():
            for p in points:
                writer.writerow(dict(method=method, **{key: p[key] for key in writer.fieldnames[1:]}))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14})
    fig, ax = plt.subplots(figsize=(11.8, 6.4))
    styles = dict(episode=("#2864a4", "o", "Episode-level h (attention pooling)"),
                  step=("#c36526", "s", "Step-level h (GRU)"),
                  transformer=("#26755d", None, "Episode-level h (Transformer)"))
    for method, points in series.items():
        color, marker, label = styles[method]
        ax.plot([p["total_steps"]/1000 for p in points],
                [p["deadline_score"] for p in points], color=color, marker=marker,
                markersize=3.2, linewidth=1.8, label=label,
                zorder=4 if method == "transformer" else 3)
    # Do not relabel a training event: the actual BC endpoint remains 600k.
    ax.axvline(bc_marker, color=".5", linestyle="--", linewidth=1.1)
    ax.text(bc_marker+7, -.82, "BC reference*", color=".35", fontsize=10)
    all_values = [p["deadline_score"] for points in series.values() for p in points]
    ax.set(title="Final episode of each trial: time-neutral evaluation",
           xlabel="Total environment steps (k)",
           ylabel="Episode-3 deadline score (six-case mean)",
           xlim=(0, view_end/1000 + (7 if max_steps is None else 0)),
           ylim=(min(all_values)-.15, max(all_values)+.18))
    ticks = list(range(0, int(view_end/1000)+1, 100))
    ax.set_xticks(ticks)
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.text(.075, .042,
             f"Sources: retained + continuation evaluations | Display: 0-{view_end/1000:.0f}k steps | 3 seeds/case | No smoothing.\n"
             "Full 800k runs retained; task sets/curricula differ (not an architecture-only ablation).\n"
             f"* BC display reference at {bc_marker:g}k steps; actual BC weight reaches zero at 600k.",
             fontsize=9, color=".35")
    fig.tight_layout(rect=(0, .135, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"three_methods_deadline_score.{extension}", dpi=180)
    plt.close(fig)
    print(json.dumps({method: dict(points=len(points), last_steps=points[-1]["total_steps"],
                                   last_score=points[-1]["deadline_score"])
                      for method, points in series.items()}, indent=2))
    print(output / "three_methods_deadline_score.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "extra_evaluation_0_80w")
    parser.add_argument("--bc-marker", type=float, default=500., help="Display reference in k steps")
    parser.add_argument("--max-steps", type=int, help="Crop the display at this environment step, preserving full source data")
    options = parser.parse_args()
    generate(options.output_dir.resolve(), options.bc_marker, options.max_steps)

"""Export the user-selected 800,058-step endpoint without deleting raw data."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

import torch

from tidyup_plan2.baseline3_gate_meta_rl.episode_transformer_bc import train, compare_returns

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/seed0_1m"
OUTPUT = HERE / "release_0_80w"
CUTOFF = 800_058


def trim_csv(source, destination):
    with source.open(newline="") as src, destination.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        rows = [row for row in reader if int(row["total_steps"]) <= CUTOFF]
        writer.writerows(rows)
    assert rows and int(rows[-1]["total_steps"]) == CUTOFF
    return len(rows)


def main():
    if OUTPUT.exists() and any(p.name != "README.md" for p in OUTPUT.iterdir()):
        raise RuntimeError("Selected release already contains data; refusing to overwrite")
    OUTPUT.mkdir(exist_ok=True)
    checkpoint = SOURCE / "checkpoints" / f"steps_{CUTOFF}.pt"
    original_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert value["total_steps"] == CUTOFF
    original_manifest = json.loads((SOURCE / "source_manifest.json").read_text())
    for relative, expected in original_manifest.items():
        assert hashlib.sha256((train.ROOT / relative).read_bytes()).hexdigest() == expected
    for name in ("initial_model.pt", "expert_trials.json", "source_manifest.json"):
        shutil.copy2(SOURCE / name, OUTPUT / name)
    shutil.copy2(SOURCE / "config.json", OUTPUT / "original_training_config.json")
    shutil.copy2(checkpoint, OUTPUT / "final_model.pt")
    assert hashlib.sha256((OUTPUT / "final_model.pt").read_bytes()).hexdigest() == original_sha
    config = dict(value["config"])
    config.update(total_steps=800_000, output_dir=str(HERE / "runs/seed0_80w"))
    train.write_json(OUTPUT / "config.json", config)
    counts = {name: trim_csv(SOURCE / name, OUTPUT / name)
              for name in ("train_trials.csv", "evaluation_curve.csv")}
    evaluations = OUTPUT / "evaluations"
    evaluations.mkdir()
    for path in (SOURCE / "evaluations").glob("steps_*.json"):
        if int(path.stem.split("_")[1]) <= CUTOFF:
            shutil.copy2(path, evaluations / path.name)
    final_eval = json.loads((evaluations / f"steps_{CUTOFF}.json").read_text())
    train.write_json(OUTPUT / "latest_evaluation.json", final_eval)
    train.write_json(OUTPUT / "summary.json", dict(
        total_steps=CUTOFF, updates=value["updates"], trials=value["trial_count"],
        endpoint_selection="User-selected checkpoint after inspection of its six-task GIFs",
        evaluation=final_eval["aggregate"],
        third_episode_mean_return=final_eval["third_episode_mean_return"],
        third_episode_success_rate=final_eval["third_episode_success_rate"],
    ))
    lines = []
    for line in (SOURCE.parent / "seed0_1m.console.log").read_text().splitlines(keepends=True):
        match = re.search(r"\bsteps=(\d+)", line)
        if match and int(match.group(1)) > CUTOFF:
            break
        lines.append(line)
    (OUTPUT / "training.console.log").write_text("".join(lines))
    shutil.copytree(SOURCE / f"renders_{CUTOFF}", OUTPUT / "renders")
    compare_returns.compare(OUTPUT)
    selection = dict(
        selected_checkpoint_steps=CUTOFF, nominal_retained_limit=800_000,
        original_completed_steps=json.loads((SOURCE / "summary.json").read_text())["total_steps"],
        source_run=str(SOURCE.relative_to(HERE)), source_checkpoint=str(checkpoint.relative_to(HERE)),
        selected_checkpoint_sha256=original_sha,
        semantics="Retrospective truncation, not a claim that the actual run stopped at 800k",
        raw_after_cutoff="Preserved in the original ignored runs/seed0_1m directory",
        csv_rows=counts, evaluation_files=len(list(evaluations.glob("*.json"))),
    )
    train.write_json(OUTPUT / "selection.json", selection)
    sha256 = {str(p.relative_to(OUTPUT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(OUTPUT.rglob("*")) if p.is_file()}
    train.write_json(OUTPUT / "manifest.json", dict(selected_steps=CUTOFF, sha256=sha256))
    print(json.dumps(selection, indent=2))
    print("Selected release:", OUTPUT)


if __name__ == "__main__":
    main()

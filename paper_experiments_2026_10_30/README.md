# Deadline-aware meta-RL experiment results

This folder is an additive snapshot; existing repository experiments are retained.
Run commands from this folder so its sibling runtime packages resolve correctly.

- `discrete_paper_comparison`: discrete navigation; use the retained 800k full model.
- `continuous_paper_comparison`: continuous navigation; full model in `final_experiment_2`.
- `pick_place_paper_comparison`: nine-option ideal Cartesian pick-place; full model in `final_experiment_3`.
- Go2 gate delivery is still training and is not part of this initial upload. It will be added after training and paired evaluation finish.

Each experiment retains its actual code/configuration, checkpoints, training logs,
paired test tasks and individual results. Read each experiment's README and
result REPORT for reward units, reference labels, interaction budgets and
single-training-seed limitations. In continuous navigation, the primary ODR
result is the **fixed-reference** result; the test-expanded reference is a
separately labeled diagnostic. The Go2 gate task has known fixed station
values and randomized gates, not hidden package values or hardware results.

Before using large demonstration files, run `python restore_large_files.py`.
Chunks preserve the exact original bytes; SHA256 hashes are checked on restore.
Python caches, credentials, and unrelated experiment trees are excluded.
`bundle_manifest.json` records byte sizes and hashes of the uploaded files.

The original training snapshots retain local absolute paths for provenance.
The Go2 locomotion backend additionally needs Genesis, the original robot URDF
and meshes, and its matching controller configuration; these external robot
assets are not silently represented as included. Adapt the documented paths
before replaying on another machine. Saved test records and plots can be read
without that simulator. Preserved archives themselves have not been rewritten.

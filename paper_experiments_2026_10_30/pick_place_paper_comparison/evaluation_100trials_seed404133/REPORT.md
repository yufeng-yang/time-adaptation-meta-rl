# Pick-and-place comparison

The protected Deadline-Aware checkpoint is `final_experiment_3/half_bc/best.pt`. All 146 original archive files remain unchanged. The three other policies were trained from scratch with the archived training protocol. This comparison uses the archived nine high-level options with an ideal fixed-speed Cartesian executor.

## Evaluation

All four policies were evaluated on the same 100 randomly sampled trials, each comprising three episodes. Deadlines were sampled uniformly from the integers 30–150; object positions and hidden values were randomized between trials and held fixed within each trial. Policies used deterministic actions with frozen weights. Seed: 404133. This is a single-training-seed experiment.

Returns use delivered object value divided by four. See `metrics.csv` for raw-value companions. The figure `results_table.png` shows means and 95% bootstrap confidence intervals. ODR labels the final attempted delivery target using the full-information, deadline-feasible optimum in the archived option space; successful optimal delivery is reported separately in `results.json`.

## Findings

Deadline-Aware has the highest mean trial return (2.5275), final-episode return (0.905), and ODR (92%). Context-only obtains 2.415, 0.9025, and 91%, respectively. The paired Deadline-Aware minus context-only differences are +0.1125 [0.0100, 0.2125] for trial return, +0.0025 [-0.0350, 0.0400] for final return, and +0.01 [-0.05, 0.07] for ODR. Thus the latter two metrics do not establish a clear advantage over context-only in this run. The larger improvements are over the two policies without cross-episode context.

Final-episode delivery succeeds in 100% of Deadline-Aware trials and 92% of context-only trials. Only five sampled trials have deadlines below the minimum direct far-target delivery time; this small subset should not support broad statistical claims on its own.

## Reproducibility

The three baselines completed 200 PPO updates each, after 1,000 BC updates. PPO interaction totals were 846,764 (state-only), 840,347 (time-aware), and 797,340 (context-only) low-level steps. The archived full model originally trained for 775,917 steps. Update counts and trial counts match; consumed low-level interaction budgets are not identical. Checkpoints were selected using validation data, never these test tasks.

Six protocol checks passed, and reward/duration consistency was audited for all 1,200 test episodes. Demonstration task/action records match across all three baseline runs. Source snapshots, training logs, task records, individual trial outcomes, and paired bootstrap comparisons are retained. See the parent README for architecture, observation masking, training details, and the distinction between these ideal options and physical manipulation.

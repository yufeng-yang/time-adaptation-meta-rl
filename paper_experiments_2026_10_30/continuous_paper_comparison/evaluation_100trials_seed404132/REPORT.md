# Continuous-navigation comparison: completed

The protected full method uses final_experiment_2/best_model.pt selected at211,346 steps. All125 archive files remain byte-identical. Three missing ablations were trained fresh to500k steps under archived settings; best checkpoints were selected on fixed six-case validation every10k steps.

Testing:100 paired random trials per method; gate probability.5; deadline uniformly sampled over75..230 high-level commands (15..46s). Each trial contains3 episodes. Fixed original geometry and frozen lower controller; no unseen-map claim.

Primary companion results use a route library fixed before testing (`fixed_reference_results.json`, `fixed_reference_metrics.csv`, `fixed_reference_results_table.png`).
- state: trial return -0.131280; final-episode return -0.043760; fixed-reference ODR 87.0%; selected at241658 training steps.
- time: trial return -0.499140; final-episode return -0.166380; fixed-reference ODR 87.0%; selected at330106 training steps.
- context: trial return -0.665700; final-episode return 0.145380; fixed-reference ODR 68.0%; selected at231771 training steps.
- full: trial return -0.189700; final-episode return 0.006680; fixed-reference ODR 87.0%; selected at211346 training steps.

The full method does not dominate all three metrics. Context-only has the highest final-episode return point estimate; state-only has the highest trial-return point estimate. Fixed-reference ODR ties at87% for full/state/time. Marginal bootstrap intervals are broad; these point estimates are not multiseed significance claims.
The full method succeeds in68 of100 final episodes. It has more collision events (mean.85/final episode) than context-only (0) and occasionally times out despite choosing the reference route. Route choice alone does not measure execution success.

ODR limitations: this is empirical route-reference agreement, not a certified global optimum oracle for continuous dynamics. The pre-test archive library contains direct70, upper128, lower155-step successful executions. Test trajectories reveal a123-step upper execution. Original results.json preserves expanded-reference diagnostics (ODR full88%, state/time86%, context67%); these are not substituted for the fixed pre-test primary reference (full/state/time87%, context68%). Both route libraries and full raw trajectories remain available.

Original undiscounted training rewards retain .002/action costs and the archived D145/episode1 danger bonus. Time-neutral return companions are saved separately. No reward rule was changed for the frozen policy.
All1200 test episodes were checked against step budgets and the success/danger/collision reward decomposition. Five protocol tests passed. CIs bootstrap test trials5000 times and do not represent training-seed variability.

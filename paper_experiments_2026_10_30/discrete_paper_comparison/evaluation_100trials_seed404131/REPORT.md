# Discrete navigation: completed matched comparison

Four policies; one training seed; 100 iid paired test trials each, three episodes per trial.
Primary returns retain original step costs. ODR uses time-neutral canonical route decisions.

- state: trial return -0.8508; final return -0.2836; ODR 66.0%; collected steps 800116.
- time: trial return 0.9954; final return 0.3318; ODR 86.0%; collected steps 800002.
- context: trial return -1.7530; final return -1.1858; ODR 0.0%; collected steps 800180.
- full: trial return 0.6490; final return 0.3242; ODR 78.0%; collected steps 800058.

The full model does not outperform the time-aware baseline on the three overall means in this run.
The full model has 100% final route ODR on the 78 feasible sampled trials; the 22 closed-D15 trials are infeasible, but choosing danger causes avoidable extra cost and makes its overall ODR78%.
Time-aware chooses danger in open-D17, which is feasible but suboptimal; it avoids danger in closed-D15. In closed-D25 it probes before safely detouring, costing22 steps rather than the full model's18 steps in the final episode.
The context-only final policy degrades after BC decays. Intermediate validation checkpoints are retained; no early best checkpoint is substituted into the final-budget comparison.
The full model also pays exploration cost on its first closed-D17 episode. Trial return records this cost; the final-episode metric records later performance.

Eight protocol tests passed. All1200 evaluation episodes were replay-audited, including reward sums, terminal status, task persistence, time-neutral event-score identities and action budgets.
Bootstrap CIs quantify sampled task-mixture variability only, not independent training-seed robustness.
The six configurations share a fixed map. These results do not establish unseen-layout generalization.

Important paper alignment: this comparison preserves the selected original gamma=.99, per-step-cost training. Post-hoc deadline scores are saved separately; they are not evidence that the models were trained with compensated gamma=1 rewards.

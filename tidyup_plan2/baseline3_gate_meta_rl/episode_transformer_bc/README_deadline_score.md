# Extra evaluation: time-neutral deadline score of the original ASAP policy

The active model remains `release_0_80w/final_model.pt`, trained with gamma=0.99
and the original step costs. Neither its weights, training reward, transition
rewards supplied to the episode encoder, nor inherited h are changed.

The additional reporting metric is computed AFTER each original rollout:

    score = original_undiscounted_return + 0.01*N
          = I_success - 0.2*I_danger - 0.05*N_collisions - I_timeout.

Here N is the actual number of actions, including the final action. Success
without danger/collisions scores 1 at any feasible arrival time, danger success
scores 0.8, and timeout without danger/collisions scores -1. This is a separate
evaluation score, not the return optimized during ASAP training.

The old curve already logged UNDISCOUNTED episode returns. Gamma=.99 was used
in PPO targets, not in these logged values. Therefore only `0.01*N` needs to be
added to saved records: adding an extra gamma refund would double-count.
When starting from per-step rewards and a discounted return G_gamma, the audit
identity is

    score = G_gamma + sum_t((1-gamma^t)*r_t) + 0.01*N.

All event penalties retain their full fixed amounts; none are erased. These
identities require the original environment's zero progress reward and fixed
goal reward, which are the settings of this release.

## What is re-evaluated

- The entire 0-800,058-step curve is reconstructed from the 161 retained actual
  evaluation points, each with six tasks and three evaluation seeds. These
  historical points are rescored saved records, not fresh checkpoint rollouts.
- The selected 800,058-step model is freshly evaluated on all six tasks, three
  seeds, three episodes. All 54 records match the saved original evaluation.
  Scoring is applied after collection, so the model receives its original
  rewards when computing h.
- Original release files and their integrity manifest remain untouched.

Third-episode final scores: open D15/17/25 each 1; closed D15 -1.2; closed D17
0.8; closed D25 1. The six-case average is 0.6. The optimal average is 0.633333:
the difference is the unnecessary danger penalty in the infeasible closed-D15
case. It is retained in evaluation rather than forgiven as a time cost.

## Generate

From the repository root:

```bash
/home/yufeng.yang/Downloads/enter/envs/sb3sg/bin/python \
  tidyup_plan2/baseline3_gate_meta_rl/episode_transformer_bc/evaluate_deadline_score.py
```

Output: `extra_evaluation_0_80w/`, containing the PNG/PDF curve, scored curve CSV,
fresh model rollouts with original rewards and score corrections, and summary
with source/checkpoint hashes. Both the original raw-return line and additional
deadline-score line are shown without smoothing.

## Three-method comparison

`compare_deadline_scores.py` applies the same score to the old episode-level
attention-pooling and step-level GRU series and the original ASAP Transformer.
Only these three scored series are drawn; raw-return and theoretical-reference
curves are omitted. Outputs are `three_methods_deadline_score.png/.pdf/.csv/.json`
in `extra_evaluation_0_80w/`.

The x-axis uses k steps (1k=1,000 actions); the former x=50 BC display reference
is therefore at x=500 (500k), and 800 corresponds to 800k steps.
It is explicitly marked as a display reference: the actual training BC weights
still reach zero at 600k, and no training data or schedule is changed.
The older curves retain every original point through ~645k and append the
completed continuation evaluations from `../continuation_h_comparison/runs_80w/`.
The repeated resume-boundary evaluation is checked against the retained point
and deduplicated. Actual endpoints: episode-level 800,027, step-level 800,003,
Transformer 800,058. Final episode-3 scores: -0.35, 0.60, 0.60 respectively.
No extrapolation or smoothing is applied; deadlines, curricula and time
normalization still differ between experiments. Original retained release
files remain unchanged. Each appended point records its source file and hash.

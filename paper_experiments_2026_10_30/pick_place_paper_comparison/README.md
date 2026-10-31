# Archived pick-place comparison

The Deadline-Aware policy is the protected`final_experiment_3/half_bc/best.pt`,
selected at PPO update40 /152,964 low-level steps. The entire archive is hashed
before and after evaluation. Its original128-trials-per-deadline replay matches
the stored results exactly. Nothing is written inside the archive.

This is the archived nine-option experiment: inspectA-D and deliverS/A-D, with
an ideal fixed-speed Cartesian executor. It is not the newer eight primitive
actions, a learned physical lower controller, or a simulation of actual grasping.

Three missing baselines are trained fresh. Common settings match the archive:
seed0;768 legal-history demonstration trials;1000 BC updates at Adam1e-3;
200 PPO updates,32 complete three-episode trials per update; Adam3e-4;
gamma1 full-trial returns;4 epochs and2 minibatches;clip.2, value coefficient.5,
entropy.01, gradient norm.5. Auxiliary BC.2->0 over100 PPO updates. Validation
every20 updates on64 paired trials per each of30/65/150, seeds200000+; choose
best mean trial score exactly as the archive did. No test checkpoint selection.
The number of updates and trials is matched; actual consumed low-level steps
are policy dependent and reported separately, rather than claimed identical.

For state-only/time-aware policies, cross-episode context is always zero AND
every current observation is encoded as an independent one-token sequence, so
the within-episode Transformer cannot leak contact history. Context-only keeps
both causal within-episode history and the cross-episode GRU, but its explicit
remaining-time feature(index21) is zeroed before the encoder. All policies keep
the same49 observation slots and nine-action interface; other current-observation
features, including the transient feedback and episode markers, are unchanged.

Testing:100 identical sampled tasks per method, task seed404133, layout/value
seeds700000..700099. Deadlines are uniform integers30..150. Each trial fixes its
sampled ball layout, hidden value permutation and deadline across three episodes.
Parameters are frozen, inference deterministic argmax, context reset per trial.

Primary returns use the actual training reward: delivered value/4 (so trial
return is at most3 and final-episode return at most1). Unscaled delivered-value
companions (max12 and4) are saved too. ODR compares the final attempted delivery
target to the full-information maximum-value feasible target set. The oracle
only reads hidden values after policy execution, for labeling; policies and
demonstrators do not receive them. An attempted optimal target can still time
out after excessive inspection, so delivered-optimal rate is reported separately.

In this ideal option space, direct delivery minimizes the duration for each
target: inspection inserts positive waits/vertical movements, and horizontal
detours cannot beat the direct segment. Thus the oracle enumerates all five
direct deliveries from home with the sampled task's actual integer durations.
This is exact for the archived deterministic options, not an optimality claim
for real manipulation or the eight-action primitive environment.

95% CIs use5000 bootstrap resamples of test trials. Paired full-minus-baseline
differences are also saved. One training seed is not multiseed reliability.
Six protocol tests cover masked inputs/history, archive compatibility, PPO
rollout/replay parity, exact last-step delivery, oracle labels and inspection
duration bounds. All1200 test episodes are audited against rewards and durations.

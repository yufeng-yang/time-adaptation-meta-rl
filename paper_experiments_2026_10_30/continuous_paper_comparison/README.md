# Continuous-navigation comparison using the protected archive

The full method is `../final_experiment_2/best_model.pt`, selected at211,346
steps from its original500k-step training run. Its archived runtime and frozen
low-level actor are used read-only. `archive_before.json` hashes all125 files;
the evaluator checks the entire archive before and after evaluation. No source,
weights, or results are written inside that archive. Its original18-episode
replay was verified against the saved records before starting this comparison.

Missing state-only, time-aware and context-only policies are trained fresh under
`runs/*_seed0_matched500k/`. Time is masked after normalization in both actor
and trajectory encoder when absent. Context is identically zero in state/time
policies. All models retain the archived63-observation/32-context interface,
two64-wide Transformer layers with4 heads and the256x256 controller MLP.
Context updates only at completed episodes; new trials reset it.

Training uses the archive's five training cases (closedD75 excluded),3 episodes
per trial,20 BC warm-up updates, turn-window weighting5 with radius2, Adam3e-4,
gamma1, weighted episode rewards(.2,.5,1),4 PPO epochs, clip.2, entropy.005,
and BC1->.1 over400k steps. The environment-step budget is500k; complete trial
batches may overshoot. The original physical0.2s commands, frozen learned lower
controller, ideal odometry/local radar, fixed map and gate geometry, and original
deadline-dependent first-episode reward are preserved. In particular, first
danger entry gives+.02 only in episode1 atD145; otherwise it costs.2.

Every10k steps, validate on the six archived gate/deadline cases, and select the
checkpoint with the highest episode3 success rate, breaking ties with return,
as in the original archive. The full method keeps its previously selected best
checkpoint; it is not reselected on the new test tasks. New baselines are also
evaluated via validation-selected best checkpoints. Final checkpoints remain
available independently. Initial short starts under`*_seed0_500k/` were stopped
while correcting validation frequency; they are not used in the comparison.

Test protocol: seed404132,100 iid paired trials, gates Bernoulli.5, deadlines
uniform over integer75..230 commands (15..46 seconds). All four methods receive
the same list. Physical resets reproduce the archive's cached initialization
seed100000. This tests interpolation across deadlines on the frozen geometry;
it does not randomize the map, gate positions or sensor interface. Frozen network
parameters, deterministic argmax actions, three episodes, context reset per trial.

Metrics: undiscounted original trial return, final-episode return, and an
empirical validated-route decision rate. Raw returns include.002/action costs;
post-hoc time-neutral returns are saved separately and do not change encoder
rewards. All1200 test episode reward sums are checked against success, danger,
collision events and step costs.

ODR limitation: continuous physical control has no certified shortest-time
oracle. A pre-test library uses successful collision-free routes from the
archived full model on its six fixed validation cases. An expanded library adds
any faster collision-free successful route discovered during the test, solely
to score feasible route choices; no policy or checkpoint is updated. Both ODRs
are saved. Values are time-neutral task returns, retaining danger/failure costs.
These are references over demonstrated candidate routes, not a claim of global
optimality or infeasibility for every possible controller. The output explicitly
labels the metric`validated-route ODR`, unlike a certified oracle ODR.

Confidence intervals bootstrap100 trials5000 times. They measure task-sampling
uncertainty for one training seed, not multiseed training reliability.

Run with the existing`sb3sg`Python and-B, with OPENBLAS/OMP/MKL threads set to1.
`python -B -m continuous_paper_comparison.train --mode MODE --output NEW_DIR`
and, after all runs complete,
`python -B -m continuous_paper_comparison.evaluate --output NEW_OUTPUT_DIR`.

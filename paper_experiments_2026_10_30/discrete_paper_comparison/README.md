# Matched discrete-navigation comparison

Four policies: state-only, time-aware, context-only, and retained Deadline-Aware
Transformer. The three missing input ablations are trained from scratch with
the original retained model's task/reward/optimization settings. Historical SB3
baselines used different deadlines or gate conditions and are not used here.

All models share 155 observation slots and a 32-slot context interface. Absent
time inputs are zeroed in both the actor and the trajectory encoder. Absent
context is identically zero in both rollout and optimization. Unused modules
remain in checkpoint shapes; this does not give a memory-free policy memory.
The context-only policy can infer time indirectly from legitimate histories,
episode length and previous rewards, as intended by this baseline definition.

Training: seed 0, 800k environment-step budget, 4 trials per update, 3 episodes
per trial, gate balanced per batch, deadlines sampled uniformly from 15/17/25.
Gamma=.99, trial discount=1, Adam3e-4, 6 PPO epochs, clip=.2, entropy=.02,
value coefficient=.5 (existing engine additionally multiplies critic MSE by .5),
gradient norm1. Six shared demonstration trials, no pure-BC warm-up, BC1->0
over600k steps. Complete batches may overshoot the budget. The full model is
the existing retained800,058-step checkpoint, not newly trained or test-selected.
Checkpoint selection for every method is the fixed final training endpoint.

Evaluation: exactly100 iid draws from the six gate/deadline configurations,
seed404131. All four policies use the identical task list and deterministic
argmax actions. Context is reset each trial and updated only after completed
episodes. Parameters are frozen; no demonstrations, training or test selection.
This map is fixed. Random trial seeds do not produce100 distinct maps; there
are only six unique configurations. One training seed is not a multiseed study.

Three primary metrics: mean trial raw return (sum of3 episode returns), mean
final-episode raw return, and final-episode task-level optimal decision rate.
Raw returns are undiscounted original rewards, retaining .01/action costs.
Separate post-hoc time-neutral returns add back those costs; they do not replace
the rewards fed to the encoder and do not imply training used terminal refunds.

ODR decisions are the last departure from the route fork (3,4) before episode
termination. A return to the fork clears the pending decision, allowing an
inspection-and-return strategy to choose a different route. No departure is
incorrect. This measures route choice, independently of execution success.
The oracle executes three canonical route plans with known gate and budget,
avoiding collisions if a route is known blocked. Its time-neutral values retain
danger and failure costs. At openD25, direct/safe tie. At infeasible closedD15,
direct/safe prefixes tie at-1 and danger scores-1.2; these trials remain included
in the primary ODR, with feasible-only results also reported. This oracle is
over the stated route decision set, not a proof of globally optimal control.

95% confidence intervals use5000 trial-bootstrap replicates. They describe
uncertainty in sampling the six-task mixture, not training-seed variability.
Full per-action rewards, positions, actions, scores, source/checkpoint hashes,
and paired task definitions are preserved for audit and the paper case study.

Run each `python3 -m discrete_paper_comparison.train --mode MODE --output NEW_DIR`.
After all three runs finish, run `python3 -m discrete_paper_comparison.evaluate
--output discrete_paper_comparison/evaluation_100trials_seed404131`.

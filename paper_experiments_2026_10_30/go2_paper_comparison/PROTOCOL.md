# Go2 paired comparison

Protected full model: `../deadline_go2_gates/runs/half_bc_steps100k_seed0/best.pt`.
Original source files match the current environment, simulator, layout, teacher,
model and training source exactly. `archive_before.json` records all 33 protected
files. All comparison outputs go here; source snapshots are in `runtime_source`.

Three fresh single-seed input ablations use the exact archived demonstration
file, with its SHA256 recorded per run. No new demonstrations are collected.
State-only and time-aware process each current observation as a one-token
sequence with zero GRU state. Context-only retains causal episode history and
boundary GRU updates but zeros state slots 10 and 11 (both time features).
Full model weights and inference architecture are unchanged.

Common training: 200 BC updates (Adam 1e-3), then at least 100,000 active 2 Hz
high-level transitions, completing the last trial batch. PPO Adam 3e-5,
3 epochs, 2 minibatches, clip 0.2, value weight 0.5, action entropy weight
0.001, gradient norm 0.5; full-trial undiscounted return. Auxiliary BC weight
1 declines linearly to zero by 80,000 transitions, sampled at rollout start.
12 parallel physical trials; deadlines 18/48/100 seconds; original frozen
Full_8000 locomotion at 50 Hz, Genesis CPU physics and actual 2 m lidar.
Validation every 5 updates and at the end; checkpoint selection uses mean
third-episode delivered value over all 12 deadline/gate cases, matching the
archived full model. Tests do not select checkpoints.

`run_all.py` runs state, time and context sequentially, stops on any failure,
and then runs evaluation automatically. `queue_status.json` and individual
logs expose status. No hardware is accessed.

Held-out evaluation: 100 paired trials, seed 404134, deadline uniform over
0.5-second ticks from 18 to 100 seconds; two independent Bernoulli(0.5) gates.
Same task and joint-initialization seed for every method, 3 episodes/trial,
fixed map, deterministic inference and trial-boundary memory reset. A separate
12-case evaluation retains original training deadlines. Rewards are delivered
value /4. Trial/final returns and delivery rates are reported, with 5,000
trial bootstrap samples for return CIs. Reference route timings do not prove
physical global optimality: ODR is intentionally withheld until its reference
labels have been independently validated. No unverified ODR is fabricated.

Thirteen protocol tests passed, including exact compatibility with the full
model, causal inference, no state/time history leakage, both time inputs
masked, rollout/replay likelihood parity for all variants, cross-episode
credit and BC schedule. This is single-seed, seen-pattern/fixed-layout
simulation; not evidence of hardware or unseen-layout generalization.

# Time Adaptation Meta-RL

Research code and experiment artifacts for deadline-aware reinforcement learning,
Bayesian adaptation, and meta-RL in discrete and continuous navigation tasks.

## Contents

- `tidyup_plan2/`: discrete baselines, belief-based methods, and experiment artifacts.
- `continuous_pointnav_adaptation/`: continuous point-navigation environments and trials.

Third-party source trees and superseded exploratory work are intentionally excluded.

## Current discrete result: 0–64.5w

The curated [release](tidyup_plan2/baseline3_gate_meta_rl/release_0_64p5w/README.md)
contains the last-episode return curves, their evaluation records, training logs,
selected checkpoints, and commands to reproduce plots, evaluation and continuation.
`w` means 10,000 environment steps. Later local training is not included in this release.

![Episode-3 return, 0–64.5w](tidyup_plan2/baseline3_gate_meta_rl/release_0_64p5w/third_episode_return.png)

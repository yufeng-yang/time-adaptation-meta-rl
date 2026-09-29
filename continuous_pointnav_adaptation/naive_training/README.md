# Naive continuous PPO

This baseline trains one PPO policy directly on the Point agent's continuous
actions.  It contains no route labels, waypoint controllers, options, belief,
or deadline input.

- Episode limit: 1000 simulator steps.
- Purple-wall lidar: ordinary 16-bin pseudo lidar, range 3.
- Yellow-gate lidar: the same ordinary 16-bin pseudo lidar, range 1.
- An open gate produces no yellow-gate return because its geoms are inactive.
- The gate state is sampled independently at reset with probability 0.5 open.

Run from the repository root with the `sb3sg` environment:

```bash
conda run -n sb3sg python -m continuous_pointnav_adaptation.naive_training.train
```

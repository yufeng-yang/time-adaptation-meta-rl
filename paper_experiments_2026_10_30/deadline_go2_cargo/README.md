# Experiment 4: unloaded Go2 cargo-trial feasibility

This is the retained earlier hidden-value cargo pilot. The active next
environment is [deadline_go2_gates](../deadline_go2_gates/README.md): local
voxel sensing, trial-persistent gates, and known fixed near/far rewards.

The ordinary version uses the same unloaded robot for every cargo value. It
tests deadline-dependent exploration and cross-episode memory. There is no
load-dependent mass, speed limit, friction, or reward shaping. Separate
[physical payload checks](payload_tests/README.md) are retained for reference.

Three fixed stations lie in the forward 180-degree half-plane: (2, 0),
(3, +5.196), (3, -5.196) metres. The base is the origin. Each trial permutes
values {1,2,4}, held fixed for three episodes. Contact reveals the value and
automatically loads/replaces one logical cargo. Delivery ends the episode;
timeout or instability yields zero. Cargo and stations are task markers;
cargo rigid-body dynamics are not simulated in this unloaded version.

## Inputs and structure

The original run below used `map_source=synthetic`. The new optional
`map_source=lidar_voxel` sensor pipeline is described in
[VOXEL_MAP.md](VOXEL_MAP.md). It has passed sensor/rollout checks, but has no
new trained policy yet. Existing checkpoints and archived results are retained.

- A synthetic local **3×24×24 grid** encodes occupancy, height, and observed
  range. Station surface points are projected using robot pose within an ideal
  8 m circular range. It is not real lidar raycasting, a dense 3D voxel tensor,
  an RViz screenshot, or a ROS VoxelMap adapter. Noise and occlusion are absent.
- The **24-value state** contains base/station relative positions, heading,
  remaining time/deadline, carried value/status, contact feedback, previous
  command, episode index, previous reward, and terminal flag. All station
  coordinates and ideal odometry are supplied, so the grid is not the sole
  source of geometry. Hidden values do not change pre-contact geometry/state.
- Two 2D convolution layers encode the grid into 32 features. Combined state
  and grid features enter a 64-wide, two-layer, four-head causal Transformer.
  Attention/max episode pooling updates a 64-wide GRU only at boundaries.
  The actor uses current Transformer features and previous-episode GRU state.

Actions are body vx in ±0.6 m/s and vy in ±0.4 m/s. Yaw command is fixed at
zero, matching the existing low-level policy's training range. Actual heading
can drift; target-relative feedback compensates using vx/vy. Arbitrary yaw
control has not been established. High-level control is 2 Hz; frozen locomotion
is 50 Hz.

## Simulation and deadlines

Training runs actual unloaded Go2 locomotion in **Genesis 0.4.6**, using local
Full_8000.pt and its matching configuration. Low-level weights, PD gains,
robot mass, and ground are fixed. It is not the factory Unitree Sport controller.
The MuJoCo test0.py window is a separate static scene preview. The policy
transfer to MuJoCo has not passed empty walking. No hardware or SDK is used.

The unloaded navigation check measured about 6.5 seconds out-and-back to the
near station and 28.5/24.5 seconds to the two far stations, without instability.
See payload_tests/unloaded_navigation_check.json. A 10-second teacher check
was too tight for its conservative planning bound in later episodes; the
short deadline was increased to 12 seconds. Current deadlines are:

- **12 seconds:** deliver the near cargo.
- **34 seconds:** inspect a subset, then exploit remembered values later.
- **65 seconds:** inspect multiple stations and deliver.

With the public {1,2,4} permutation, observing two values identifies the third;
three visits are not necessary. The observed-history teacher was checked on
all six assignments and three initial-state seeds, with all deliveries
successful. Final-episode means were 2.333, 4, and 4. These are feasibility
checks, not robust timing bounds for varied maps or real sensors.

## Training and primary metric

"Half BC" means demonstration warmstart followed by PPO plus an auxiliary
BC loss with a decaying coefficient, **not optimizer weight decay**:
`loss = PPO_actor + 0.5 * critic - 0.001 * entropy + lambda * BC_MSE`,
where `lambda = 0.2 * max(0, 1 - update / (PPO_updates / 2))`.
For 200 planned PPO updates this reaches zero at update 100; at update 20
it is 0.16. The selected checkpoint in the original Go2 pilot is the BC
warmstart, because its subsequent PPO checkpoint performed worse.

The user prioritizes **episode 3**, reported separately for every deadline and
used to recommend a checkpoint. Episodes 1–2 are exploration context and
secondary measurements. The first run's learning objective still sums three
delivered values / 4 with gamma=1; final-only reward training has not been run.

[runs/unloaded_half_bc_seed0](runs/unloaded_half_bc_seed0/README.md) completed
1,000 BC updates and saved a PPO-update-20 comparison. PPO reduced episode-3
performance, so further updates were intentionally stopped; the initial
200-update PPO target was not completed. **best_episode3.pt** contains the
recommended BC weights, not degraded PPO weights. source/ preserves the
actual launch code. The current top-level trainer defaults checkpoint
selection to episode 3 and saves every five updates.

Evaluation enumerates six seen value assignments at each deadline, with
validation initial-joint seeds 300000+ and held-out initial-condition seeds
400000+. This is single-seed feasibility, not generalization to unseen latent
tasks or real sensors. Clearing GRU memory at inference is a diagnostic,
not an independently trained no-memory ablation.

## Commands

Static map preview: drag to orbit, ] selects head camera, Esc returns to free
camera. Add --camera head to start there. The virtual camera is a viewing aid,
not real Go2 camera calibration.

```bash
/home/yufeng.yang/Downloads/enter/envs/time_adapt/bin/python /home/yufeng.yang/codespace/time_adaptation/deadline_go2_cargo/test0.py
```

Eight protocol/model checks include hidden-value secrecy, unloaded dynamics,
causal attention, and matching rollout/PPO likelihoods:

```bash
/home/yufeng.yang/Downloads/enter/envs/time_adapt/bin/python /home/yufeng.yang/codespace/time_adaptation/deadline_go2_cargo/test_protocol.py
```

Evaluate a saved checkpoint independently to a fresh output filename:

```bash
GS_CACHE_FILE_PATH=/tmp/time_adapt_genesis_cache XDG_CACHE_HOME=/tmp/time_adapt_cache MPLCONFIGDIR=/tmp/time_adapt_mpl /home/yufeng.yang/Downloads/enter/envs/genesis/bin/python /home/yufeng.yang/codespace/time_adaptation/deadline_go2_cargo/evaluate.py /home/yufeng.yang/codespace/time_adaptation/deadline_go2_cargo/runs/unloaded_half_bc_seed0/best_episode3.pt --output /tmp/go2_episode3_evaluation.json
```

Add --memory-reset for the inference diagnostic. The run archives low-level
weights/config and high-level checkpoints. Runtime still references the
original robot URDF in the local RTSS artifact tree, recorded in source.

![Static map preview](map_preview.png)

The final_experiment_2 and final_experiment_3 archives are unchanged.

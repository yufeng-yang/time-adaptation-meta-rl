# Simulated lidar voxel-map input, version 1

Implemented and checked against installed Genesis 0.4.6. The original
synthetic-map checkpoint and its recorded results are unchanged. This is
sensor integration, with no new policy training or score benchmark yet.

## Data path

`Genesis first-hit raycaster -> XYZ points + ray distances/directions ->`
`robot-centred 3D voxels -> 3-channel local BEV -> existing 2D CNN ->`
`causal Transformer + cross-episode GRU`.

- Sensor follows the actual body's roll, pitch, and yaw, at offset
  (0.30, 0, 0.15) m. This is a virtual generic lidar, not calibrated Unitree
  L1/L2 hardware. Horizontal FOV 360 degrees, vertical [-45,+15] degrees,
  360 azimuth samples by 32 elevation samples, max range 8 m.
- The simulator scans its actual rigid collision-mesh scene. Three identical
  fixed boxes, 0.36 by 0.36 by 0.35 m, represent persistent loading stations.
  A reserved contact bit excludes box/robot contact forces while preserving
  surfaces for raycasting. Automatic loading/swapping remains logical, with
  identical unloaded locomotion for all scores. No cargo disappears or rides
  on the robot in this version.
- Local voxels have side 0.125 m, dimensions `(Z,Y,X)=(12,128,128)`, cover
  XY [-8,+8) m around the robot and world Z [0,1.5) m above the flat floor.
  XY is aligned with heading; height uses known ground Z=0.
- Values are -1 unknown, 0 sampled free, 1 occupied. Free-space samples are
  taken every 0.125 m along rays up to their first hit, never beyond it.
  This is sampled evidence rather than exact voxel DDA coverage or a
  probabilistic occupancy filter. No-hit rays supply free samples up to 8 m
  and never create occupied endpoints. Ground returns below 0.08 m are
  excluded from occupied endpoints. A conservative Go2 self box filters
  body returns; those masked rays do not clear space behind the robot.
- XY max pooling gives `3 x 64 x 64` at 0.25 m/cell: any occupied voxel,
  maximum occupied voxel-centre height clipped to [0,1] m, and any observed
  above-ground voxel in the column (also includes occupied cells). A known
  column does not imply its entire vertical extent has been observed.
  The existing CNN's adaptive pooling accepts this new resolution unchanged.
- All maps are rebuilt from the current scan, with no cross-scan accumulation,
  SLAM, range noise, or ROS VoxelMap adapter. The 24D state still supplies
  target-relative coordinates and ideal odometry, so this is not map-only
  navigation. Target coordinates and hidden scores are never passed to the
  voxelizer; coordinates are used only to construct task/scene geometry.

Genesis 0.4.6 caches sensor data on each physics step. After a robot reset,
the adapter explicitly refreshes its internal sensor manager so the initial
observation does not contain a previous episode's scan. This version-specific
call is isolated in the simulation adapter. Scans currently update at 50 Hz;
the high-level policy reads the latest scan at 2 Hz.

## Verified output

The [verified input image](inspection/lidar_voxel_v1_verified/voxel_input.png),
[raw sensor/voxel arrays](inspection/lidar_voxel_v1_verified/voxel_input.npz),
and [physical check records](inspection/lidar_voxel_v1_verified/inspection.json)
are saved together. They verify three visible targets, hidden-score secrecy,
real-input CNN/Transformer/GRU forward/backward, two-environment scan
separation, and first-hit occlusion against an analytic box-intersection
reference. Two short-deadline physical teacher trials completed all six
deliveries without falling. These are integration checks, not learned-policy
performance measurements. The logical map checks additionally cover misses,
unknown space behind hits, self filtering, ground, heading, and independence
from station coordinates/scores.

## Commands

New input integration and physical checks, using a fresh output directory:

```bash
/home/yufeng.yang/Downloads/enter/envs/genesis/bin/python /home/yufeng.yang/codespace/time_adaptation/deadline_go2_cargo/inspect_voxel_map.py --output /tmp/go2_voxel_inspection
```

Training now accepts `--map-source lidar_voxel`. The following starts a new
half-BC run; it has **not** been launched:

```bash
/home/yufeng.yang/Downloads/enter/envs/genesis/bin/python /home/yufeng.yang/codespace/time_adaptation/deadline_go2_cargo/train.py --map-source lidar_voxel --output /home/yufeng.yang/codespace/time_adaptation/deadline_go2_cargo/runs/lidar_voxel_half_bc_seed0 --bc-updates 1000 --ppo-updates 200 --primary-metric episode3
```

The BC coefficient decays from 0.2 to zero over the first 100 PPO updates.
Deadline/task/reward/memory settings remain those of the original unloaded
run. New checkpoints record `map_source`, voxel settings, and grid resolution;
the evaluator reconstructs these from the saved task configuration.

For reference, the [official raycaster guide](https://genesis-world.readthedocs.io/en/latest/user_guide/sensing/raycaster.html)
describes this sensor family. Runtime details were taken from the installed
0.4.6 source because latest documentation can describe a newer API.

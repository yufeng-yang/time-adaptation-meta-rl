"""Single-scan voxelization of first-hit lidar, with no task-geometry inputs.

Voxel order is Z,Y,X: -1 unknown, 0 sampled free, 1 occupied. World Z uses
the known flat floor at zero. XY is robot-centred and heading-aligned. This
is an ideal-odometry sensor adapter, not SLAM or a ROS message implementation.
"""
import numpy as np


def build_local_voxels(scan, pose, config):
    extent, size = getattr(config, 'map_extent', config.sensor_range), config.voxel_size
    scan_range = config.sensor_range
    n = round(2 * extent / size)
    nz = round(config.voxel_height / size)
    out = config.grid_size
    if abs(n * size - 2 * extent) > 1e-6 or n % out:
        raise ValueError('Voxel XY size must exactly cover the range and divide the BEV grid')
    if abs(nz * size - config.voxel_height) > 1e-6:
        raise ValueError('Voxel height must be a multiple of voxel size')
    points = np.asarray(scan['points']).reshape(-1, 3)
    distances = np.asarray(scan['distances']).reshape(-1)
    directions = np.asarray(scan['directions']).reshape(-1, 3)
    origin = np.asarray(scan['origins'])
    c, s = np.cos(pose[2]), np.sin(pose[2])
    rotation = np.array([[c, -s], [s, c]])

    def cells(world):
        body = (world[..., :2] - pose[:2]) @ rotation
        xy = np.floor((body + extent) / size).astype(np.int32)
        z = np.floor(world[..., 2] / size).astype(np.int32)
        valid = ((xy >= 0).all(-1) & (xy < n).all(-1) &
                 (z >= 0) & (z < nz) & (np.linalg.norm(body, axis=-1) <= extent))
        return xy, z, valid

    voxels = np.full((nz, n, n), -1, dtype=np.int8)
    hit = np.isfinite(distances) & (distances >= .15) & (distances < scan_range)
    # Conservative Go2 self mask in heading coordinates. Masked body returns
    # remain occluders: dropping the endpoint must not clear space behind it.
    body_xy = (points[:, :2] - pose[:2]) @ rotation
    body_hit = (hit & (np.abs(body_xy[:, 0]) <= .55) &
                (np.abs(body_xy[:, 1]) <= .30) & (points[:, 2] <= .75))
    usable = ~body_hit
    lengths = np.where(hit, distances, scan_range)
    samples = np.arange(0., scan_range, size, dtype=np.float32)
    # Sample only along first-hit rays. No fictitious free space behind surfaces.
    free_points = origin + directions[:, None, :] * samples[None, :, None]
    select = usable[:, None] & (samples[None, :] < lengths[:, None] - size / 2)
    free_points = free_points[select]
    xy, z, valid = cells(free_points)
    voxels[z[valid], xy[valid, 1], xy[valid, 0]] = 0
    obstacle = (hit & usable & np.isfinite(points).all(1) &
                (points[:, 2] >= config.ground_cutoff) &
                (points[:, 2] < config.voxel_height))
    xy, z, valid = cells(points[obstacle])
    voxels[z[valid], xy[valid, 1], xy[valid, 0]] = 1
    occupied = voxels == 1
    heights = (np.arange(nz, dtype=np.float32) + .5) * size
    above_ground = heights >= config.ground_cutoff
    occupancy = occupied.any(0).astype(np.float32)
    height = (occupied * heights[:, None, None]).max(0).clip(0., 1.)
    observed = ((voxels[above_ground] >= 0).any(0) | occupied.any(0)).astype(np.float32)
    fine = np.stack([occupancy, height, observed])
    factor = n // out
    grid = fine.reshape(3, out, factor, out, factor).max(axis=(2, 4))
    return dict(grid=grid.astype(np.float32), voxels=voxels,
                hit_count=int(hit.sum()), obstacle_hits=int(obstacle.sum()),
                obstacle_mask=obstacle,
                occupied_voxels=int(occupied.sum()), observed_voxels=int((voxels >= 0).sum()))

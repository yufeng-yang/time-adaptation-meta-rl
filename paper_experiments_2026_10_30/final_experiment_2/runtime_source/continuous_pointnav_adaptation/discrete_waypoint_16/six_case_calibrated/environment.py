"""Independent stretched three-route map; no mutation of the D=50 map."""
from pathlib import Path
import mujoco
import numpy as np

from continuous_pointnav_adaptation.naive_training.environment import LocalGateLidarTask
from continuous_pointnav_adaptation.discrete_waypoint_16.environment import segment_box_distance_squared
from continuous_pointnav_adaptation.discrete_waypoint_16.trial_bc_radar.environment import (
    RadarScene, RadarWaypointEnv,
)

DEADLINES = (30, 43, 100)
TIME_SCALE = 50.  # Fixed scale; D100 starts at 2, without clipping.
CALIBRATION_FILE = Path(__file__).parent/"calibration_D43_upper38_T50/results.json"
STEP_PENALTY = .005
UPPER_Y, LOWER_Y = 2.75, -4.75
DANGER_CENTER = np.array([0., UPPER_Y])
DANGER_HALF_SIZE = .49
MAP_VERSION = "stretched_routes_left_upper_entry_v1"
ROUTES = {
    "direct": ((2.5, 0.),),
    "upper": ((-2.6, 1.5), (-.75, 2.65), (.75, 2.65), (2.5, 0.)),
    "lower": ((-.8, LOWER_Y), (.8, LOWER_Y), (2.5, 0.)),
}


def wall_locations():
    points = set()
    for x in np.arange(-3.25, 3.251, .5):
        points.add((float(x), -5.75))
        points.add((float(x), 3.75))
    for y in np.arange(-5.25, 3.251, .5):
        points.add((-3.25, float(y)))
        points.add((3.25, float(y)))
    for y in np.arange(-5.5, 3.501, .5):
        if abs(y) <= .5 or 2.5 <= y <= 3. or -5. <= y <= -4.5:
            continue
        points.add((0., float(y)))
    # The upper branch is entered from the left. A central probe followed by
    # retreat must pay real backtracking distance rather than cutting upward
    # alongside the gate. A ~0.4m-wide center corridor remains around the tip.
    for x in (-2., -1.5, -1., -.5):
        points.add((x, 1.))
    return tuple(sorted(points))


class SeparatedRouteTask(LocalGateLidarTask):
    def __init__(self, config):
        super().__init__(config)
        self.purple_walls.locations = wall_locations()
        self.danger_patches.locations = (tuple(DANGER_CENTER),)
        self.placements_conf.extents = [-3.5, -6., 3.5, 4.]
        self.floor_conf.size = (3.5, 6.1, .1)
        self.placements_conf.placements = None


class SeparatedRouteScene(RadarScene):
    def _get_task(self):
        task = SeparatedRouteTask(self.config)
        task.build_observation_space()
        return task

    @staticmethod
    def _crosses_danger(start, end):
        return segment_box_distance_squared(np.asarray(start), np.asarray(end),
            DANGER_CENTER-DANGER_HALF_SIZE, DANGER_CENTER+DANGER_HALF_SIZE) == 0.


class SixCaseEnv(RadarWaypointEnv):
    def __init__(self, **kwargs):
        kwargs.setdefault("deadlines", DEADLINES)
        kwargs.setdefault("time_scale", TIME_SCALE)
        kwargs.setdefault("step_penalty", STEP_PENALTY)
        super().__init__(**kwargs)

    def _make_scene(self, **kwargs):
        return SeparatedRouteScene(**kwargs)

    @property
    def observation_spec(self):
        return super().observation_spec | dict(version="kinematic_radar_xy_sixcase_v1",
                                               map_version=MAP_VERSION,
                                               danger_center=DANGER_CENTER.tolist(),
                                               upper_entry="left of x=-2.42 at y=1")

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        # A diagnostic overhead camera fits the taller map. No policy sees it.
        model = self.scene.task.agent.engine.model
        camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixedfar")
        if camera >= 0:
            model.cam_pos[camera] = [0., -1., 14.]
            model.cam_quat[camera] = [1., 0., 0., 0.]
            model.cam_fovy[camera] = 45.
        return result


def make_env(**kwargs):
    return SixCaseEnv(**kwargs)

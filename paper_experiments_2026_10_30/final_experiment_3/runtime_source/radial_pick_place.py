"""Panda 内圈保底球和外圈四球场景；只实现布局与物理对象。"""

import xml.etree.ElementTree as ET

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.observables import Observable, sensor

from radial_layout import RadialLayout


class RadialPickPlace(ManipulationEnv):
    """以桌面中心为原点，内圈一个保底球、外圈四个随机位置的未知球。

    A/B/C/D 外观、尺寸和物理参数相同。S 的金色仅标识保底球身份。
    当前没有隐藏分数、自动抓取、deadline、跨局记忆或训练奖励。
    """

    def __init__(self, layout=None, **kwargs):
        self.layout = layout or RadialLayout()
        self.home_pos = np.zeros(3)
        self.origin = np.zeros(3)
        self._home_qpos = None
        defaults = dict(
            robots="Panda",
            initialization_noise=None,
            use_camera_obs=False,
            has_offscreen_renderer=False,
            hard_reset=False,
        )
        defaults.update(kwargs)
        super().__init__(**defaults)

    @staticmethod
    def _geom(parent, name, kind, pos, size, rgba, collision=False, **kwargs):
        return ET.SubElement(
            parent,
            "geom",
            name=name,
            type=kind,
            pos=array_to_string(pos),
            size=array_to_string(size),
            rgba=array_to_string(rgba),
            contype="1" if collision else "0",
            conaffinity="1" if collision else "0",
            group="1",
            **kwargs,
        )

    def _load_model(self):
        super()._load_model()
        self._home_qpos = None
        cfg = self.layout
        robot_model = self.robots[0].robot_model
        robot_model.set_base_xpos(robot_model.base_xpos_offset["table"](cfg.table_size[0]))
        arena = TableArena(
            table_full_size=cfg.table_size,
            table_friction=(1.0, 0.005, 0.001),
            table_offset=(0, 0, cfg.table_height),
        )
        arena.set_origin([0, 0, 0])
        arena.table_visual.attrib.pop("material", None)
        arena.table_visual.set("rgba", "0.24 0.28 0.32 1")

        # 盘子和圆周固定在桌面中心；出生位姿单独对齐到盘子正上方。
        markers = ET.SubElement(arena.worldbody, "body", name="layout_origin")
        for label, radius, color in (
            ("near", cfg.safe_radius, [1.0, 0.72, 0.18, 1]),
            ("far", cfg.unknown_radius, [0.25, 0.75, 0.9, 1]),
        ):
            for i, angle in enumerate(np.linspace(0, 2 * np.pi, 96, endpoint=False)):
                self._geom(
                    markers, f"{label}_ring_{i}", "cylinder",
                    [radius * np.cos(angle), radius * np.sin(angle), 0.001],
                    [0.0017, 0.0006], color,
                )

        # 原点的浅盘具有真实底面和盘沿，圆周标记则完全没有碰撞。
        self._geom(
            markers, "delivery_floor", "cylinder", [0, 0, 0.003],
            [cfg.tray_radius, 0.003], [0.28, 0.65, 0.48, 1], collision=True,
            friction="1 0.005 0.001",
        )
        for i, angle in enumerate(np.linspace(0, 2 * np.pi, 24, endpoint=False)):
            wall_radius = cfg.tray_radius
            self._geom(
                markers, f"delivery_wall_{i}", "box",
                [wall_radius * np.cos(angle), wall_radius * np.sin(angle), 0.014],
                [wall_radius * np.tan(np.pi / 24) + 0.001, 0.004, 0.01],
                [0.48, 0.85, 0.63, 1], collision=True,
                euler=array_to_string([0, 0, angle + np.pi / 2]),
            )
        self._geom(
            markers, "origin_cross_x", "box", [0, 0, 0.0065],
            [0.012, 0.001, 0.0004], [0.9, 1, 0.9, 1],
        )
        self._geom(
            markers, "origin_cross_y", "box", [0, 0, 0.0065],
            [0.001, 0.012, 0.0004], [0.9, 1, 0.9, 1],
        )

        self.balls = {
            label: BallObject(
                name=f"ball_{label}", size=[cfg.ball_radius], density=400,
                friction=[1.0, 0.005, 0.001],
                rgba=[1.0, 0.72, 0.18, 1] if label == "S" else [0.2, 0.65, 0.95, 1],
            )
            for label in ("S", "A", "B", "C", "D")
        }
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=list(self.balls.values()),
        )

    def _setup_references(self):
        super()._setup_references()
        self.ball_body_ids = {
            label: self.sim.model.body_name2id(ball.root_body)
            for label, ball in self.balls.items()
        }
        self.origin_body_id = self.sim.model.body_name2id("layout_origin")

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.forward()
        if self._home_qpos is None:
            self._initialize_centered_home()
        site_id = self.robots[0].eef_site_id["right"]
        self.home_pos = self.sim.data.site_xpos[site_id].copy()
        self.origin = np.array([0.0, 0.0, self.layout.table_height])
        self.sim.model.body_pos[self.origin_body_id] = self.origin
        self.ball_offsets = self.layout.sample_ball_offsets(self.rng)
        for label, offset in self.ball_offsets.items():
            pos = np.array([
                *(self.origin[:2] + offset),
                self.layout.table_height + self.layout.ball_radius + 0.001,
            ])
            ball = self.balls[label]
            self.sim.data.set_joint_qpos(ball.joints[0], np.r_[pos, [1, 0, 0, 0]])
            self.sim.data.set_joint_qvel(ball.joints[0], np.zeros(6))
        self.sim.forward()

    def _initialize_centered_home(self):
        """初始化时求解一次中心出生位姿，不运行轨迹、抓取或训练。"""
        import mujoco
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        robot = self.robots[0]
        model = self.sim.model._model
        data = mujoco.MjData(model)
        reference_qpos = self.sim.data.qpos.copy()
        arm_indices = np.asarray(robot._ref_joint_pos_indexes)
        joint_ids = [self.sim.model.joint_name2id(name) for name in robot.robot_joints]
        limits = model.jnt_range[joint_ids]
        site_id = robot.eef_site_id["right"]
        orientation = self.sim.data.site_xmat[site_id].reshape(3, 3).copy()
        target = np.array([
            0.0, 0.0, self.layout.table_height + self.layout.home_height_above_table,
        ])

        def residual(q):
            data.qpos[:] = reference_qpos
            data.qpos[arm_indices] = q
            mujoco.mj_fwdPosition(model, data)
            rotation_error = Rotation.from_matrix(
                orientation @ data.site_xmat[site_id].reshape(3, 3).T
            ).as_rotvec()
            return np.r_[data.site_xpos[site_id] - target, 0.2 * rotation_error]

        solution = least_squares(
            residual, reference_qpos[arm_indices],
            bounds=(limits[:, 0] + 0.001, limits[:, 1] - 0.001),
            max_nfev=300, gtol=1e-10, ftol=1e-10, xtol=1e-10,
        )
        error = residual(solution.x)
        if np.linalg.norm(error[:3]) > 0.001 or np.linalg.norm(error[3:]) > 0.002:
            raise RuntimeError("无法将夹爪出生位姿设置到桌面中心上方。")
        self._home_qpos = solution.x.copy()
        robot.init_qpos = self._home_qpos.copy()
        # 在新出生关节位置重新初始化控制器，避免首次动作跳回旧位姿。
        robot.reset(deterministic=True, rng=self.rng)
        self.sim.forward()

    def _setup_observables(self):
        observables = super()._setup_observables()

        def make_position_sensor(label):
            @sensor(modality="object")
            def ball_pos(obs_cache):
                return self.sim.data.body_xpos[self.ball_body_ids[label]].copy()

            ball_pos.__name__ = f"ball_{label}_pos"
            return ball_pos

        @sensor(modality="object")
        def home_pos(obs_cache):
            return self.home_pos.copy()

        @sensor(modality="object")
        def delivery_pos(obs_cache):
            return self.origin + [0, 0, 0.006 + self.layout.ball_radius]

        sensors = [make_position_sensor(label) for label in self.balls]
        for fn in [*sensors, home_pos, delivery_pos]:
            observables[fn.__name__] = Observable(
                name=fn.__name__, sensor=fn, sampling_rate=self.control_freq,
            )
        return observables

    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        return False

    def layout_summary(self):
        return {
            "origin_xyz": self.origin.tolist(),
            "home_eef_xyz": self.home_pos.tolist(),
            "ball_radius_m": self.layout.ball_radius,
            "outer_ball_count": 4,
            "outer_angle_ranges_degrees": self.layout.unknown_angle_ranges,
            "ball_positions_xyz": {
                label: self.sim.data.body_xpos[body_id].tolist()
                for label, body_id in self.ball_body_ids.items()
            },
            "ball_radii_from_origin_m": {
                label: float(np.linalg.norm(self.sim.data.body_xpos[body_id][:2] - self.origin[:2]))
                for label, body_id in self.ball_body_ids.items()
            },
        }

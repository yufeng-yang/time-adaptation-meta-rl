"""ThreeRouteEnv V4：开局都向东，三条路只是长度不同。

V4 保留 V3 的时间 / 隐藏门语义，但把几何改成：

- 起点朝东，前几步只能向东（曼哈顿下降），然后才分叉；
- 短路过门，14 步；
- 上边危险中路，16 步；
- 下边安全绕行，18 步。

建议仍使用 D={15, 17, 19}：

- D=15：门开可走直达；门关不可完成；
- D=17：门开走直达，门关只能走 16 步危险路线；
- D=19：门开走直达，门关可走 18 步安全绕行以避开危险。
"""

from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Floor, Goal, Wall

from .base_env_V3 import ThreeRouteHiddenGateEnvV3


class ThreeRouteHiddenGateEnvV4(ThreeRouteHiddenGateEnvV3):
    """V4 地图：共享向东走廊后分成短 / 中 / 长三条路。"""

    GRID_WIDTH = 12
    GRID_HEIGHT = 8

    START_POSITION = (1, 4)
    GOAL_POSITION = (10, 2)
    GATE_POSITION = (6, 4)
    DANGER_POSITIONS: set[tuple[int, int]] = {(6, 1)}

    # 0=left, 1=right, 2=forward；出生在 (1,4)，朝东。
    # 前 7 步全部前进，曼哈顿单调下降，然后折向目标。
    DIRECT_SAFE_ACTIONS = (
        2, 2, 2, 2, 2, 2, 2,
        0, 2,
        1, 2, 2,
        0, 2,
    )
    # 向东两步后左转走上边走廊，经过红色 danger，再下到终点。
    MEDIUM_PATH_ACTIONS = (
        2, 2,
        0, 2, 2, 2,
        1, 2, 2, 2, 2, 2, 2, 2,
        1, 2,
    )
    # 向东两步后右转走下边走廊，再沿右列爬到终点。
    DETOUR_SAFE_ACTIONS = (
        2, 2,
        1, 2, 2,
        0, 2, 2, 2, 2, 2, 2, 2,
        0, 2, 2, 2, 2,
    )

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        for x in range(width):
            for y in range(height):
                self.grid.set(x, y, Wall())

        # 共享向东走廊：x=1,2,3 的 y=4。x=1,2 没有上下开口，开局只能前进。
        shared_stem = {(1, 4), (2, 4), (3, 4)}
        # 短路：继续向东过门，在 (8,4) 折向 (8,3)-(10,3)-(10,2)。
        # 故意不挖 (9,4)，避免门开时走 (10,4) 抄到 12 步。
        direct_path = {
            (4, 4),
            (5, 4),
            (6, 4),
            (7, 4),
            (8, 4),
            (8, 3),
            (9, 3),
            (10, 3),
            (10, 2),
        }
        # 中路：x=3 向上到 y=1，再向东到右列后落到终点。
        medium_path = {
            (3, 3),
            (3, 2),
            (3, 1),
            *{(x, 1) for x in range(4, 11)},
            (10, 2),
        }
        # 绕行：x=3 向下到 y=6，向东后沿 x=10 爬回终点。
        detour_path = {
            (3, 5),
            (3, 6),
            *{(x, 6) for x in range(4, 11)},
            (10, 5),
            (10, 4),
            (10, 3),
            (10, 2),
        }
        fork_column = {(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6)}

        for x, y in shared_stem | direct_path | medium_path | detour_path | fork_column:
            self.grid.set(x, y, None)

        for x, y in self.DANGER_POSITIONS:
            self.grid.set(x, y, Floor("red"))

        if self.gate_state not in self.GATE_STATES:
            raise RuntimeError("生成地图前必须先确定 gate_state")
        self.put_obj(
            Door(
                "yellow",
                is_open=self.gate_state == "open",
                is_locked=False,
            ),
            *self.GATE_POSITION,
        )
        self.put_obj(Goal(), *self.GOAL_POSITION)
        self.agent_pos = self.START_POSITION
        self.agent_dir = 0
        self.mission = (
            "reach the goal; infer whether the hidden gate is open, "
            "and respect the remaining time"
        )


TwoRouteHiddenGateEnvV4 = ThreeRouteHiddenGateEnvV4

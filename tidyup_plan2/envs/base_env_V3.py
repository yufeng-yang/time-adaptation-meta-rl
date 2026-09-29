"""ThreeRouteEnv V3：显式剩余时间 + trial 级隐藏门状态。

V3 将需要适应的信息拆成两部分：

- policy 每个 episode 都可以观察原始剩余步数 ``D-t``；
- 门的开关状态对起点不可见，并在同一 trial 的 episode 之间保持不变。

视觉 observation 是固定北向、agent 居中的 5×5 局部地图。局部范围内不做
墙壁遮挡（墙仍然阻挡移动），图像也不随 agent 朝向旋转。

三条解析路线（每步 -0.01，到达终点额外 +1.0；转向也算一步）。
势函数 shaping 使用 ``Φ = -0.02 * 曼哈顿``，``F = Φ(s') - Φ(s)``（对应 γ=1）：
靠近目标 +0.02，走远 -0.02。任意成功路径的 shaping 总和相同，无法靠来回摆动刷分。
撞到墙或关闭的门时停在原地（MiniGrid no-op），额外奖励 -0.05，不因此终止。

- 安全直达：14 步，过门，只在门打开时可走；
- 危险路线：16 步，经过上边红色 danger zone，不经过门；
- 安全绕行：18 步，门关闭时从门下绕到右列再上去。

建议使用 D={15, 17, 19}：

- D=15：门开可走直达；门关不可完成；
- D=17：门开走直达，门关只能走 16 步危险路线；
- D=19：门开走直达，门关可走 18 步安全绕行以避开危险。

trial API：

- ``reset(options={"new_trial": True, "deadline": D})``：采样新门状态；
- ``reset(options={"new_trial": False, "deadline": D})``：保持当前门状态；
- ``gate_state="open"/"closed"`` 可用于测试和 oracle evaluation。
"""

from __future__ import annotations

import os

os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "mesa")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("SDL_RENDER_DRIVER", "software")

import numpy as np
import pygame
import pygame.freetype
from gymnasium import spaces
from gymnasium.utils import seeding
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Floor, Goal, Wall
from minigrid.minigrid_env import MiniGridEnv


class ThreeRouteHiddenGateEnvV3(MiniGridEnv):
    """三路线环境；隐藏门状态跨 episode 保持，deadline 每局可变。"""

    GATE_STATES = ("open", "closed")
    DEFAULT_DEADLINES = (15, 17, 19)
    LOCAL_VIEW_SIZE = 5
    GRID_WIDTH = 10
    GRID_HEIGHT = 8

    DANGER_PENALTY = 0.2
    TIMEOUT_PENALTY = -1.0
    COLLISION_PENALTY = -0.05
    DEFAULT_STEP_PENALTY = 0.01
    DEFAULT_PROGRESS_SCALE = 0.02

    START_POSITION = (1, 4)
    GOAL_POSITION = (8, 2)
    GATE_POSITION = (6, 5)
    # 上方 16 步路线中的红色可通行区域。每个 episode 首次进入时扣一次。
    DANGER_POSITIONS: set[tuple[int, int]] = {(5, 1)}

    # 0=left, 1=right, 2=forward；出生在 (1,4)，朝东。
    # 安全直达：过门。
    DIRECT_SAFE_ACTIONS = (
        1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2
    )
    # 中型危险路线：上边走廊再下来。
    MEDIUM_PATH_ACTIONS = (
        0, 2, 2, 1, 2, 0, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2
    )
    # 安全绕行：门下绕过。
    DETOUR_SAFE_ACTIONS = (
        1, 2, 0, 2, 2, 2, 2,
        1, 2, 0, 2, 2, 2, 0, 2, 2, 2, 2
    )

    def __init__(
        self,
        render_mode: str | None = None,
        min_deadline: int = 15,
        max_deadline: int = 19,
        step_penalty: float = DEFAULT_STEP_PENALTY,
        progress_scale: float = DEFAULT_PROGRESS_SCALE,
        danger_penalty: float = DANGER_PENALTY,
        defer_danger_penalty: bool = False,
    ) -> None:
        if min_deadline <= 0 or max_deadline <= 0:
            raise ValueError("min_deadline / max_deadline 必须为正整数")
        if min_deadline > max_deadline:
            raise ValueError("min_deadline 不能大于 max_deadline")
        if step_penalty < 0:
            raise ValueError("step_penalty 必须为非负数")
        if progress_scale < 0:
            raise ValueError("progress_scale 必须为非负数")
        if danger_penalty < 0:
            raise ValueError("danger_penalty 必须为非负数")

        self.min_deadline = int(min_deadline)
        self.max_deadline = int(max_deadline)
        self.step_penalty = float(step_penalty)
        self.progress_scale = float(progress_scale)
        self.danger_penalty = float(danger_penalty)
        self.defer_danger_penalty = bool(defer_danger_penalty)
        self.deadline = self.max_deadline
        self.gate_state: str | None = None
        self.trial_id = -1
        self.episode_in_trial = -1
        self.danger_visits = 0
        self.danger_history: list[dict] = []
        self.danger_cost = 0.0
        self.episode_return = 0.0
        self.last_progress_reward = 0.0
        self.show_local_obs = False

        mission_space = MissionSpace(
            mission_func=lambda: (
                "reach the goal; infer whether the hidden gate is open, "
                "and respect the remaining time"
            )
        )
        super().__init__(
            mission_space=mission_space,
            width=self.GRID_WIDTH,
            height=self.GRID_HEIGHT,
            max_steps=self.max_deadline,
            agent_view_size=self.LOCAL_VIEW_SIZE,
            # centered observation 使用全可见 mask；墙只阻挡移动，不遮挡视觉。
            see_through_walls=True,
            render_mode=render_mode,
        )
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.agent_view_size, self.agent_view_size, 3),
                    dtype=np.uint8,
                ),
                "direction": spaces.Discrete(4),
                # base env 同时提供 D-t 与 D；训练 wrapper 应只选择 D-t。
                "time": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Discrete(3)
        self.reward_range = (-np.inf, np.inf)

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        for x in range(width):
            for y in range(height):
                self.grid.set(x, y, Wall())

        danger_path = {
            (1, 2),
            (2, 2),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (7, 1),
            (8, 1),
        }
        direct_path = (
            {(1, y) for y in range(2, 6)}
            | {(x, 5) for x in range(1, 9)}
            | {(8, y) for y in range(1, 6)}
        )
        # 主路沿 y=5 横向通过；门关闭时沿 y=6 绕到右列再上去。
        safe_detour = {(5, 6), (6, 6), (7, 6), (8, 6)}

        for x, y in danger_path | direct_path | safe_detour:
            self.grid.set(x, y, None)

        # Floor is traversable and encodes/renders the danger zone in red.
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

    def _is_traversable(self, x: int, y: int) -> bool:
        """Whether forward can enter a cell under the current true gate state."""

        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        obj = self.grid.get(x, y)
        return obj is None or bool(obj.can_overlap())

    @property
    def remaining_steps(self) -> int:
        return max(self.deadline - self.step_count, 0)

    def _reward(self) -> float:
        """到达目标的 terminal bonus；逐步代价在 step() 中统一扣除。"""
        return 1.0

    def gen_obs(self) -> dict:
        """固定北向、agent 居中的 5×5 局部地图，无墙壁视觉遮挡。"""
        half_view = self.LOCAL_VIEW_SIZE // 2
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        local_grid = self.grid.slice(
            agent_x - half_view,
            agent_y - half_view,
            self.LOCAL_VIEW_SIZE,
            self.LOCAL_VIEW_SIZE,
        )
        visibility_mask = np.ones(
            (self.LOCAL_VIEW_SIZE, self.LOCAL_VIEW_SIZE),
            dtype=bool,
        )
        image = local_grid.encode(visibility_mask)
        return {
            "image": image,
            "direction": self.agent_dir,
            "time": np.asarray(
                [self.remaining_steps, self.deadline],
                dtype=np.float32,
            ),
        }

    def _sample_gate_state(self) -> str:
        index = int(self.np_random.integers(0, len(self.GATE_STATES)))
        return self.GATE_STATES[index]

    def reset(self, *, seed=None, options=None):
        options = {} if options is None else dict(options)
        if seed is not None:
            # gate_state 必须在 super().reset() 生成地图前确定。
            self.np_random, _ = seeding.np_random(seed)

        requested_gate = options.get("gate_state")
        new_trial = bool(options.get("new_trial", self.gate_state is None))
        if requested_gate is not None:
            requested_gate = str(requested_gate)
            if requested_gate not in self.GATE_STATES:
                raise ValueError(
                    f"gate_state 必须属于 {self.GATE_STATES}，"
                    f"收到 {requested_gate!r}"
                )
            self.gate_state = requested_gate
            new_trial = True
        elif new_trial or self.gate_state is None:
            self.gate_state = self._sample_gate_state()

        if new_trial:
            self.trial_id += 1
            self.episode_in_trial = 0
        else:
            self.episode_in_trial += 1

        requested_deadline = options.get("deadline")
        if requested_deadline is None:
            self.deadline = int(
                self.np_random.integers(
                    self.min_deadline,
                    self.max_deadline + 1,
                )
            )
        else:
            self.deadline = int(requested_deadline)
            if self.deadline <= 0:
                raise ValueError("deadline 必须是正整数")
        self.max_steps = self.deadline

        self.danger_visits = 0
        self.danger_history = []
        self.danger_cost = 0.0
        self.episode_return = 0.0
        self.last_progress_reward = 0.0
        super().reset(seed=None, options=options)

        observation = self.gen_obs()
        info = self._info(
            entered_danger=False,
            danger_penalty_applied=False,
            success=False,
            timeout=False,
            collision=False,
        )
        return observation, info

    def _info(
        self,
        *,
        entered_danger: bool,
        danger_penalty_applied: bool,
        success: bool,
        timeout: bool,
        collision: bool,
    ) -> dict:
        # gate_state 供 trainer/logger/oracle 使用，不能拼入 policy observation。
        return {
            "deadline": self.deadline,
            "remaining_steps": self.remaining_steps,
            "gate_state": self.gate_state,
            "trial_id": self.trial_id,
            "episode_in_trial": self.episode_in_trial,
            "danger_visits": self.danger_visits,
            "danger_history": list(self.danger_history),
            "danger_cost": self.danger_cost,
            "episode_return": self.episode_return,
            "step_penalty": self.step_penalty,
            "progress_scale": self.progress_scale,
            "danger_penalty": self.danger_penalty,
            "defer_danger_penalty": self.defer_danger_penalty,
            "progress_reward": self.last_progress_reward,
            "entered_danger": entered_danger,
            "danger_penalty_applied": danger_penalty_applied,
            "success": success,
            "timeout": timeout,
            "collision": collision,
        }

    def step(self, action):
        action = int(action)
        previous_pos = tuple(int(v) for v in self.agent_pos)
        goal_x, goal_y = self.GOAL_POSITION
        previous_manhattan = abs(previous_pos[0] - goal_x) + abs(
            previous_pos[1] - goal_y
        )

        collision = False
        if action == 2:  # MiniGrid Actions.forward
            direction_vectors = ((1, 0), (0, 1), (-1, 0), (0, -1))
            dx, dy = direction_vectors[int(self.agent_dir)]
            target_x, target_y = previous_pos[0] + dx, previous_pos[1] + dy
            collision = not self._is_traversable(target_x, target_y)

        _, reward, base_terminated, _, _ = MiniGridEnv.step(self, action)
        terminated = bool(base_terminated)
        reward -= self.step_penalty
        current_pos = tuple(int(v) for v in self.agent_pos)
        current_manhattan = abs(current_pos[0] - goal_x) + abs(
            current_pos[1] - goal_y
        )
        # Ng potential-based shaping with Φ(s) = -progress_scale * Manhattan.
        # For γ=1 this is F = Φ(s') - Φ(s), so moving away and back nets zero.
        self.last_progress_reward = self.progress_scale * (
            previous_manhattan - current_manhattan
        )
        reward += self.last_progress_reward

        entered_danger = (
            current_pos != previous_pos
            and current_pos in self.DANGER_POSITIONS
        )
        danger_penalty_applied = False
        if entered_danger:
            self.danger_visits += 1
            self.danger_history.append(
                {"step": self.step_count, "position": current_pos}
            )
            if self.danger_visits == 1:
                self.danger_cost += self.danger_penalty
                if not self.defer_danger_penalty:
                    danger_penalty_applied = True
                    reward -= self.danger_penalty

        if collision:
            reward += self.COLLISION_PENALTY

        timeout = self.step_count >= self.deadline and not terminated
        if timeout:
            reward += self.TIMEOUT_PENALTY
            # deadline 是任务本身的真实终止，不是采样器的人为截断。
            terminated = True

        # Optional episodic settlement used by the BRL control experiments.
        # Charge on every true episode ending, including timeout, so entering
        # danger cannot avoid its cost by deliberately failing to reach Goal.
        if terminated and self.defer_danger_penalty and self.danger_cost > 0.0:
            reward -= self.danger_cost
            danger_penalty_applied = True

        truncated = False
        self.episode_return += float(reward)
        observation = self.gen_obs()
        info = self._info(
            entered_danger=entered_danger,
            danger_penalty_applied=danger_penalty_applied,
            success=bool(base_terminated),
            timeout=timeout,
            collision=collision,
        )
        return observation, float(reward), terminated, truncated, info

    def _local_view_highlight_mask(self) -> np.ndarray:
        """与 gen_obs 一致：agent 居中、固定北向的 LOCAL_VIEW_SIZE 方形视野。"""
        highlight_mask = np.zeros((self.width, self.height), dtype=bool)
        half_view = self.LOCAL_VIEW_SIZE // 2
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        for dx in range(-half_view, half_view + 1):
            for dy in range(-half_view, half_view + 1):
                x, y = agent_x + dx, agent_y + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    highlight_mask[x, y] = True
        return highlight_mask

    def get_full_render(self, highlight, tile_size):
        """全图渲染，但高亮区域对齐自定义 5×5 居中观测，而非默认 egocentric FOV。"""
        highlight_mask = (
            self._local_view_highlight_mask()
            if highlight
            else np.zeros((self.width, self.height), dtype=bool)
        )
        return self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask,
        )

    def render_local_observation(self, tile_size: int | None = None) -> np.ndarray:
        """把当前 policy 看到的 5×5 局部格子画成 RGB（便于 show 对照）。"""
        tile_size = int(tile_size or self.tile_size)
        half_view = self.LOCAL_VIEW_SIZE // 2
        agent_x, agent_y = int(self.agent_pos[0]), int(self.agent_pos[1])
        local_grid = self.grid.slice(
            agent_x - half_view,
            agent_y - half_view,
            self.LOCAL_VIEW_SIZE,
            self.LOCAL_VIEW_SIZE,
        )
        return local_grid.render(
            tile_size,
            agent_pos=(half_view, half_view),
            agent_dir=int(self.agent_dir),
        )

    def render(self):
        if self.render_mode != "human" or not self.show_local_obs:
            return MiniGridEnv.render(self)

        full_img = self.get_frame(self.highlight, self.tile_size, False)
        local_img = self.render_local_observation(self.tile_size)
        full_img = np.transpose(full_img, axes=(1, 0, 2))
        local_img = np.transpose(local_img, axes=(1, 0, 2))

        pygame.freetype.init()
        pygame.display.init()
        if self.clock is None:
            self.clock = pygame.time.Clock()

        gap = 16
        pad = 16
        label_h = 40
        full_w, full_h = int(full_img.shape[0]), int(full_img.shape[1])
        local_w, local_h = int(local_img.shape[0]), int(local_img.shape[1])
        src_w = max(full_w, local_w) + 2 * pad
        src_h = pad + full_h + gap + local_h + label_h + pad
        win_w = int(getattr(self, "window_width", None) or self.screen_size or 640)
        win_h = int(getattr(self, "window_height", None) or self.screen_size or 640)
        win_w = max(1, win_w)
        win_h = max(1, win_h)
        if self.window is None or self.window.get_size() != (win_w, win_h):
            self.window = pygame.display.set_mode((win_w, win_h))
            pygame.display.set_caption(self.__class__.__name__)

        full_surf = pygame.surfarray.make_surface(full_img)
        local_surf = pygame.surfarray.make_surface(local_img)
        bg = pygame.Surface((src_w, src_h))
        bg.convert()
        bg.fill((255, 255, 255))
        bg.blit(full_surf, ((src_w - full_w) // 2, pad))
        bg.blit(local_surf, ((src_w - local_w) // 2, pad + full_h + gap))

        font_size = 18
        font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
        text = (
            f"{self.mission}  |  "
            f"local {self.LOCAL_VIEW_SIZE}x{self.LOCAL_VIEW_SIZE} north-up"
        )
        text_rect = font.get_rect(text, size=font_size)
        font.render_to(
            bg,
            ((src_w - text_rect.width) // 2, src_h - pad - font_size),
            text,
            size=font_size,
        )

        scale = min(win_w / src_w, win_h / src_h)
        content_w = max(1, int(round(src_w * scale)))
        content_h = max(1, int(round(src_h * scale)))
        bg = pygame.transform.smoothscale(bg, (content_w, content_h))
        canvas = pygame.Surface((win_w, win_h))
        canvas.fill((255, 255, 255))
        canvas.blit(
            bg,
            ((win_w - content_w) // 2, (win_h - content_h) // 2),
        )
        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()
        return None


TwoRouteHiddenGateEnvV3 = ThreeRouteHiddenGateEnvV3

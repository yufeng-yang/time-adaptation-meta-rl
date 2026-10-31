"""Shared layout geometry for physics preview and abstract high-level training."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RadialLayout:
    table_size: tuple = (0.9, 0.9, 0.05)
    table_height: float = 0.8
    home_height_above_table: float = 0.12
    ball_radius: float = 0.025
    safe_radius: float = 0.18
    unknown_radius: float = 0.36
    unknown_angle_ranges: tuple = ((80.0, 145.0), (215.0, 280.0))
    minimum_ball_separation: float = 0.10
    tray_radius: float = 0.065

    def __post_init__(self):
        if not 0 < self.ball_radius < self.tray_radius:
            raise ValueError("投放盘半径必须大于球半径。")
        if not self.tray_radius + self.ball_radius + 0.015 < self.safe_radius:
            raise ValueError("保底球与投放盘之间需要留出空间。")
        if not self.safe_radius < self.unknown_radius:
            raise ValueError("未知球所在圆的半径必须大于保底球半径。")
        if self.unknown_radius + self.ball_radius >= min(self.table_size[:2]) / 2:
            raise ValueError("远球圆周需要完整落在桌面范围内，并留出球半径。")
        if not self.unknown_angle_ranges or any(
            not 0 <= start < end <= 360 for start, end in self.unknown_angle_ranges
        ):
            raise ValueError("外圈随机角度区间需要在 0–360 度之间。")
        if self.minimum_ball_separation <= 2 * self.ball_radius:
            raise ValueError("最小球心距离必须大于球直径。")

    def sample_ball_offsets(self, rng):
        """按圆弧长度均匀抽样外圈位置，拒绝球间距过小的摆放。"""
        offsets = {"S": np.array([-self.safe_radius, 0.0])}
        lengths = np.array([end - start for start, end in self.unknown_angle_ranges])
        for label in ("A", "B", "C", "D"):
            for _ in range(1000):
                arc = int(rng.choice(len(lengths), p=lengths / lengths.sum()))
                angle = np.deg2rad(rng.uniform(*self.unknown_angle_ranges[arc]))
                candidate = self.unknown_radius * np.array([np.cos(angle), np.sin(angle)])
                if all(
                    np.linalg.norm(candidate - placed) >= self.minimum_ball_separation
                    for placed in offsets.values()
                ):
                    offsets[label] = candidate
                    break
            else:
                raise RuntimeError("无法在外圈放下四个球，请扩大圆弧或减小最小球间距。")
        return offsets


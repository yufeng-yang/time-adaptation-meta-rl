# Continuous 1 — PointNav Belief Adaptation

这个目录用于离散三路线任务的第一版连续化实验。

## 推荐基础

第一阶段直接使用项目已有的 Safety-Gymnasium/MuJoCo Point 环境：
`tidyup_plan2/envs/sg_envs/point_gate_goal.py`。使用 Conda 环境 `sb3sg` 运行，
不修改该环境中的依赖。

已验证 `sb3sg` 可以正常导入 Safety-Gymnasium、MuJoCo 和 Stable-Baselines3；现有地图
可以完成 reset、连续 action step、`rgb_array` 离屏渲染和 `human` 窗口渲染。开门时
中央通道为空，关门时中央出现黄色碰撞门。

## 第一版任务定义

- 连续状态：位置 `(x, y)`、朝向 `theta`，可选加入线速度 `v`。
- 连续动作：线速度/加速度与角速度，例如 `[v_cmd, omega_cmd]`。
- 连续观测：局部 lidar 距离、目标相对方向与距离、`D-t`、门/障碍物 belief、trial 局数。
- 连续地图：保留直达、中路和绕路三个走廊；中央门改成可能出现的实体障碍物。
- 局部可见性：只有进入传感器范围后才能确认障碍物是否存在；起点不能直接看到答案。
- 一个 trial 包含 3 个 episode；地图、障碍物状态和 deadline 在 trial 内固定。
- 第 1 局负责探索，第 2、3 局利用 belief 调整路线。
- 奖励继续沿用当前语义：逐步/时间代价、到达目标奖励、碰撞或超时惩罚、危险区代价。

## 建议实验顺序

1. 在现有 MuJoCo Point 地图上标定三条路线的连续路径长度与三个 deadline。
2. 用三个 waypoint option（直达/中路/绕路）和高层三选一路线策略，复现离散成功结果。
3. 将固定 waypoint option 替换为分别训练并冻结的连续低层 controller。
4. 联合微调高层与低层，检查是否仍保持 belief 条件下的路线切换。
5. 最后再扩展为障碍物位置、尺寸或数量变化，并比较 exact Bayesian、VariBAD 等方法。

## 第一阶段暂不加入

- 不随机改变三条走廊的位置。
- 不同时加入多个未知障碍物。
- 不使用图像输入。
- 不直接训练端到端连续控制。

这些变化应在最简单的连续版本复现离散结论后逐项加入，否则无法判断失败来自 belief、
时间约束、感知还是低层控制。

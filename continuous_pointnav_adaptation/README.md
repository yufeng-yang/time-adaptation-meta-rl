# Continuous PointNav Adaptation

与 `tidyup_plan2` 同级的连续环境与后续连续实验目录。

## 当前环境

环境实现位于 `envs/point_gate_goal.py`，基于 Safety-Gymnasium/MuJoCo Point：

- 二维连续动作；
- 连续机器人状态与 lidar 观测；
- 固定起点和目标；
- 直达、上方较短绕行、下方较长绕行三条通路；
- 中央黄色门在每个 trial 内保持打开或关闭；
- 上路开口中央有一个不可碰撞的红色 danger 方块，覆盖上路必经截面；
- 每个 episode 第一次进入 danger 时直接扣除一次 `danger_penalty`（默认 `0.2`），
  停留或再次进入不重复扣分，并且不产生 Safety-Gymnasium cost；
- deadline 和 `D-t`；
- 支持 `rgb_array` 与 `human` 渲染。

使用 Conda 环境 `sb3sg`。该环境已经完成只读运行验证，不需要修改或安装依赖。

```bash
conda run -n sb3sg python continuous_pointnav_adaptation/envs/show.py
```

## 建议实验顺序

1. 标定三条连续路线的实际步数和三个 deadline。
2. 建立三个连续 waypoint option，并训练高层 belief 路线选择器。
3. 用训练并冻结的低层连续 controller 替换固定 waypoint option。
4. 联合微调高层和低层。
5. 再逐步加入障碍物位置、尺寸和数量变化。

## 当前奖励

- 每个物理步：`-0.001`
- 到达目标：`+1.0`
- 新的墙/门碰撞事件：`-0.05`
- 超时：`-1.0`
- 每局第一次经过上路 danger：`-0.2`
- Safety-Gymnasium cost 不参与训练，环境返回的 cost 固定为 `0.0`

固定 waypoint controller 的实际成功轨迹为：直达 183 步、上路 246 步、下路 266 步。
对应 deadline 设置为 `195 / 255 / 275`。

## 层次化训练

```bash
MUJOCO_GL=egl conda run -n sb3sg \
  python continuous_pointnav_adaptation/train_hierarchical.py
```

训练入口先在真实 MuJoCo 环境中分别执行 `2种门 × 3个deadline × 3条路线`，生成冻结
低层 option 的真实步数、reward、danger 与碰撞结果。高层 PPO 随后在该 SMDP option 表
上训练三局 belief 路线策略，避免反复仿真完全相同的冻结轨迹。结果保存在
`artifacts_hierarchical/`。

`tidyup_plan2/envs/sg_envs/` 中的原文件暂时保留，避免破坏已有导入与实验记录；
新的连续实验应从本目录继续开发。

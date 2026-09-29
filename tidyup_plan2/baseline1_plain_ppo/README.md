# Baseline 1 — Plain PPO

普通 PPO 对照组：策略输入不包含时间变量 `D-t`，也不包含 `h` 或门状态。

- 训练入口：`train.py`
- 可视化运行：`show.py`
- 最终模型：`artifacts/final_model.zip`
- 训练设定：固定 `D=500`、门关闭，用于验证普通 PPO 能否先学会到达目标。

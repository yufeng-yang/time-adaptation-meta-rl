# Baseline 2 — Time-Aware PPO

时间感知 PPO 对照组：在 Baseline 1 的观测基础上只加入剩余时间 `D-t`，不包含 `h` 或可见门状态。

- 训练入口：`train.py`
- 最终模型：`artifacts/final_model.zip`
- 行为克隆初始化模型：`artifacts/bc_model.zip`
- 目标策略：`D=15` 直达、`D=17` 危险中路、`D=19` 安全绕行。
- `D=15` 且门关闭时失败属于该不完美 baseline 的预期行为。

# Baseline 0.1 — Time-Only BC + PPO

与 Baseline 0 完全相同，策略只观察 `D-t`。区别是 PPO 前使用三条解析
专家路线做行为克隆：

- D=15，开门：direct
- D=17，关门：medium
- D=19，关门：detour

由于状态 `s` 被移除，相同的 `D-t` 可能对应不同位置、方向和专家动作，因而
BC 标签天然存在冲突。本实验保留这些冲突，并分别记录 BC 后与 PPO 后结果。

训练入口：`train.py`


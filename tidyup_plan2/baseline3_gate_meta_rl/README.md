# Baseline 3 — Time-conditioned Trial Meta-RL

三局 trial 的 meta-RL：策略显式观察当前剩余预算 `D-t`，真实门状态隐藏并在
trial 内固定。上下文 `h` 只从已经完成 episode 的
`(s,a,r_env,s',done)` 中推断。

- 三个 episode 使用完全相同的环境奖励、策略和终止规则；没有 probe reward
- 一个共享 actor 负责探索局和后续局，不使用人为分开的探索/执行策略头
- `h` 在 episode 内固定，只在 episode 边界更新
- 每个 trial 固定 3 个 episode，门开/关各 50%
- 目标 deadline 阶段每局独立均匀采样 `{15,17,19}`
- 训练采用 `500 → 100 → 50 → 25 → {15,17,19}` 路径发现 curriculum
- PPO 优化层级 trial return：每局内部使用 `gamma`，跨局使用 `beta_trial`
- 默认 `beta_trial=1`，即等权优化 `G1+G2+G3`
- 后两局回报进入第一局动作的 advantage，提供跨 episode 信用分配

主要训练入口：`train.py`。

`train_trial_ppo_legacy.py` 是旧的无时间、probe reward、final-only 对照，不再作为
推荐训练入口。

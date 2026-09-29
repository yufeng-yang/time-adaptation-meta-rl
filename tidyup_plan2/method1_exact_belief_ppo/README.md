# Method 1 — Exact Gate-Belief PPO

这个方法把门状态写成显式 Bayesian belief：

```text
b_t = P(gate=open | 当前 trial 到目前为止合法看到的局部观测)
```

- 每个 trial 包含 3 个 episode。
- 同一 trial 内地图、门状态和 deadline 固定。
- 新 trial 开始时 `b_0=0.5`。
- 门尚未进入局部视野时 belief 保持不变；观察到门后更新为 `0` 或 `1`。
- 策略输入为 baseline2 的局部状态与 `D-t`，再加 `b_t` 和 trial 剩余局数。
- 三个 episode 被拼成一个 Gymnasium episode，让 PPO 的 return 跨越 episode 边界。
- 训练沿用 baseline2 的三路线反向课程，最终进入随机门状态和随机 `D∈{15,17,19}` 的真实分布。
- Method1 使用 `7x7` 居中局部视野（只对本方法生效）。原 V4 的上路距离门 3 格、
  下路距离门 2 格，因此现在走两条非直达路线时都能观察门状态；路径长度、
  deadline 和奖励保持不变。为避免在公共分叉点提前泄露门状态，显式 gate sensor
  只在 agent 已进入任一分支（`x>=4`）后更新 belief。
- BC 只使用 baseline2 的三个 anchor 专家路径，并在 trial 的三个 episode 位置重复采样：
  `D=15/open→direct`、`D=17/closed→danger`、`D=19/closed→detour`。
  另外三个 `(gate, D)` 组合不提供专家轨迹。默认还会删去这三条专家轨迹在公共岔路口
  的 9 个选择标签：BC 只教每条走廊内部的导航，不直接规定应该选择哪条路线。
- 使用两次探索脉冲：反向课程第一次推进到公共岔路（stage 14）时软化策略输出一次，
  用于学习三个锚点岔路；进入随机门状态的最终任务分布时再次软化，并重新开始熵退火，
  专门用于探索 belief 条件下的路线切换。每次熵系数都从 `0.05` 在 12 万步内逐渐降回
  `0.01`。探索动作由 PPO 自己采样并按实际执行动作记录，因此另外三种行为仍由原始
  环境奖励学习。最终阶段的锚点 rehearsal 从 `30%` 降至 `10%`，避免持续压制新路线。

训练（新结果写入 `artifacts_route_exploration_two_pulse/`，之前结果均保留）：

```bash
python tidyup_plan2/method1_exact_belief_ppo/train.py
```

快速检查：

```bash
python tidyup_plan2/method1_exact_belief_ppo/train.py \
  --timesteps 256 --n-steps 128 --batch-size 64 --ppo-epochs 1 \
  --checkpoint-frequency 1000000
```

这里的 belief 更新是精确的，但 controller 仍由 PPO 学习，因此更准确的名称是
`exact-belief PPO`，不是通过动态规划求得的 exact Bayes-optimal policy。

## Hierarchical route-option version

`train_hierarchical.py` 将导航与路线决策分开：

- 低层是 baseline2 已知的三个导航 option：`direct / medium / detour`；
- 高层每个 base episode 只选择一次 option；
- 高层输入是 deadline one-hot、门的 exact belief 和 episode-index one-hot；
- 一个 trial 仍是三局，`gamma=1`，所以 PPO 直接优化三次路线选择的总回报；
- 门状态仍然只有进入路线并合法看到门之后才写入 belief。

训练结果写入 `artifacts_hierarchical/`：

```bash
python tidyup_plan2/method1_exact_belief_ppo/train_hierarchical.py
```

默认训练 50,000 个高层决策。这里低层 option 使用固定专家动作序列，是对“低层导航已经
由 BC 学会并冻结”的理想化实现；实验专门检验高层是否能从 belief 学会路线切换。

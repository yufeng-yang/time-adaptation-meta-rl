# Baseline 3 — 最终离散 Meta-RL

当前正式结果只保留 [0–64.5 万步](release_0_64p5w/README.md)，统计每个 trial 的
最后一局 return。旧纯 PPO、legacy 方案、重复绘图与完整中间实验目录已移出。

## 保留的代码

- `train_final.py`：统一续训入口，从保留的约 55 万步模型训练至约 64.5 万步。
- `train_decay_bc.py`：episode-level PPO + 退火 BC 的核心实现，也提供第一阶段从头训练。
- `ablation_stepwise_h/train.py`：step-level 的逐转移 h 更新、完整 trial 重放与训练。
- `continuation_h_comparison/train.py`：两版共享的模型/Adam 恢复、任务采样及续训实现。
- `continuation_h_comparison/render_*_checkpoint.py`：使用保留的模型渲染真实三局轨迹。
- `release_0_64p5w/`：曲线、模型、评估、截取日志、复现和结果包测试。
- `test_trial_protocol.py`、逐步版本及结果包的测试：验证 trial 协议和可复现性。

## 机制

每个 trial 三局，门和 deadline 固定。每局重置位置、计时，h 跨局继承；新 trial
才将 h 清零。episode-level 的 h 局内固定、局末更新；step-level 每步更新，
包含终止转移。两版均沿完整 trial 反向传播。

策略输入为局部 5×5 观测、方向和显式 D-t。门进入局部视野时可观察开关状态，
视野外不提供标签。三局环境奖励完全相同，不额外设置第一局 probe reward。
奖励为步进 −0.01、到达 goal +1、撞墙/门 −0.05、超时 −1，danger 每局首次进入扣 0.2。

每次更新开门/关门各两个 trial，顺序随机；对应 deadline 等概率抽样。
初始阶段为开/关 × D15/17/19。约 55 万步之后，episode-level 仅将关门 D19
换成 D25，BC 专家与评估同步替换；step-level 保持原任务。
BC 权重按累计步数在 60 万步归零，续训不会重启 BC 或课程。

## 常用入口

从仓库根目录执行，使用已准备好的依赖环境：

```bash
python tidyup_plan2/baseline3_gate_meta_rl/train_final.py --mode episode
python tidyup_plan2/baseline3_gate_meta_rl/train_final.py --mode step
python tidyup_plan2/baseline3_gate_meta_rl/release_0_64p5w/reproduce.py plot
python tidyup_plan2/baseline3_gate_meta_rl/release_0_64p5w/reproduce.py verify
python tidyup_plan2/baseline3_gate_meta_rl/continuation_h_comparison/render_episode_checkpoint.py
python tidyup_plan2/baseline3_gate_meta_rl/continuation_h_comparison/render_step_checkpoint.py
```

训练写入 `release_0_64p5w/runs/`，绘图和渲染写入 `generated/`；都不会覆盖保留的数据。
续训遇到非空输出目录会拒绝覆盖。依赖版本、完整结果及测试命令见结果目录 README。

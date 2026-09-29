# Exact-belief Bayesian RL

这个实现把隐藏门写成显式 belief：

```text
b_t = P(gate=open | policy 到目前为止合法看到的局部图像)
```

训练时环境会在每个 trial 开始时采样一次 open/closed，并在 5 个子 episode
中保持不变。策略不会读取真实 `gate_state`；filter 也只检查局部 `5x5` 图像。

策略输入是：

```text
25x6 semantic local map
+ 4-d direction one-hot
+ raw D-t
+ b_t
+ episode-boundary
+ trial 剩余 episode 数（包含当前局）
```

## 训练

从项目根目录运行：

```bash
python tidyup_plan2/main_2_BRL/train_brl.py
```

小规模 smoke test：

```bash
python tidyup_plan2/main_2_BRL/train_brl.py \
  --timesteps 256 --n-steps 128 --batch-size 64 --ppo-epochs 1 \
  --checkpoint-frequency 1000000
```

只评估已有模型：

```bash
python tidyup_plan2/main_2_BRL/train_brl.py --mode eval
```

## 从头分阶段训练

下面的入口默认随机初始化一个全新策略，依次执行：

1. closed-only：只采样 `D={17,19}`；
2. joint：均匀采样 open/closed 和 `D={15,17,19}`。

两个阶段的每个新 trial 都从 `belief=0.5` 开始。closed-only 只是固定环境中的
隐藏门，不会把 closed 直接输入 policy；只有 sensor 看见门后 belief 才变成 0。

```bash
python tidyup_plan2/main_2_BRL/train_staged_brl.py
```

所有新模型、日志与阶段评估写入：

```text
artifacts/staged/
```

只有明确希望续训时才传入旧模型：

```bash
python tidyup_plan2/main_2_BRL/train_staged_brl.py \
  --initial-model tidyup_plan2/main_2_BRL/artifacts/exact_belief_ppo.zip
```

训练时每个子 episode 独立从 `D={15,17,19}` 采样 deadline；门状态在整个
trial 中固定。评估时会分别测试六个固定的 `(gate, D)` 组合。

## 关键语义

外层环境把 5 个子 episode 拼成一个 Gymnasium episode。前 4 个子 episode
结束时不会向 PPO 返回 terminal，而是立即 reset 到同一个 gate 的下一局；只有
第 5 局结束时才返回 terminal。因此 PPO 优化的是整个 trial return，早期探测门
状态的成本可以由后续 episode 的收益补偿。

`info["gate_state"]` 仅供日志和评估使用。把它直接输入策略会变成 oracle，
不属于这里实现的 exact-belief Bayesian RL。

# 版本二：无 BC 的上层 PPO

从随机初始化开始，只训练 PPO；不加载旧权重、不采集训练示范、不做 BC 热身、
不使用 BC 辅助项。沿用两层局内 Transformer、64 维跨局 GRU、49 维观测、
九个上层动作和整个三局 trial 的回报信用分配。

任务保持 S=1，四个外球每 trial 随机排列 `{0, 1, 2, 4}`；三局共享布局、
价值和 deadline。预算仍为 30/65/150，固定速度理想下层，成功放回一个球结束一局。

默认与版本一匹配 200 次 PPO 更新、每批 32 个 trial、每 20 次更新验证，seed=0。
两组策略选择的动作不同，所以相同更新预算不保证相同的下层步数。
验证每个 deadline 64 个 trial，最终测试每个 deadline 128 个 trial，
测试使用与版本一相同的布局和分值种子；最佳检查点只依验证集选择。
纯 PPO 的训练与采样随机数过程不经过版本一的 BC 阶段。

```bash
/home/yufeng.yang/Downloads/enter/envs/time_adapt/bin/python -B \
  /home/yufeng.yang/codespace/time_adaptation/deadline_pick_place/high_level/v2_no_bc/train.py \
  --output-dir /home/yufeng.yang/codespace/time_adaptation/deadline_pick_place/high_level/runs/v2_no_bc_seed0
```

`initial.pt` 保存随机初始模型；`best.pt`、`latest.pt`、`final.pt` 保存训练检查点。
`config.json` 与 `test_results.json` 明确记录 BC 更新和训练示范数量为零。
历史启发式与 oracle 只用于训练完成后的独立基线评估，不参与训练。
共享训练器的 `--no-bc` 模式强制清零所有 BC 设置；版本一默认流程保留。

## seed 0 已完成的首轮结果

完成 200 次 PPO 更新，共 642395 个抽象下层步。15 项协议检查通过，
新增的无 BC 检查会拦截训练教师和交叉熵 BC 损失调用；保存记录确认它们均未用于训练。
与版本一的 384 个测试 trial 逐个核对，布局、分值和 deadline 完全相同。

验证集选出第 100 次 PPO 更新（326454 步）作为 `best.pt`。
其独立测试结果，每个 deadline 128 个 trial：

- 30 步：三局均分 1.00 → 1.00 → 1.00，总分 3.00。
- 65 步：1.539 → 2.242 → 2.305，总分 6.086。
- 150 步：1.734 → 1.734 → 1.734，总分 5.203。

上述检查点测试投放率均为 100%，主动检查动作均为 0；策略直接取球送回，
中 deadline 下已有部分跨局利用，但尚未学出期望的主动检查行为。
最终第 200 次更新的检查点总分为 3.00 / 6.383 / 6.375；它不替换按验证集选出的模型。

版本一的对应总分为 3.00 / 9.891 / 12.00，见本轮目录内 `comparison_v1.json`。
这是单种子、相同 PPO 更新数的初测，两组实际下层步数不同，
也不能据此断言纯 PPO 在更长训练预算下仍学不会。

曲线：`../runs/v2_no_bc_seed0/training_curves.png`，另存 PDF/SVG。
图中从随机模型初始验证开始，没有 BC 阶段；右下图展示 150 步的三局得分。
所有模型、日志、完整测试轨迹与绘图数据均已保存，当前没有训练进程继续运行。

## 延长到总计 1000 次更新

从首轮 `final.pt` 的第 200 次更新继续到总计 1000 次，不改变任务、奖励、
学习率或熵系数，不增加任何 BC。输出到 `runs/v2_no_bc_seed0_1000`，保留首轮目录。
恢复模型和 Adam 状态；旧检查点没有随机采样状态，因此续训采用新的可复现采样种子 200。
新检查点同时保存随机采样状态。断点恢复的单元检查验证，新格式保存/恢复后
与不间断训练得到逐参数完全相同的模型。

```bash
/home/yufeng.yang/Downloads/enter/envs/time_adapt/bin/python -B \
  /home/yufeng.yang/codespace/time_adaptation/deadline_pick_place/high_level/v2_no_bc/train.py \
  --ppo-updates 1000 \
  --resume-from /home/yufeng.yang/codespace/time_adaptation/deadline_pick_place/high_level/runs/v2_no_bc_seed0/final.pt \
  --output-dir /home/yufeng.yang/codespace/time_adaptation/deadline_pick_place/high_level/runs/v2_no_bc_seed0_1000
```

新增每次更新的策略熵（nats）、检查动作比例、上层动作数量和三局分别的训练得分。
保留前 200 次训练与验证记录，旧记录缺少的诊断量留空，不填零。
最佳模型继续按固定验证集选择，并保留上一轮验证最优检查点作为比较起点。

奖励规则保持：成功投放球得原始分值 / 4；检查、移动、未投放、超时均为 0。
无额外探索奖励，无每步惩罚，无超时扣分，无提前投放奖励。
放回 0 分球也结束本局。三局原始最高累计分数为 12，对应训练回报为 3；
gamma=1，前局的动作使用本次 trial 后续各局的累计奖励进行信用分配。
PPO 的熵正则系数为 0.01，它是优化目标的一部分，不是环境发放的奖励。

### 1000 次更新已完成

总计 2943521 个抽象下层步，本次新增 800 次更新、2301126 步。
16 项协议检查通过；恢复起点的 42 个模型张量和全部 Adam 状态与原第 200 次检查点逐项完全一致。
所有 1000 次更新的 BC 权重均为零。续训期间检查动作占全部上层动作的 0.104%。

固定验证集选出第 280 次更新（894529 步）的检查点。
独立测试每个 deadline 128 个 trial，三局均分：

- 30 步：1.00 → 1.00 → 1.00，总分 3.00。
- 65 步：1.656 → 2.359 → 3.117，总分 7.133。
- 150 步：1.656 → 2.359 → 3.117，总分 7.133。

相对首轮 200 次更新的验证最优模型，65 步总分从 6.086 提高到 7.133，
150 步从 5.203 提高到 7.133。最佳模型在独立测试中依旧没有主动检查动作，
中、宽预算的行为和得分相同，尚未学出利用宽预算进行局内检查的策略。
最终第 1000 次模型的总分为 2.922 / 5.586 / 7.133，存在后期退化，
因此交付的是按验证集选择的 `best.pt`，同时保留 `final.pt`。

本轮单种子结果支持跨局利用有所改善，不能据此推断纯 PPO 永远无法学会检查。
`extension_summary.json` 保存续训指标，`resume_audit.json` 保存恢复核对结果。
`training_curves.png` 包含前 200 次和续训后的完整曲线；
`training_diagnostics.png` 展示第 201–1000 次更新的策略熵及检查动作比例。
当前训练和独立测试均已完成。

## 渲染一个完整 trial

`../render_trial.py` 使用真实策略逐个选择上层动作，对理想执行器的每个移动步进行
俯视动画渲染，并与环境实际返回的末端位置、球位置、持球状态和耗时核对。
它不启动 MuJoCo 窗口，也不模拟机械臂关节、碰撞或真实抓取。
分值仅在接触后显示；右侧已知分值表是回放辅助信息，不是额外策略输入。
各局重置位置与时间，跨局 GRU 保留；接触和局间转换放慢播放以便观察。

```bash
/home/yufeng.yang/Downloads/enter/envs/time_adapt/bin/python -B \
  /home/yufeng.yang/codespace/time_adaptation/deadline_pick_place/high_level/render_trial.py \
  --deadline 150 --seed 300000
```

默认载入 `runs/v2_no_bc_seed0_1000/best.pt`；可用 `--checkpoint` 指定其他检查点。
默认输出到该检查点目录下的 `renders/trial_D150_seed300000/`，包含 MP4、GIF、
最终画面和 `trial.json`。示例选择固定种子 300000，未搜索高分轨迹：
第 1/2/3 局分别取回 B/C/A，得 2/0/4 分，各用 37 步，完整动画 20 秒。

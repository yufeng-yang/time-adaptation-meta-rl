# 上层策略可行性实验

新增 [版本二：完全无 BC 的 PPO](v2_no_bc/README.md)，从随机初始化训练，
单独保存到 `runs/v2_no_bc_seed0`。版本一的 BC + PPO 结果保留。

本目录用理想的固定速度笛卡尔执行器代替未来的冻结下层策略。
共享 `../radial_layout.py` 的布局参数，但不运行 MuJoCo、不检查关节可达性、
碰撞或真实抓取成功率，也没有训练下层控制器。

## 任务规则

- 中央盘子与夹爪出生点的水平位置重合。近球 S 半径 18 cm；外圈 A/B/C/D 半径 36 cm。
- S 恒为 1 分，作为已知保底。四个外圈球每个 trial 恰好包含 `{0, 1, 2, 4}` 各一次，
  随机排列到 A/B/C/D，外观不透露价值。每个 trial 都存在一个 4 分球，三局理论最高总分为 12。
  分值集合属于共同任务先验，已观察的分值可以帮助推断剩余分值。
- 每个 trial 三个 episode，共享布局、价值和 deadline。episode 重置位置与时间；trial 重置全部记忆。
- 只有接触球之后才给出球的身份及分值。下一次动作后的反馈覆盖前一次反馈，不提供历史价值表。
- 一次成功放回盘子立即结束 episode；放回 0 分球也结束。超时未放回得 0 分。
- 奖励仅为放回球的分值除以 4，不加探索奖励、运动惩罚或超时惩罚。

## 观测、动作和记忆

观测共 49 维：夹爪三维位置、盘子三维位置、五个球三维位置、剩余时间、
剩余回合比例、回合开始标记、接触反馈的球身份与分值、上一动作与奖励、
动作结果、持球状态。位置使用桌面中心为原点的坐标；位置、时间和分值已缩放。
球身份靠固定槽位区分，价值不包含在位置或外观输入中。

九个离散动作：检查 A/B/C/D；取回 S/A/B/C/D。检查包含移动、接触与等待；
取回包含移动、接触、抓取、返程与释放。取回动作内部不重新做上层决策。
每段移动按距离计时，超时时停在部分轨迹处，不能免费瞬移。

局内采用两层、四头、宽度 64 的因果 Transformer；局末注意力与最大池化摘要
经 GRUCell 更新 64 维跨局记忆。下一局初始观测清空接触反馈、上一动作和奖励，
上一局末的反馈只进入上一局的摘要。新 trial 将 GRU 清零。
这是沿用第一实验的记忆分工，模型在本实验中重新训练。

## 时间预算

抽象执行器速度为 3 cm/步；检查等待 2 步、抓取 2 步、释放 2 步。
直接取回近球为 25 步，外球为 37 步。
200 个布局的离线标定中，检查四个外球后再取回任意最佳候选需要 96–132 步
（逐个布局选最优检查顺序，并考虑最坏返程目标），因此初测使用 **30 / 65 / 150**。
该抽样标定不等于所有布局的严格界，也不等于真实机械臂控制耗时。

## 训练和复现

使用已安装 torch 的 `time_adapt` 环境；`time_adapt_robosuite` 用于物理场景。
无需 stable-baselines3。运行以下命令不会打开渲染窗口：

```bash
/home/yufeng.yang/Downloads/enter/envs/time_adapt/bin/python -B \
  /home/yufeng.yang/codespace/time_adaptation/deadline_pick_place/high_level/train.py \
  --output-dir /home/yufeng.yang/codespace/time_adaptation/deadline_pick_place/high_level/runs/my_run \
  --expert-trials 768 --bc-updates 1000 --ppo-updates 200 \
  --batch-trials 32 --eval-trials 64 --eval-every 20
```

先从只读取合法接触历史的启发式教师收集示范，做行为克隆热身，再做 PPO。
PPO 前半程含逐渐归零的行为克隆辅助项；没有可行动作掩码。
每个 trial 的三局合计回报用于信用分配，折扣为 1。
这是 BC + PPO 可行性实验，不能称为纯 PPO 从零训练。

验证集与最终测试集使用不同种子。三个 deadline 使用配对的布局和价值样本。
`best.pt` 只按验证集选择；测试每个 deadline 128 个 trial。
输出包含配置、源码快照、检查点、验证曲线、测试指标和分 deadline 的行为示例。
新运行还保存完整的 `test_trials.json`，包含每个测试 trial 的五球分值、位置和三局动作，
并在 `test_results.json` 的 `stage_tests` 中分别报告 BC 阶段和最终 PPO 检查点成绩。
基线含固定取近球、随机、合法历史启发式、完全知道分值的 oracle。
推理时清零 GRU 仅为诊断；它没有重新训练，不能代替正式消融实验。

单独复评并打印每局动作：

```bash
/home/yufeng.yang/Downloads/enter/envs/time_adapt/bin/python -B \
  /home/yufeng.yang/codespace/time_adaptation/deadline_pick_place/high_level/evaluate.py \
  --checkpoint /home/yufeng.yang/codespace/time_adaptation/deadline_pick_place/high_level/runs/fixed_score_set_seed0/best.pt
```

协议检查：在此目录运行同一 Python 的 `-B -m unittest test_protocol.py`。
覆盖隐藏价值输入隔离、末步投放、超时、单球终止、三局重置、
末态反馈编码、Transformer 因果性和 PPO 回放概率一致性。

正式实验仍需冻结真实下层后重新标定耗时、多个训练种子、独立训练的记忆与时间输入消融。

## 旧版独立抽分实验的结论（历史结果）

`runs/feasibility_clean_boundary_seed0` 使用外球独立抽分、允许重复的旧任务分布，
不能与固定 `{0, 1, 2, 4}` 集合的新实验直接比较原始均分。
它完成 768 个示范 trial、1000 次 BC 更新、
200 次 PPO 更新（PPO 共 793591 个抽象下层步）。验证集选择的最佳检查点
来自 BC 第 1000 次更新；这轮 PPO 没有超过它。测试结果不能用于宣称
PPO 已经自主学出探索行为。

每个 deadline 测试 128 个 trial。平均每局原始分数：

- 30 步：三局均为 1.00；直接取近球。
- 65 步：2.20 → 2.92 → 3.06；第一、二局投放率为 91.4%、92.2%，第三局 99.2%。
- 150 步：3.21 → 3.20 → 3.20；第一局平均检查 2.77 个外球，后两局无需额外检查，投放率均为 100%。

65 步下每 trial 总分为 8.18，合法历史启发式为 8.67，完全信息 oracle 为 9.66。
推理时跨局清零 GRU 后同一测试集总分降至 6.35，但正式结论还需要独立训练的消融。
当前结果支持结构和任务的初步可行性，也显示中 deadline 的投放可靠性仍需改善。

## 固定分值集合的重新训练

`runs/fixed_score_set_seed0` 从相同随机种子重新初始化，采用 768 个示范 trial、
1000 次 BC 更新、200 次 PPO 更新。PPO 的 BC 辅助项从 0.2 线性衰减，
第 100 次更新起为 0，后 100 次更新完全使用 PPO 目标。
布局、deadline、执行器、模型、奖励和其余超参数沿用上一轮。
最终测试集和验证集分离；测试每个 deadline 128 个 trial。
本轮单独保存到新目录，旧结果与旧任务分布保留。
旧 checkpoint 的 `evaluate.py` 复评自动使用旧的独立抽分模式。

本轮已完成，PPO 共 775917 个抽象下层步；14 项协议检查通过。
所有 384 个测试 trial 均核对外球分值集合为 `{0, 1, 2, 4}`，三局内不变。
验证集选出 PPO 第 40 次更新作为最佳检查点，但独立测试并未显示整体稳定超过 BC。
最佳检查点的测试结果：

- 30 步：三局平均均为 1.00 分，均 25 步成功投放。
- 65 步：三局均分 2.430 → 3.492 → 3.969，总分 9.891；
  投放率分别为 89.1%、92.2%、100%。
- 150 步：128 个 trial 均得到 4 → 4 → 4，总分 12；
  第一局平均检查 2.586 次、用时 75.758 步，后两局均直接取回、各 37 步。
  第一局最长 143 步；该样本存在一次重复检查，说明策略仍有冗余行为。

同一独立测试集，BC 检查点在 65/150 下总分为 10.023/11.969；
最终 PPO 检查点为 9.773/11.938。因此仍需区分“验证集选到 PPO 检查点”
与“PPO 稳定提高泛化表现”，当前仅支持前者。
本轮不是纯 PPO 从零训练；结果也不能与旧独立抽分分布直接比较原始均分。

已导出本轮训练图 `runs/fixed_score_set_seed0/training_curves.png`（另有 PDF/SVG）。
图中区分训练采样回报、固定验证集回报、65 步的跨局得分以及 BC 损失。
PPO 横轴从 BC 训练结束后开始，单位为抽象下层步；BC 损失使用单独的梯度更新横轴。
`validation_curves.csv` 与 `bc_loss.csv` 保存绘图数据；原始训练数据为 `training.csv`。
复现：使用同一 Python 运行 `plot_training.py --run-dir runs/fixed_score_set_seed0`
（从本目录执行）。绘图只读取已有日志，不重新训练。

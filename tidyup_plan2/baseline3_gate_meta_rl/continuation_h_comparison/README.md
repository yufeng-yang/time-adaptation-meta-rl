# 约55万步到64.5万步：续训与渲染

## 当前追加到80万步的训练

两版已从保留末端继续训练到累计800000步，不从头训练，也不重启BC：

- 蓝色 episode-level：从 `episode_645145.pt` 恢复，输出 `runs_80w/episode/`。
  开门 D15/17/19，关门 D15/17/25。
- 橙色 step-level：从 `step_643351.pt` 恢复，输出 `runs_80w/step/`。
  开关门均为 D15/17/19。

两版均恢复模型、Adam、累计计数和检查点RNG；gamma=0.99、原ASAP奖励、
每trial三局且门和deadline固定，h保留，各自原有更新方式不变。
BC权重已为0，课程已结束。每5000步评测并保存检查点，完成当前更新后停止，
所以实际总步数可能略超800000。原0–64.5万步结果包不会覆盖。
查看各输出目录的 `progress.json`；完成以 `summary.json` / `final_model.pt` 为准。

以下是原0–64.5万步整理版的说明。

这里只保留共用续训实现和两种结构的真实轨迹渲染。旧 0–100 万步绘图脚本、重复测试、
完整中间目录和超出当前展示范围的结果已从活动目录移走。

`train.py` 恢复保留模型的网络、Adam 和累计计数；默认累计目标645000步。
初始模型在 `../release_0_64p5w/models/`。episode-level 默认第六种任务为关门 D25，
step-level 保持关门 D19；其余任务、trial 协议、PPO 参数及 BC 退火不变。
推荐使用父目录的 `train_final.py --mode episode|step`，该入口固定当前整理版参数。

渲染默认使用结果包末端模型，输出到 `../release_0_64p5w/generated/`：

```bash
python tidyup_plan2/baseline3_gate_meta_rl/continuation_h_comparison/render_episode_checkpoint.py
python tidyup_plan2/baseline3_gate_meta_rl/continuation_h_comparison/render_step_checkpoint.py
```

episode-level 默认只渲染关门 D25 的三局；step-level 默认渲染原六条件的完整三局。
可以指定保留的55万步模型，例如：

```bash
python tidyup_plan2/baseline3_gate_meta_rl/continuation_h_comparison/render_episode_checkpoint.py --checkpoint tidyup_plan2/baseline3_gate_meta_rl/release_0_64p5w/models/episode_551179.pt --deadline 19
```

图中的全地图仅供观看，不会加入策略输入；右侧显示实际局部语义输入。
局间保持原模型的 h 更新方式，并尽可能与结果包的逐局评估对照。
完整曲线由 `../release_0_64p5w/reproduce.py plot` 生成，不再依赖旧训练目录。

# 约55万步到64.5万步：续训与渲染

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

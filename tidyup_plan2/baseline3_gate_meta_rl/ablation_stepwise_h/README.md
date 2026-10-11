# Step-level h

本目录只保留最终逐步记忆更新的实现及测试。每个真实转移后更新 h，包括终止转移；
跨 episode 继承，只有新 trial 清零。PPO 与 BC 顺序重放完整 trial，不在局间 detach。

使用父目录的统一入口从保留的约 55 万步模型续训：

```bash
python tidyup_plan2/baseline3_gate_meta_rl/train_final.py --mode step
```

`train.py` 同时提供最初约 55 万步的从头训练实现，默认输出
`../release_0_64p5w/runs/step_from_scratch/`。
原完整训练目录、重复评估入口已移出；需要复测时使用
`../release_0_64p5w/reproduce.py verify`，不要依赖旧默认路径。

协议测试：

```bash
python -m unittest tidyup_plan2.baseline3_gate_meta_rl.ablation_stepwise_h.test_protocol
```

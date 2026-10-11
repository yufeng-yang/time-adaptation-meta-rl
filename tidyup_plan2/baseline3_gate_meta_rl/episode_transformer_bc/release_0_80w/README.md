# Selected Transformer result: 0–80w

The selected endpoint is **800,058 environment steps**, nominally 800k. This is
a retrospectively selected checkpoint: the original run completed 1,000,050
steps. The raw run remains unchanged in `../runs/seed0_1m/`; nothing was deleted.

- `final_model.pt`: exact selected checkpoint, including Adam/counters/RNG state.
- `initial_model.pt`: the original random initialization, not pretrained weights.
- `train_trials.csv`, `evaluation_curve.csv`, `evaluations/`: retained only through
  800,058 steps. Historical fluctuations before that point are not removed.
- `latest_evaluation.json`, `summary.json`: selected endpoint, not the 1M model.
- `renders/`: all six cases, all three episodes; matches all 54 saved evaluation
  records; rendered actions independently match the training collector.
- `comparison/`: selected Transformer curve through 80w, compared with the original
  episode-/step-level curves through 64.5w. No smoothing or extrapolation.
- `config.json`: settings for a new 800k-budget reproduction, using a fresh output
  directory. `original_training_config.json` preserves the actual 1M run config.
  The checkpoint itself also preserves the original training config unchanged.
- `selection.json`, `manifest.json`: selection provenance and file checksums.

At the selected endpoint, episode-3 mean return over six tasks is **0.448333**.
Open D15/17/25: direct, 14 steps, .86. Closed D17: dangerous route, 16 steps, .64.
Closed D25: safe detour, 18 steps, .82. Closed D15 is infeasible; the later episodes
take the dangerous route and time out, −1.35 (the safer timeout would be −1.15).

Thus all feasible tasks select their optimal episode-3 routes at this checkpoint;
this does not assert that every evaluation before 80w was stable or optimal.
Both gate state and D are fixed across three episodes. Episode-level h is inherited;
Transformer has no positional encoding; remaining time is (D−t)/25, unclipped.

Regenerate the comparison from the project root:

```bash
/home/yufeng.yang/Downloads/enter/envs/sb3sg/bin/python -m \
  tidyup_plan2.baseline3_gate_meta_rl.episode_transformer_bc.compare_returns \
  --run tidyup_plan2/baseline3_gate_meta_rl/episode_transformer_bc/release_0_80w
```

Start a new 800k-budget reproduction (does not resume pretrained weights):

```bash
/home/yufeng.yang/Downloads/enter/envs/sb3sg/bin/python -u \
  tidyup_plan2/baseline3_gate_meta_rl/episode_transformer_bc/train.py \
  --total-steps 800000 \
  --output-dir tidyup_plan2/baseline3_gate_meta_rl/episode_transformer_bc/runs/seed0_80w
```

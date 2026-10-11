# Episode-level Transformer Meta-RL + annealed BC

This folder holds the original gamma=0.99 Transformer experiment, with
per-step costs retained in the return (no terminal refund). Its training
entry, tests, models, curves and renders are kept together here. The previous
pooled-MLP release is unchanged.

**Current active version:** this original ASAP-trained model is retained.
An extra post-hoc deadline score removes step costs and temporal discount only
for reporting, without changing rewards received by the model or its h.
See [extra evaluation](README_deadline_score.md); generated curves and fresh
80w evaluation are in `extra_evaluation_0_80w/`.

**Selected result:** `release_0_80w/`, ending at 800,058 steps. Models, logs,
evaluations, six-case GIFs and comparison curves are retained through that point.
The full raw 1M run is preserved under ignored `runs/seed0_1m/`, not used as the
selected endpoint. See the release README and selection provenance for details.

**Inactive comparison experiment:** the gamma=1 terminal-refund version lives in
`../episode_transformer_deadline_reach/`, with its own `train.py`, reward,
tests and output directory. Its failed 80w run is preserved for reference,
but it is not the active version. No new-version entry or results are stored here.
The architecture and PPO engine are reused, but the old defaults and trained
checkpoint configurations keep their original reward behavior.

- Six tasks: open/closed × D={15,17,25}; D=19 is replaced for both gate states.
- Three episodes per trial; gate and deadline fixed; h inherited across episodes.
- Transition MLP (64 dimensions) → two Transformer Encoder layers (four heads,
  feedforward width 128) → attention + max pooling → episode MLP → cross-episode
  GRU (32 dimensions). The actor/critic receives [o_t; (D-t)/25; h].
- No position embeddings, extra time embeddings, or causal mask. No dropout,
  so PPO rollout and replay use the same deterministic memory computation.
  Episodes are processed individually without padding. The completed-episode
  summary is never used by the controller in that same episode.
- The remaining-time feature is (D−t)/25 without clipping (previously
  min(D−t,20)/20). This is a fixed unit conversion, not a per-deadline ratio;
  larger future deadlines are not saturated. No extra time encoding is added.
- PPO and full-trial BC run together; λ decreases linearly from 1 to 0 at 600k
  environment steps. No pure-BC warmup. Same trial credit assignment/rewards as
  the retained episode-level experiment.
- No deadline curriculum: uniform {15,17,25} from the first trial. Gates balanced
  per four-trial update, deadlines sampled once independently per trial. Train
  to 1,000,000 steps, finishing the current update (at most one update overshoot).
- Six demonstration trials only. Episode 1 approaches until the door is visible;
  if open, continues direct; if closed, returns to the fork and safely detours.
  Closed D=25: 22/18/18 steps, returns .78/.82/.82, all successful. Closed D=17:
  first episode safe timeout, later episodes dangerous route (16 steps, .64).
  Closed D=15 remains infeasible. Open tasks use direct (14 steps, .86).

Run from the project root with the existing environment:

```bash
/home/yufeng.yang/Downloads/enter/envs/sb3sg/bin/python -u \
  tidyup_plan2/baseline3_gate_meta_rl/episode_transformer_bc/train.py
```

Default output: `runs/seed0_1m/`. Each 5k steps saves all three episodes of all six
tasks (three seeds per case), a curve CSV, and a latest checkpoint. Separate
checkpoints are saved every 25k. Initial and final weights, Adam state, RNG state,
expert actions, configuration and source hashes are saved. Nonempty output
directories are rejected. Generated runs are ignored by Git.

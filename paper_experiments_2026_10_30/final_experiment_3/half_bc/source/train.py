"""Feasibility run: legal-history demonstrations, then trial-return PPO.

Run with the existing time_adapt Python environment. No MuJoCo or low-level
policy is trained. Each trial contains three episodes and one hidden score map.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from env import ACTIONS, N_ACTIONS, OBS_DIM, PickPlaceTrial, TaskConfig, calibration
from model import TransformerMetaPolicy


class HistoryPlanner:
    """Demonstration / baseline planner. Reads only legal contact feedback."""
    def __init__(self, safe_score=1.):
        self.known = np.array([safe_score, np.nan, np.nan, np.nan, np.nan])

    def observe(self, env, observation):
        marker = observation[24:29]
        if marker.sum():
            self.known[int(marker.argmax())] = float(observation[29] * env.config.score_scale)

    def choose(self, env, observation):
        self.observe(env, observation)
        feasible_known = [i for i in range(5) if np.isfinite(self.known[i]) and env.duration(4 + i) <= env.remaining]
        best = max(feasible_known, key=lambda i: (self.known[i], -env.duration(4 + i))) if feasible_known else None
        if best is not None and self.known[best] >= max(env.config.unknown_scores):
            return 4 + best
        candidates = []
        for i in range(1, 5):
            if np.isfinite(self.known[i]):
                continue
            cost = env.duration(i - 1)
            pos = env.initial_balls[i] + [0, 0, .005]
            fallback = min(
                (env.duration(4 + j, pos) for j in range(5) if np.isfinite(self.known[j]) and self.known[j] >= env.config.safe_score),
                default=10**6,
            )
            # Explore only while there is time to return a candidate or fallback.
            if cost + max(fallback, env.duration(4 + i, pos)) <= env.remaining:
                candidates.append((cost, i))
        if candidates:
            return min(candidates)[1] - 1
        choices = []
        for i in range(5):
            if env.duration(4 + i) > env.remaining:
                continue
            value = self.known[i] if np.isfinite(self.known[i]) else np.mean(env.config.unknown_scores)
            choices.append((value, -env.duration(4 + i), i))
        if choices:
            return 4 + max(choices)[2]
        # A final observation may still help later episodes, even if this one times out.
        unknown = [i for i in range(1, 5) if not np.isfinite(self.known[i]) and env.duration(i - 1) <= env.remaining]
        if unknown and env.episode + 1 < env.config.episodes_per_trial:
            return min(unknown, key=lambda i: env.duration(i - 1)) - 1
        return 4 + (best if best is not None else 0)


@torch.no_grad()
def collect_trial(model, env, deadline=None, *, deterministic=False, teacher=False,
                  baseline=None, reset_meta=False):
    obs = env.reset(deadline=deadline)
    hidden = model.initial_hidden() if model is not None else None
    planner = HistoryPlanner(env.config.safe_score) if teacher else None
    episodes = []
    current = dict(obs=[obs], actions=[], rewards=[], log_probs=[], values=[], durations=[])
    while True:
        if model is not None:
            tensor = torch.from_numpy(np.array(current['obs']))[None]
            logits, values, _ = model.forward_episode(tensor, torch.tensor([tensor.shape[1]]), hidden)
            dist = Categorical(logits=logits[0, -1])
            action = int(logits[0, -1].argmax()) if deterministic else int(dist.sample())
            log_prob, value = float(dist.log_prob(torch.tensor(action))), float(values[0, -1])
        else:
            log_prob, value = 0., 0.
            action = 0
        if planner is not None:
            action = planner.choose(env, obs)
        elif baseline == 'safe':
            action = 4
        elif baseline == 'oracle':
            feasible = [i for i in range(5) if env.duration(4 + i) <= env.remaining]
            action = 4 + max(feasible, key=lambda i: env._scores[i]) if feasible else 4
        elif baseline == 'random':
            action = int(env.rng.integers(N_ACTIONS))
        obs, reward, done, info = env.step(action)
        current['actions'].append(action)
        current['rewards'].append(reward)
        current['log_probs'].append(log_prob)
        current['values'].append(value)
        current['durations'].append(info['duration'])
        current['obs'].append(info.get('terminal_observation', obs))
        if info['episode_done']:
            if planner is not None:
                planner.observe(env, info['terminal_observation'])
            if model is not None:
                tensor = torch.from_numpy(np.array(current['obs']))[None]
                _, _, hidden = model.forward_episode(tensor, torch.tensor([tensor.shape[1]]), hidden)
                if reset_meta:
                    hidden = model.initial_hidden()
            episodes.append(current)
            current = dict(obs=[obs], actions=[], rewards=[], log_probs=[], values=[], durations=[])
        if done:
            break
    # gamma=1 and full-trial Monte Carlo returns: earlier exploration receives
    # credit for later episodes. Deadlines account for all movement durations.
    future = 0.
    for episode in reversed(episodes):
        targets = []
        for reward in reversed(episode['rewards']):
            future = reward + future
            targets.append(future)
        episode['returns'] = targets[::-1]
        episode['advantages'] = np.array(episode['returns']) - np.array(episode['values'])
    return dict(episodes=episodes, records=env.records, low_steps=env.total_low_steps,
                scores=env._scores.tolist(), positions=env.initial_balls.tolist(), deadline=env.deadline)


def replay(model, trials):
    hidden = model.initial_hidden(len(trials))
    output = {key: [] for key in ('logits', 'values', 'actions', 'old_log_probs', 'returns', 'advantages')}
    for index in range(3):
        episodes = [trial['episodes'][index] for trial in trials]
        lengths = torch.tensor([len(e['obs']) for e in episodes])
        longest = int(lengths.max())
        observations = torch.zeros(len(trials), longest, OBS_DIM)
        actions = torch.zeros(len(trials), longest - 1, dtype=torch.long)
        arrays = {key: torch.zeros(len(trials), longest - 1) for key in ('old_log_probs', 'returns', 'advantages')}
        for row, episode in enumerate(episodes):
            n = len(episode['actions'])
            observations[row, :n + 1] = torch.from_numpy(np.array(episode['obs']))
            actions[row, :n] = torch.tensor(episode['actions'])
            arrays['old_log_probs'][row, :n] = torch.tensor(episode['log_probs'])
            arrays['returns'][row, :n] = torch.tensor(episode['returns'])
            arrays['advantages'][row, :n] = torch.tensor(episode['advantages'])
        logits, values, hidden = model.forward_episode(observations, lengths, hidden)
        valid = torch.arange(longest - 1)[None] < (lengths - 1)[:, None]
        output['logits'].append(logits[:, :-1][valid])
        output['values'].append(values[:, :-1][valid])
        output['actions'].append(actions[valid])
        for key, array in arrays.items():
            output[key].append(array[valid])
    return {key: torch.cat(value) for key, value in output.items()}


def summarize(trials):
    result = {}
    for deadline in sorted({trial['deadline'] for trial in trials}):
        selected = [trial for trial in trials if trial['deadline'] == deadline]
        result[str(deadline)] = dict(
            mean_trial_score=float(np.mean([sum(r['score'] for r in t['records']) for t in selected])),
            episodes={str(index + 1): dict(
                score=float(np.mean([t['records'][index]['score'] for t in selected])),
                delivery_rate=float(np.mean([t['records'][index]['delivered'] for t in selected])),
                low_steps=float(np.mean([t['records'][index]['low_steps'] for t in selected])),
                inspections=float(np.mean([len(t['records'][index]['inspections']) for t in selected])),
            ) for index in range(3)},
        )
    return result


def evaluate(model, config, count, seed, reset_meta=False):
    trials = [collect_trial(model, PickPlaceTrial(config, seed + i), d, deterministic=True, reset_meta=reset_meta)
              for d in config.deadlines for i in range(count)]
    return summarize(trials), trials


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--expert-trials', type=int, default=384)
    parser.add_argument('--bc-updates', type=int, default=250)
    parser.add_argument('--ppo-updates', type=int, default=120)
    parser.add_argument('--batch-trials', type=int, default=32)
    parser.add_argument('--eval-trials', type=int, default=32)
    parser.add_argument('--eval-every', type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output = args.output_dir or Path(__file__).resolve().parent / 'runs' / datetime.now().strftime('feasibility_%Y%m%d_%H%M%S')
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Output directory is not empty; refusing to overwrite.')
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / 'source'
    source_dir.mkdir()
    for filename in ('env.py', 'model.py', 'train.py', 'evaluate.py', 'test_protocol.py'):
        shutil.copy2(Path(__file__).resolve().parent / filename, source_dir / filename)
    shutil.copy2(Path(__file__).resolve().parents[1] / 'radial_layout.py', source_dir / 'radial_layout.py')
    config = TaskConfig()
    calibration_result = calibration(200)
    save_json(output / 'calibration.json', calibration_result)
    model = TransformerMetaPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_json(output / 'config.json', {
        **vars(args), 'output_dir': str(output), 'task': PickPlaceTrial(config).config_dict(),
        'architecture': 'causal within-episode Transformer + episode-boundary GRU',
        'observation_dim': OBS_DIM, 'actions': ACTIONS, 'torch': torch.__version__,
        'training': 'legal-history demonstration warmstart + full-trial PPO; no action masks',
        'environment': 'ideal fixed-speed Cartesian options; not MuJoCo execution',
        'episode_boundary': 'terminal event encoded before reset; next episode starts with clean feedback/action/reward fields',
    })
    print('Output:', output, flush=True)
    print('Calibration:', calibration_result, flush=True)
    experts = [collect_trial(None, PickPlaceTrial(config, 100000 + i), config.deadlines[i % 3], teacher=True)
               for i in range(args.expert_trials)]
    save_json(output / 'expert_summary.json', summarize(experts))
    total_low_steps = 0
    start = time.monotonic()
    progress = []

    def checkpoint(name, phase, update):
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        config=model_config(), phase=phase, update=update,
                        low_steps=total_low_steps), output / name)

    def model_config():
        return dict(seed=args.seed, task=PickPlaceTrial(config).config_dict())

    for update in range(args.bc_updates):
        batch = [experts[int(i)] for i in rng.integers(len(experts), size=args.batch_trials)]
        out = replay(model, batch)
        loss = F.cross_entropy(out['logits'], out['actions']) + .1 * F.mse_loss(out['values'], out['returns'])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
        optimizer.step()
        if (update + 1) % 50 == 0:
            print(f'BC {update+1}/{args.bc_updates}: loss={float(loss.detach()):.4f}, elapsed={time.monotonic()-start:.1f}s', flush=True)
    checkpoint('bc.pt', 'bc', args.bc_updates)
    initial_summary, _ = evaluate(model, config, args.eval_trials, 200000)
    save_json(output / 'after_bc_validation.json', initial_summary)
    print('After BC:', initial_summary, flush=True)
    best_metric = float(np.mean([v['mean_trial_score'] for v in initial_summary.values()]))
    checkpoint('best.pt', 'bc', args.bc_updates)
    for group in optimizer.param_groups:
        group['lr'] = 3e-4

    for update in range(args.ppo_updates):
        trials = [collect_trial(model, PickPlaceTrial(config, int(rng.integers(1000000000))))
                  for _ in range(args.batch_trials)]
        total_low_steps += sum(t['low_steps'] for t in trials)
        all_adv = np.concatenate([e['advantages'] for t in trials for e in t['episodes']])
        mean, std = all_adv.mean(), all_adv.std() + 1e-8
        for trial in trials:
            for episode in trial['episodes']:
                episode['advantages'] = (episode['advantages'] - mean) / std
        for _ in range(4):
            order = rng.permutation(len(trials))
            for indices in np.array_split(order, 2):
                out = replay(model, [trials[int(i)] for i in indices])
                dist = Categorical(logits=out['logits'])
                logp = dist.log_prob(out['actions'])
                ratio = (logp - out['old_log_probs']).exp()
                actor_loss = -torch.minimum(ratio * out['advantages'], ratio.clamp(.8, 1.2) * out['advantages']).mean()
                critic_loss = F.mse_loss(out['values'], out['returns'])
                loss = actor_loss + .5 * critic_loss - .01 * dist.entropy().mean()
                # Same legal-history auxiliary as the previous experiment; anneals to zero.
                bc_weight = .2 * max(0., 1 - (update + 1) / max(1, args.ppo_updates * .5))
                if bc_weight:
                    demos = [experts[int(i)] for i in rng.integers(len(experts), size=max(4, len(indices) // 2))]
                    demo_out = replay(model, demos)
                    loss += bc_weight * F.cross_entropy(demo_out['logits'], demo_out['actions'])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                optimizer.step()
        row = dict(update=update + 1, low_steps=total_low_steps, elapsed_seconds=time.monotonic() - start,
                   train_mean_trial_score=float(np.mean([sum(r['score'] for r in t['records']) for t in trials])))
        progress.append(row)
        if (update + 1) % args.eval_every == 0 or update + 1 == args.ppo_updates:
            summary, _ = evaluate(model, config, args.eval_trials, 200000)
            save_json(output / f'validation_{update+1:04d}.json', summary)
            metric = float(np.mean([v['mean_trial_score'] for v in summary.values()]))
            if metric > best_metric:
                best_metric = metric
                checkpoint('best.pt', 'ppo', update + 1)
            print('PPO:', row, 'validation:', summary, flush=True)
            save_json(output / 'progress.json', progress)
            checkpoint('latest.pt', 'ppo', update + 1)
    checkpoint('final.pt', 'ppo', args.ppo_updates)
    selected_checkpoint = torch.load(output / 'best.pt', weights_only=False)
    model.load_state_dict(selected_checkpoint['model'])
    test_summary, tests = evaluate(model, config, args.eval_trials * 2, 300000)
    reset_summary, _ = evaluate(model, config, args.eval_trials * 2, 300000, reset_meta=True)
    stage_tests = {}
    for stage in ('bc', 'final'):
        stage_checkpoint = torch.load(output / f'{stage}.pt', weights_only=False)
        model.load_state_dict(stage_checkpoint['model'])
        stage_tests[stage], _ = evaluate(model, config, args.eval_trials * 2, 300000)
    baselines = {}
    for name in ('safe', 'random', 'history_planner', 'oracle'):
        trials = [collect_trial(None, PickPlaceTrial(config, 300000 + i), d,
                                teacher=name == 'history_planner', baseline=name)
                  for d in config.deadlines for i in range(args.eval_trials * 2)]
        baselines[name] = summarize(trials)
    result = dict(
        selected_checkpoint={key: selected_checkpoint[key] for key in ('phase', 'update', 'low_steps')},
        stage_tests=stage_tests,
        policy=test_summary, inference_time_meta_memory_reset=reset_summary,
        baselines=baselines, ppo_low_steps=total_low_steps,
        elapsed_seconds=time.monotonic()-start,
        caveat='Single-seed feasibility run with legal-history BC warmstart; ideal lower layer; memory reset is an inference diagnostic, not a separately trained ablation.',
    )
    save_json(output / 'test_results.json', result)
    save_json(output / 'test_trials.json', [
        {k: trial[k] for k in ('records', 'scores', 'positions', 'deadline')} for trial in tests
    ])
    save_json(output / 'test_examples.json', [
        {k: trial[k] for k in ('records', 'scores', 'positions', 'deadline')}
        for deadline in config.deadlines for trial in [t for t in tests if t['deadline'] == deadline][:2]
    ])
    with (output / 'training.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(progress[0])) if progress else None
        if writer:
            writer.writeheader()
            writer.writerows(progress)
    print('FINAL:', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

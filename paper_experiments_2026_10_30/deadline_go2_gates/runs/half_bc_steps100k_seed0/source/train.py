"""Half-BC: history-only demonstration warmstart, then full-trial PPO."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from layout import Config, PATTERNS
from environment import GateTrial, STATE_DIM
from model import MetaPolicy, log_probability
from teacher import HistoryController


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def bc_coefficient(steps, decay_steps, initial=1.):
    """Count active high-level transitions, excluding expert/evaluation steps."""
    if steps < 0 or decay_steps <= 0 or initial < 0:
        raise ValueError('Expected steps>=0, decay_steps>0, initial>=0')
    return initial * max(0., 1. - steps / decay_steps)


_WORLDS = {}


def make_env_batch(config, seeds):
    # Reuse the compiled physics scene; every trial is explicitly reset below.
    from simulation import GateWorld, Backend
    key = (config, len(seeds))
    if key not in _WORLDS:
        _WORLDS[key] = GateWorld(len(seeds), config)
    world = _WORLDS[key]
    return [GateTrial(Backend(world, i, seed), config, seed) for i, seed in enumerate(seeds)]


def advance_batch(envs, actions, active):
    from simulation import step_batch
    return step_batch(envs, actions, active)


def next_episodes(envs):
    return [env.next_episode() for env in envs]


def empty_episode(observation):
    return dict(obs=[observation], actions=[], raw_actions=[], log_probs=[], values=[], rewards=[])


def finish_returns(trials):
    for trial in trials:
        future = 0.
        for episode in reversed(trial['episodes']):
            targets = []
            for reward in reversed(episode['rewards']):
                future += reward
                targets.append(future)
            episode['returns'] = targets[::-1]
            episode['advantages'] = np.array(episode['returns']) - np.array(episode['values'])
    return trials


def teacher_batch(envs, deadlines, patterns):
    observations = [e.reset(d, p) for e, d, p in zip(envs, deadlines, patterns)]
    planners = [HistoryController(e.config) for e in envs]
    trials = [dict(episodes=[], deadline=e.deadline, doors=e._doors.tolist()) for e in envs]
    for episode_index in range(3):
        if episode_index:
            observations = next_episodes(envs)
        for planner in planners:
            planner.start_episode()
        currents = [empty_episode(o) for o in observations]
        complete = np.zeros(len(envs), bool)
        while not complete.all():
            world = envs[0].backend.world
            scan = world.sim.lidar_scan()
            actions = [p.choose(e.backend.pose, e.carrying,
                               {k: v[i] for k, v in scan.items()}, e.elapsed, e.deadline)
                       if not complete[i] else np.zeros(2)
                       for i, (p, e) in enumerate(zip(planners, envs))]
            results = advance_batch(envs, actions, ~complete)
            for i, result in enumerate(results):
                if result is None:
                    continue
                obs, reward, done, info = result
                current = currents[i]
                for key, val in [('actions', actions[i]),
                                 ('raw_actions', np.arctanh(np.clip(actions[i], -.999, .999))),
                                 ('log_probs', 0.), ('values', 0.), ('rewards', reward)]:
                    current[key].append(val)
                current['obs'].append(obs)
                observations[i] = obs
                complete[i] = done
        for t, e, planner, env in zip(trials, currents, planners, envs):
            t['episodes'].append(e)
            env.records[-1].update(route=planner.route, known_gates_after=planner.known.copy(),
                                   gate_observations=planner.events.copy())
    for t, e in zip(trials, envs):
        t.update(records=e.records, steps=sum(len(ep['actions']) for ep in t['episodes']))
    return finish_returns(trials)


def observation_tensors(observations, device):
    return (torch.as_tensor(np.array([o['state'] for o in observations]), device=device),
            torch.as_tensor(np.array([o['grid'] for o in observations]), device=device))


@torch.no_grad()
def collect_batch(model, envs, deadlines, *, deterministic=False, patterns=None, reset_memory=False):
    device = next(model.parameters()).device
    batch = len(envs)
    observations = [e.reset(d, None if patterns is None else patterns[i])
                    for i, (e, d) in enumerate(zip(envs, deadlines))]
    trials = [dict(episodes=[], deadline=e.deadline, doors=e._doors.tolist()) for e in envs]
    hidden = model.initial_hidden(batch)
    for episode_index in range(3):
        if episode_index:
            observations = next_episodes(envs)
        currents = [empty_episode(o) for o in observations]
        complete = np.zeros(batch, bool)
        lengths = torch.ones(batch, dtype=torch.long, device=device)
        cache = []
        while not complete.all():
            states, grids = observation_tensors(observations, device)
            encoded = model.encode(states[:, None], grids[:, None])[:, 0]
            cache.append(encoded)
            features = torch.stack(cache, 1)
            means, values, _ = model.forward_features(features, lengths, hidden)
            rows = torch.arange(batch, device=device)
            mean = means[rows, lengths - 1]
            value = values[rows, lengths - 1]
            raw = mean if deterministic else mean + model.log_std.clamp(-3, 0).exp() * torch.randn_like(mean)
            action = raw.tanh()
            logp = log_probability(mean, model.log_std, raw)
            raw_np, act_np, lp_np, val_np = [a.cpu().numpy() for a in (raw, action, logp, value)]
            results = advance_batch(envs, act_np, ~complete)
            for i, env in enumerate(envs):
                if complete[i]:
                    continue
                next_obs, reward, _, info = results[i]
                current = currents[i]
                for key, val in [('actions', act_np[i]), ('raw_actions', raw_np[i]),
                                 ('log_probs', float(lp_np[i])), ('values', float(val_np[i])), ('rewards', reward)]:
                    current[key].append(val)
                current['obs'].append(info.get('terminal_observation', next_obs))
                observations[i] = next_obs
                complete[i] = info['episode_done']
                if not complete[i]:
                    lengths[i] += 1
        # Include terminal contact, reward and done before the cross-episode GRU.
        states, grids, full_lengths = pad_observations(currents, device)
        _, _, hidden = model.forward_episode(states, grids, full_lengths, hidden)
        if reset_memory:
            hidden = model.initial_hidden(batch)
        for trial, current in zip(trials, currents):
            trial['episodes'].append(current)
    for trial, env in zip(trials, envs):
        trial.update(records=env.records, steps=sum(len(ep['actions']) for ep in trial['episodes']))
    return finish_returns(trials)


def pad_observations(episodes, device):
    lengths = torch.tensor([len(e['obs']) for e in episodes], device=device)
    size = int(lengths.max())
    states = np.zeros((len(episodes), size, STATE_DIM), np.float32)
    n = episodes[0]['obs'][0]['grid'].shape[-1]
    grids = np.zeros((len(episodes), size, 3, n, n), np.float32)
    for row, episode in enumerate(episodes):
        length = len(episode['obs'])
        states[row, :length] = [o['state'] for o in episode['obs']]
        grids[row, :length] = [o['grid'] for o in episode['obs']]
    return torch.as_tensor(states, device=device), torch.as_tensor(grids, device=device), lengths


def replay(model, trials):
    device = next(model.parameters()).device
    hidden = model.initial_hidden(len(trials))
    output = {key: [] for key in ('mean', 'value', 'actions', 'raw_actions', 'log_probs', 'returns', 'advantages')}
    for index in range(3):
        episodes = [t['episodes'][index] for t in trials]
        states, grids, lengths = pad_observations(episodes, device)
        mean, value, hidden = model.forward_episode(states, grids, lengths, hidden)
        valid = torch.arange(states.shape[1] - 1, device=device)[None] < lengths[:, None] - 1
        output['mean'].append(mean[:, :-1][valid])
        output['value'].append(value[:, :-1][valid])
        for key in ('actions', 'raw_actions', 'log_probs', 'returns', 'advantages'):
            output[key].append(torch.as_tensor(np.concatenate([np.array(e[key]) for e in episodes]),
                                               dtype=torch.float32, device=device))
    return {key: torch.cat(val) for key, val in output.items()}


def summarize(trials):
    result = {}
    for d in sorted(set(t['deadline'] for t in trials)):
        selected = [t for t in trials if t['deadline'] == d]
        result[str(d)] = dict(mean_trial_score=float(np.mean([sum(r['delivered_value'] for r in t['records']) for t in selected])),
                             episodes=[dict(score=float(np.mean([t['records'][i]['delivered_value'] for t in selected])),
                                            motion_failure_rate=float(np.mean([t['records'][i].get('motion_failed',False) for t in selected])),
                                            delivery_rate=float(np.mean([t['records'][i]['delivered'] for t in selected])))
                                       for i in range(3)])
    return result


def selection_metric(summary,primary='episode3'):
    if primary=='episode3':
        return float(np.mean([row['episodes'][2]['score'] for row in summary.values()]))
    return float(np.mean([row['mean_trial_score'] for row in summary.values()]))


def evaluate(model, config, reset_memory=False, seed=300000):
    cases = [(d, p) for d in config.deadlines for p in PATTERNS]
    envs = make_env_batch(config, [seed+i for i in range(len(cases))])
    trials = collect_batch(model, envs, [d for d, p in cases], deterministic=True,
                           patterns=[p for d, p in cases], reset_memory=reset_memory)
    return summarize(trials), trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bc-updates', type=int, default=200)
    parser.add_argument('--total-steps', type=int, default=100000)
    parser.add_argument('--bc-decay-steps', type=int, default=80000)
    parser.add_argument('--bc-initial-weight', type=float, default=1.)
    parser.add_argument('--batch-trials', type=int, default=12)
    parser.add_argument('--eval-every', type=int, default=20)
    parser.add_argument('--expert-repeats',type=int,default=2)
    parser.add_argument('--calibration-only',action='store_true')
    parser.add_argument('--primary-metric',choices=('episode3','trial_total'),default='episode3')
    args = parser.parse_args()
    bc_coefficient(0, args.bc_decay_steps, args.bc_initial_weight)
    if args.total_steps <= args.bc_decay_steps:
        parser.error('Total steps must exceed BC decay steps to include a PPO-only tail')
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError('Refusing to overwrite nonempty run directory')
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / 'source'
    source.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path, source / path.name)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = Config()
    metadata = make_env_batch(config, [0]*args.batch_trials)[0].metadata()
    shared_source = source / 'shared'
    shared_source.mkdir()
    for filename in ('genesis_motion.py', 'voxel_map.py', 'payload_probe.py', 'payload_probe_genesis.py'):
        shutil.copy2(Path(__file__).parent.parent/'deadline_go2_cargo'/filename, shared_source/filename)
    from payload_probe_genesis import DEFAULT_WEIGHTS
    import hashlib
    lower=args.output/'frozen_locomotion'
    lower.mkdir()
    for filename in ('Full_8000.pt','cfgs.pkl'):
        shutil.copy2(DEFAULT_WEIGHTS/filename,lower/filename)
    metadata['locomotion_sha256']=hashlib.sha256((lower/'Full_8000.pt').read_bytes()).hexdigest()
    metadata['step_unit']='high-level decisions at 2 Hz; frozen locomotion at 50 Hz'
    metadata['bc_schedule']={'warmstart_updates':args.bc_updates,'initial_auxiliary_coefficient':args.bc_initial_weight,
                             'linear_decay_high_level_steps':args.bc_decay_steps,
                             'formula':'beta(s) = initial * max(0, 1 - s/decay_steps)',
                             'sampling':'s is the active transition count at rollout start; one beta per PPO update',
                             'step_count':'summed active high-level transitions across all parallel trials; excludes demonstrations, BC replay, evaluation, and inactive physics',
                             'loss':'PPO loss + beta * BC action MSE; PPO coefficient remains 1',
                             'optimizer_weight_decay':0.,
                             'meaning':'BC loss coefficient decays during PPO; optimizer parameters are not decayed'}
    metadata['torch']=torch.__version__
    save_json(args.output / 'config.json', {**vars(args), 'output': str(args.output), **metadata,
                                              'architecture': 'CNN + causal Transformer + episode-boundary GRU',
                                              'test_protocol': 'all four seen gate patterns at all three deadlines; validation initial seeds 300000+, final evaluation seeds 400000+; feasibility only'})
    model = MetaPolicy(action_dim=2).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    experts=[]
    for repeat in range(args.expert_repeats):
        cases=[(d,p) for d in config.deadlines for p in PATTERNS]
        envs=make_env_batch(config,[100000+repeat*len(cases)+i for i in range(len(cases))])
        experts+=teacher_batch(envs,[d for d,p in cases],[p for d,p in cases])
        print('Expert batch',repeat+1,summarize(experts),flush=True)
    torch.save(experts, args.output/'expert_trajectories.pt')
    save_json(args.output / 'expert_summary.json', summarize(experts))
    save_json(args.output / 'expert_routes.json',[{k:v for k,v in t.items() if k!='episodes'} for t in experts])
    if args.calibration_only:
        return
    start = time.monotonic()
    progress = []
    steps = 0

    def checkpoint(name, phase, update):
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        task=asdict(config), backend=metadata['backend'], phase=phase, update=update,
                        steps=steps, rng_state=dict(torch=torch.get_rng_state(), numpy=rng.bit_generator.state)),
                   args.output / name)

    print('Expert:', summarize(experts), flush=True)
    for update in range(args.bc_updates):
        batch = [experts[int(i)] for i in rng.integers(len(experts), size=args.batch_trials)]
        out = replay(model, batch)
        loss = F.mse_loss(out['mean'].tanh(), out['actions']) + .05 * F.mse_loss(out['value'], out['returns'])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        progress.append(dict(phase='bc', update=update+1, loss=float(loss.detach()), seconds=time.monotonic()-start))
        if (update + 1) % 25 == 0:
            save_json(args.output/'progress.json', progress)
        if (update + 1) % 50 == 0:
            print(f'BC {update+1}: {float(loss.detach()):.5f}; {time.monotonic()-start:.1f}s', flush=True)
    checkpoint('bc.pt', 'bc', args.bc_updates)
    summary, _ = evaluate(model, config)
    save_json(args.output / 'after_bc.json', summary)
    print('After BC:', summary, flush=True)
    progress.append(dict(phase='bc_validation',update=args.bc_updates, validation=summary, seconds=time.monotonic()-start))
    save_json(args.output/'progress.json', progress)
    best = selection_metric(summary,args.primary_metric)
    checkpoint('best.pt', 'bc', args.bc_updates)
    # Reset optimizer moments after supervised warmstart.
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    update = 0
    while steps < args.total_steps:
        schedule_steps = steps
        envs = make_env_batch(config,[int(rng.integers(10000000)) for _ in range(args.batch_trials)])
        deadlines = [config.deadlines[i % 3] for i in rng.permutation(len(envs))]
        trials = collect_batch(model, envs, deadlines)
        steps += sum(t['steps'] for t in trials)
        advantages = np.concatenate([e['advantages'] for t in trials for e in t['episodes']])
        for t in trials:
            for e in t['episodes']:
                e['advantages'] = (e['advantages'] - advantages.mean()) / (advantages.std() + 1e-8)
        weight = bc_coefficient(schedule_steps, args.bc_decay_steps, args.bc_initial_weight)
        for epoch in range(3):
            for ids in np.array_split(rng.permutation(len(trials)), 2):
                out = replay(model, [trials[int(i)] for i in ids])
                lp = log_probability(out['mean'], model.log_std, out['raw_actions'])
                ratio = (lp - out['log_probs']).exp()
                actor = -torch.minimum(ratio * out['advantages'], ratio.clamp(.8, 1.2) * out['advantages']).mean()
                critic = F.mse_loss(out['value'], out['returns'])
                # Entropy is estimated in action space using fresh current-policy samples.
                sample = out['mean'] + model.log_std.clamp(-3, 0).exp() * torch.randn_like(out['mean'])
                entropy = -log_probability(out['mean'], model.log_std, sample).mean()
                loss = actor + .5 * critic - .001 * entropy
                if weight:
                    demos = [experts[int(i)] for i in rng.integers(len(experts), size=4)]
                    bc = replay(model, demos)
                    loss += weight * F.mse_loss(bc['mean'].tanh(), bc['actions'])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                optimizer.step()
        row = dict(phase='ppo', update=update + 1, steps=steps, seconds=time.monotonic() - start,
                   train_score=float(np.mean([sum(r['delivered_value'] for r in t['records']) for t in trials])),
                   bc_weight=weight, bc_schedule_steps=schedule_steps)
        progress.append(row)
        print('PPO rollout', row, flush=True)
        save_json(args.output/'progress.json', progress)
        if (update + 1) % args.eval_every == 0 or steps >= args.total_steps:
            summary, _ = evaluate(model, config)
            row['validation'] = summary
            metric = selection_metric(summary,args.primary_metric)
            if metric > best:
                best = metric
                checkpoint('best.pt', 'ppo', update + 1)
            checkpoint('latest.pt', 'ppo', update + 1)
            save_json(args.output / 'progress.json', progress)
            print('PPO', row, flush=True)
        elif (update+1)%5==0:
            checkpoint('latest.pt','ppo',update+1)
            save_json(args.output/'progress.json',progress)
        update += 1
    checkpoint('final.pt', 'ppo', update)
    result = dict(expert=summarize(experts),primary_metric=args.primary_metric,
                  training_objective='undiscounted three-episode return; checkpoint selection uses the separately specified primary metric',
                  caveat='Single seed, four seen gate patterns in a fixed layout; feasibility, not unseen-layout generalization.')
    for name in ('bc', 'best', 'final'):
        saved = torch.load(args.output / f'{name}.pt', map_location=args.device, weights_only=False)
        model.load_state_dict(saved['model'])
        result[name], trials = evaluate(model, config,seed=400000)
        if name == 'best':
            result['memory_reset'], _ = evaluate(model, config, reset_memory=True,seed=400000)
            result['selected'] = dict(phase=saved['phase'], update=saved['update'])
            save_json(args.output / 'best_trials.json', [{k:v for k,v in t.items() if k != 'episodes'} for t in trials])
    save_json(args.output / 'results.json', result)
    print('RESULTS', result, flush=True)
    checksums = {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for folder in (source, lower) for p in folder.rglob('*') if p.is_file()}
    save_json(args.output/'checksums.json', checksums)


if __name__ == '__main__':
    main()

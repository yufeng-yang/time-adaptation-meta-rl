"""Render the retained episode-level endpoint, or a supplied checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tidyup_plan2.baseline3_gate_meta_rl.continuation_h_comparison import train
from tidyup_plan2.baseline3_gate_meta_rl.continuation_h_comparison.render_step_checkpoint import (
    compare_saved, panel, save_gif,
)

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent / 'release_0_64p5w'
DEFAULT_CHECKPOINT = RELEASE / 'models/episode_645145.pt'


@torch.no_grad()
def record_trial(model, args, seed, deadline=19):
    env = train.episode.make_env(args)
    hidden = model.initial_hidden(torch.device('cpu'))
    records, frame_groups = [], []
    try:
        for index in range(3):
            options = {'new_trial': index == 0, 'deadline': deadline}
            if index == 0:
                options['gate_state'] = 'closed'
            observation, _ = env.reset(seed=seed if index == 0 else None, options=options)
            world = env.unwrapped
            assert world.gate_state == 'closed' and world.deadline == deadline and world.step_count == 0
            initial_hidden = hidden.clone()
            if index:
                assert torch.equal(initial_hidden, previous_terminal_hidden)
            positions = [list(map(int, world.agent_pos))]
            observations, next_observations, actions, rewards, dones = [], [], [], [], []
            evidence = [train.episode.visible_gate_state(observation)]
            hidden_norms = [float(hidden.norm())]
            frames = [panel(world, observation, hidden, index + 1, 'closed', deadline, 0., 'reset',
                            'Episode-level h: fixed within episode')]
            done = False
            while not done:
                logits, _ = model(torch.as_tensor(observation, dtype=torch.float32), hidden)
                action = int(logits.argmax())
                nxt, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                observations.append(observation.copy())
                next_observations.append(nxt.copy())
                actions.append(action)
                rewards.append(float(reward))
                dones.append(float(done))
                observation = nxt
                positions.append(list(map(int, world.agent_pos)))
                evidence.append(train.episode.visible_gate_state(observation))
                hidden_norms.append(float(hidden.norm()))
                route = train.episode.classify_route([tuple(p) for p in positions])
                status = ('GOAL reached' if info.get('success', False) else 'TIMEOUT') if done else 'Running'
                frames.append(panel(world, observation, hidden, index + 1, 'closed', deadline,
                                    sum(rewards), ('left', 'right', 'forward')[action],
                                    f'{status} | route={route}' if done else status))
            # Match the original collector: update once after episodes 1 and 2,
            # using the completed trajectory, never within an episode.
            if index < 2:
                hidden = model.update_hidden(
                    hidden,
                    torch.as_tensor(np.asarray(observations), dtype=torch.float32),
                    torch.as_tensor(actions, dtype=torch.long),
                    torch.as_tensor(rewards, dtype=torch.float32),
                    torch.as_tensor(np.asarray(next_observations), dtype=torch.float32),
                    torch.as_tensor(dones, dtype=torch.float32),
                )
            previous_terminal_hidden = hidden.clone()
            records.append(dict(
                episode=index + 1, success=bool(info.get('success', False)), steps=world.step_count,
                route=route, **{'return': sum(rewards)}, probed_gate=any(e is not None for e in evidence),
                positions=positions, actions=actions, rewards=rewards, gate_observations=evidence,
                hidden_norms=hidden_norms, initial_hidden=initial_hidden.tolist(),
                terminal_hidden=hidden.tolist(),
            ))
            frame_groups.append(frames)
    finally:
        env.close()
    return dict(gate_state='closed', deadline=deadline, seed=seed, episodes=records), frame_groups


@torch.no_grad()
def validate_collector(record, model, args, seed, steps, deadline):
    """Check arbitrary test deadlines against the unchanged training collector."""
    env = train.episode.make_env(args)
    try:
        trial = train.episode.collect_trial(
            model, env, 'closed', seed, steps, args, torch.device('cpu'),
            deterministic=True, trial_deadline=deadline,
        )
    finally:
        env.close()
    for actual, expected in zip(record['episodes'], trial.episodes):
        for key in ('success', 'steps', 'route', 'probed_gate'):
            assert actual[key] == getattr(expected, key), f'Collector mismatch: {key}'
        assert np.isclose(actual['return'], expected.episode_return, atol=1e-6, rtol=0)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--fps', type=float, default=5.)
    parser.add_argument('--deadline', type=int, default=25)
    options = parser.parse_args()
    if not 0 < options.fps <= 30:
        parser.error('--fps must be between 0 and 30')
    if options.deadline <= 0:
        parser.error('--deadline must be positive')
    torch.set_num_threads(1)
    checkpoint = options.checkpoint.resolve()
    _, args, model, _, steps, _, _ = train.load_runtime(checkpoint, 'episode')
    model.eval()
    record, groups = record_trial(model, args, options.seed, options.deadline)
    validation = compare_saved([record], checkpoint, steps, options.seed)
    validation['matches_original_collector'] = validate_collector(
        record, model, args, options.seed, steps, options.deadline,
    )
    output = (options.output_dir or RELEASE / 'generated' / f'renders_episode_{steps}_closed_D{options.deadline}').resolve()
    if output in (checkpoint.parent, checkpoint.parent.parent):
        raise ValueError('Render output must not be a checkpoint/training directory')
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, group in enumerate(groups, 1):
        frames.extend(group)
        frames.extend([group[-1]] * max(1, round(options.fps * .8)))
        group[-1].save(output / f'closed_D{options.deadline}_episode{index}.png')
    save_gif(output / f'closed_D{options.deadline}_trial.gif', frames, round(1000 / options.fps))
    metadata = dict(
        checkpoint=str(checkpoint), checkpoint_steps=steps,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), deterministic=True,
        hidden_update='Once after completed episodes 1 and 2; fixed within each episode; carried across resets',
        terminal_hidden_definition='Post-boundary hidden for episodes 1/2; unchanged for final episode',
        viewer_only='Full map and true gate labels are never added to policy inputs',
        validation=validation, trials=[record],
    )
    (output / 'trajectories.json').write_text(json.dumps(metadata, indent=2))
    print('Validation:', json.dumps(validation))
    for item in record['episodes']:
        print(item['episode'], item['route'], item['success'], item['steps'], round(item['return'], 2))
    print('Render outputs:', output)


if __name__ == '__main__':
    main()

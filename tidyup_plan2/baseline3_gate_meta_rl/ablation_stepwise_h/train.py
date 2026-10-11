"""Stepwise-h ablation of the discrete fixed-trial PPO + decaying BC run.

Only new trials clear memory. After EVERY transition, including episode-ending
transitions, h is updated and passed to the next action / next episode. PPO and
BC replay complete trials with the same causal recurrence and full BPTT.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from tidyup_plan2.baseline3_gate_meta_rl import train_decay_bc as base

HERE = Path(__file__).resolve().parent


class StepwiseActorCritic(base.TrialActorCritic):
    """Same-sized encoders/controller; single-transition context updates.

    For one transition, the parent's attention and max pool both equal that
    transition's encoding. Use precisely that projection before the GRU.
    nn.GRU executes full-trial replay efficiently, with weights initially
    identical to the parent's GRUCell. The attention parameters remain in the
    checkpoint for architectural parity but have no role in a single-token pool.
    """

    def __init__(self, hidden_dim: int):
        super().__init__(hidden_dim)
        cell = self.meta_gru
        rng = torch.random.get_rng_state()
        self.meta_gru = nn.GRU(64, hidden_dim)
        with torch.no_grad():
            for name in ('weight_ih', 'weight_hh', 'bias_ih', 'bias_hh'):
                getattr(self.meta_gru, name + '_l0').copy_(getattr(cell, name))
        # Creating the equivalent sequence implementation must not change the
        # random stream used by subsequent rollout action sampling.
        torch.random.set_rng_state(rng)

    def forward(self, observations, hidden):
        if observations.ndim == 2 and hidden.ndim == 1:
            hidden = hidden.unsqueeze(0).expand(observations.shape[0], -1)
        features = self.body(torch.cat([observations, hidden], dim=-1))
        return self.actor(features), self.critic(features).squeeze(-1)

    def transition_codes(self, observations, actions, rewards, nxt, dones):
        actions = F.one_hot(actions.long(), base.N_ACTIONS).float()
        extras = torch.stack([rewards.clamp(-2., 2.), dones.float()], dim=-1)
        encoded = self.transition_encoder(torch.cat([observations, actions, extras, nxt], dim=-1))
        return self.episode_encoder(torch.cat([encoded, encoded], dim=-1))

    def hidden_sequence(self, hidden, observations, actions, rewards, nxt, dones):
        codes = self.transition_codes(observations, actions, rewards, nxt, dones)
        after, final = self.meta_gru(codes.unsqueeze(1), hidden.reshape(1, 1, -1))
        after = after.squeeze(1)
        before = torch.cat([hidden.unsqueeze(0), after[:-1]], dim=0)
        return before, final.reshape(-1)

    def update_hidden(self, hidden, observations, actions, rewards, nxt, dones):
        return self.hidden_sequence(hidden, observations, actions, rewards, nxt, dones)[1]

    def update_transition(self, hidden, observation, action, reward, nxt, done):
        device = hidden.device
        items = [torch.as_tensor(x, dtype=dtype, device=device) for x, dtype in (
            (observation, torch.float32), (action, torch.long),
            (reward, torch.float32), (nxt, torch.float32), (done, torch.float32),
        )]
        return self.update_hidden(hidden, *(x.unsqueeze(0) for x in items))

    def replay_trial(self, trial):
        device = next(self.parameters()).device
        data = [torch.cat(items, dim=0) for items in zip(
            *(base.episode_tensors(episode, device) for episode in trial.episodes)
        )]
        # Episode boundaries are represented by done=1, not by masking h.
        contexts, final = self.hidden_sequence(self.initial_hidden(device), *data)
        logits, values = self(data[0], contexts)
        return logits, values, final


@torch.no_grad()
def collect_trial(model, env, gate, seed, steps, args, device,
                  deterministic=False, trial_deadline=None):
    deadline = int(trial_deadline) if trial_deadline is not None else base.curriculum_deadline(steps, args.curriculum_steps)
    hidden = model.initial_hidden(device)
    episodes = []
    for index in range(base.EPISODES_PER_TRIAL):
        options = {'new_trial': index == 0, 'deadline': deadline}
        if index == 0:
            options['gate_state'] = gate
        observation, _ = env.reset(seed=seed if index == 0 else None, options=options)
        world = env.unwrapped
        if world.gate_state != gate or world.deadline != deadline or world.step_count != 0:
            raise RuntimeError('Gate, deadline, or full episode budget violated')
        positions = [tuple(map(int, world.agent_pos))]
        observations, actions, rewards, next_items, dones, logs, values = [], [], [], [], [], [], []
        probed = base.visible_gate_state(observation) is not None
        done = False
        while not done:
            logits, value = model(torch.as_tensor(observation, dtype=torch.float32, device=device), hidden)
            distribution = Categorical(logits=logits)
            action = logits.argmax() if deterministic else distribution.sample()
            nxt, reward, terminated, truncated, info = env.step(int(action))
            done = bool(terminated or truncated)
            observations.append(observation.copy())
            actions.append(int(action))
            rewards.append(float(reward))
            next_items.append(nxt.copy())
            dones.append(float(done))
            logs.append(float(distribution.log_prob(action)))
            values.append(float(value))
            # The transition becomes available only AFTER its action/reward.
            # Include terminal transitions, then carry the result across reset.
            hidden = model.update_transition(hidden, observation, int(action), float(reward), nxt, float(done))
            observation = nxt
            probed = probed or base.visible_gate_state(nxt) is not None
            positions.append(tuple(map(int, world.agent_pos)))
        rewards = np.asarray(rewards, np.float32)
        episodes.append(base.Episode(
            observations=np.asarray(observations, np.float32), actions=np.asarray(actions, np.int64),
            rewards=rewards, next_observations=np.asarray(next_items, np.float32),
            dones=np.asarray(dones, np.float32), old_log_probs=np.asarray(logs, np.float32),
            old_values=np.asarray(values, np.float32), episode_return=float(rewards.sum()),
            discounted_return=float(base.discounted(rewards, args.gamma)[0]),
            success=bool(info.get('success', False)), route=base.classify_route(positions),
            steps=int(world.step_count), deadline=deadline, probed_gate=probed,
        ))
    return base.Trial(gate, episodes)


def demonstration_loss(model, demonstrations):
    losses = []
    device = next(model.parameters()).device
    for trial in demonstrations:
        logits, _, _ = model.replay_trial(trial)
        actions = torch.as_tensor(np.concatenate([e.actions for e in trial.episodes]), device=device)
        losses.append(F.cross_entropy(logits, actions, reduction='none'))
    return torch.cat(losses).mean()


def ppo_update(model, optimizer, trials, targets, advantages, args, device, demonstrations, steps):
    old_logs = torch.as_tensor(np.concatenate([e.old_log_probs for t in trials for e in t.episodes]), device=device)
    target = torch.as_tensor(np.concatenate([x for trial in targets for x in trial]), device=device)
    advantages = torch.as_tensor(advantages, device=device)
    for _ in range(args.ppo_epochs):
        logs, values, entropies = [], [], []
        for trial in trials:
            logits, value, _ = model.replay_trial(trial)
            actions = torch.as_tensor(np.concatenate([e.actions for e in trial.episodes]), device=device)
            distribution = Categorical(logits=logits)
            logs.append(distribution.log_prob(actions))
            values.append(value)
            entropies.append(distribution.entropy())
        ratio = (torch.cat(logs) - old_logs).exp()
        actor = -torch.minimum(ratio*advantages, ratio.clamp(1-args.clip_range, 1+args.clip_range)*advantages).mean()
        critic = .5*(torch.cat(values)-target).square().mean()
        entropy = torch.cat(entropies).mean()
        bc = demonstration_loss(model, demonstrations)
        weight = base.bc_coefficient(steps, args)
        loss = actor + args.value_coef*critic - args.entropy_coef*entropy + weight*bc
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        metrics = dict(loss=float(loss.detach()), actor_loss=float(actor.detach()),
                       critic_loss=float(critic.detach()), bc_loss=float(bc.detach()), bc_weight=weight,
                       entropy=float(entropy.detach()), clip_fraction=float(((ratio-1).abs()>args.clip_range).float().mean().detach()))
    return metrics


@torch.no_grad()
def evaluate(model, args, device):
    # Intermediate evaluation must not change the training random stream.
    rng = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    was_training = model.training
    model.eval()
    env = base.make_env(args)
    records = []
    try:
        for gate in ('open', 'closed'):
            for repeat in range(args.eval_trials_per_gate):
                for deadline in base.DEADLINES:
                    trial = collect_trial(model, env, gate, args.eval_seed+repeat, args.total_steps,
                                          args, device, deterministic=True, trial_deadline=deadline)
                    for index, episode in enumerate(trial.episodes, 1):
                        records.append(dict(gate_state=gate, trial=repeat, episode=index, deadline=deadline,
                                            success=episode.success, **{'return':episode.episode_return},
                                            steps=episode.steps, route=episode.route, probed_gate=episode.probed_gate,
                                            deadline_sequence=[deadline]*3))
    finally:
        env.close()
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.random.set_rng_state(rng[2])
        model.train(was_training)
    aggregate = {g:{} for g in ('open','closed')}
    for gate in aggregate:
        for deadline in base.DEADLINES:
            rows = [r for r in records if r['gate_state']==gate and r['deadline']==deadline and r['episode']==3]
            aggregate[gate][str(deadline)] = dict(
                success_rate=float(np.mean([r['success'] for r in rows])),
                mean_return=float(np.mean([r['return'] for r in rows])),
                mean_steps=float(np.mean([r['steps'] for r in rows])),
                route_rates={route:float(np.mean([r['route']==route for r in rows]))
                             for route in ('direct','medium','detour','other')},
            )
    return dict(aggregate=aggregate, episodes=records,
                selection_score=float(np.mean([r['return'] for r in records if r['episode']==3])),
                trial_protocol='fixed gate and deadline; h updated after every transition and inherited across episodes')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    # Match the episode-level experiment's optimizer, curriculum and BC settings.
    for name, default, kind in (
        ('total-steps',550_000,int), ('curriculum-steps',500_000,int), ('trials-per-update',4,int),
        ('hidden-dim',32,int), ('learning-rate',3e-4,float), ('gamma',.99,float), ('beta-trial',1.,float),
        ('ppo-epochs',6,int), ('clip-range',.2,float), ('entropy-coef',.02,float), ('value-coef',.5,float),
        ('max-grad-norm',1.,float), ('bc-initial-weight',1.,float), ('bc-final-weight',0.,float),
        ('bc-decay-steps',600_000,int), ('step-penalty',.01,float), ('time-cap',20.,float),
        ('checkpoint-interval',20,int), ('log-interval',20,int), ('eval-interval-steps',50_000,int),
        ('eval-trials-per-gate',3,int), ('eval-seed',10_000,int), ('seed',0,int),
    ):
        parser.add_argument('--'+name, type=kind, default=default)
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    parser.add_argument('--output-dir', type=Path, default=HERE.parent/'release_0_64p5w/runs/step_from_scratch')
    args = parser.parse_args()
    if args.trials_per_update <= 0 or args.trials_per_update % 2:
        parser.error('trials-per-update must be positive and even')
    if min(args.total_steps,args.ppo_epochs,args.checkpoint_interval,args.log_interval,
           args.eval_interval_steps,args.eval_trials_per_gate,args.hidden_dim) <= 0:
        parser.error('step, epoch, interval and dimension counts must be positive')
    if args.curriculum_steps < 0 or args.bc_decay_steps < 0 or min(args.bc_initial_weight,args.bc_final_weight) < 0:
        parser.error('curriculum and BC schedules must be nonnegative')
    if not 0 <= args.beta_trial <= 1 or not 0 < args.gamma <= 1 or args.time_cap <= 0:
        parser.error('invalid discount, trial weight or time normalization')
    args.output_dir = str(args.output_dir.resolve())
    args.trial_deadline_mode = 'fixed'
    args.episodes_per_trial = 3
    args.hidden_update = 'after_every_transition_including_terminal'
    args.bptt = 'full_trial_no_detach_at_episode_boundaries'
    return args


def main():
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Output is not empty; select a fresh --output-dir')
    output.mkdir(parents=True, exist_ok=True)
    (output/'checkpoints').mkdir()
    (output/'config.json').write_text(json.dumps(vars(args), indent=2))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    env = base.make_env(args)
    model = StepwiseActorCritic(args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    demonstrations = base.build_expert_trials(env, args)
    torch.save([asdict(t) for t in demonstrations], output/'expert_trials.pt')
    fields = ['update','trial','total_steps','gate_state','trial_return']
    for name in ('return','success','route','deadline','steps','probe'):
        fields += [f'episode_{i}_{name}' for i in (1,2,3)]
    fields += ['loss','actor_loss','critic_loss','bc_loss','bc_weight','entropy','clip_fraction']
    steps = updates = trial_count = 0
    next_eval = args.eval_interval_steps
    best_score, best_steps = -float('inf'), 0
    recent = []
    try:
        with (output/'train_trials.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            while steps < args.total_steps:
                gates = ['open']*(args.trials_per_update//2) + ['closed']*(args.trials_per_update//2)
                random.shuffle(gates)
                trials = []
                for gate in gates:
                    trial = collect_trial(model, env, gate, args.seed+trial_count, steps, args, device)
                    trials.append(trial)
                    steps += sum(e.steps for e in trial.episodes)
                    trial_count += 1
                targets, advantages = base.build_meta_targets(trials,args.gamma,args.beta_trial)
                metrics = ppo_update(model,optimizer,trials,targets,advantages,args,device,demonstrations,steps)
                updates += 1
                for offset, trial in enumerate(trials):
                    row = dict(update=updates,trial=trial_count-len(trials)+offset,total_steps=steps,
                               gate_state=trial.gate_state,trial_return=sum(e.episode_return for e in trial.episodes),**metrics)
                    for i,e in enumerate(trial.episodes,1):
                        for name,value in dict(return_=e.episode_return,success=e.success,route=e.route,
                                               deadline=e.deadline,steps=e.steps,probe=e.probed_gate).items():
                            row[f'episode_{i}_{name.rstrip("_")}'] = value
                    writer.writerow(row)
                    recent.append(float(trial.episodes[-1].success))
                handle.flush()
                recent = recent[-50:]
                if updates == 1 or updates % args.log_interval == 0:
                    print(f'update={updates} steps={steps} final_success50={np.mean(recent):.3f} '
                          f'bc={metrics["bc_loss"]:.4f}*{metrics["bc_weight"]:.3f}',flush=True)
                if updates % args.checkpoint_interval == 0:
                    base.save_checkpoint(output/'checkpoints'/f'trial_ppo_{updates}.pt',model,optimizer,args,steps,updates)
                if steps >= next_eval or steps >= args.total_steps:
                    base.save_checkpoint(output/'checkpoints'/f'steps_{steps}.pt',model,optimizer,args,steps,updates)
                    result = evaluate(model,args,device)
                    result['total_steps'] = steps
                    (output/f'evaluation_{steps}.json').write_text(json.dumps(result,indent=2))
                    print('evaluation',steps,json.dumps(result['aggregate']),flush=True)
                    if result['selection_score'] > best_score:
                        best_score,best_steps = result['selection_score'],steps
                        base.save_checkpoint(output/'best_model.pt',model,optimizer,args,steps,updates)
                        (output/'best_evaluation.json').write_text(json.dumps(result,indent=2))
                    next_eval = (steps//args.eval_interval_steps+1)*args.eval_interval_steps
        base.save_checkpoint(output/'final_model.pt',model,optimizer,args,steps,updates)
        (output/'evaluation.json').write_text(json.dumps(result,indent=2))
        summary = dict(total_steps=steps,updates=updates,trials=trial_count,
                       hidden_update=args.hidden_update,bptt=args.bptt,
                       bc_weight=base.bc_coefficient(steps,args),best_steps=best_steps,
                       best_selection_score=best_score,evaluation=result['aggregate'])
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,indent=2),flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()

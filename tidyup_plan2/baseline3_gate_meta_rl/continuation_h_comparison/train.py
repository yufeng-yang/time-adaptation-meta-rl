"""Continue either memory architecture, preserving optimizer and global schedules."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tidyup_plan2.baseline3_gate_meta_rl import train_decay_bc as episode
from tidyup_plan2.baseline3_gate_meta_rl.ablation_stepwise_h import train as step

HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINTS = {
    'episode': HERE.parent/'release_0_64p5w/models/episode_551179.pt',
    'step': HERE.parent/'release_0_64p5w/models/step_550096.pt',
}


def capture_rng():
    return {'python':random.getstate(), 'numpy':np.random.get_state(),
            'torch':torch.random.get_rng_state()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['torch'])


def backend(mode):
    if mode == 'episode':
        return episode.TrialActorCritic, episode.collect_trial, episode.ppo_update, episode.evaluate
    if mode == 'step':
        return step.StepwiseActorCritic, step.collect_trial, step.ppo_update, step.evaluate
    raise ValueError(f'Unknown memory mode: {mode}')


def load_runtime(path, mode):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    args = argparse.Namespace(**checkpoint['config'])
    checkpoint_is_step = getattr(args,'hidden_update',None) == 'after_every_transition_including_terminal'
    if checkpoint_is_step != (mode == 'step'):
        raise ValueError('Checkpoint architecture does not match --mode')
    model_type, _, _, _ = backend(mode)
    model = model_type(args.hidden_dim)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    steps, updates = int(checkpoint['total_steps']), int(checkpoint['updates'])
    trials = int(checkpoint.get('trial_count', updates*args.trials_per_update))
    return checkpoint, args, model, optimizer, steps, updates, trials


def save_checkpoint(path, model, optimizer, args, steps, updates, trials):
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                 config=vars(args), total_steps=steps, updates=updates, trial_count=trials,
                 rng_state=capture_rng())
    temporary = path.with_suffix('.pending')
    torch.save(value, temporary)
    temporary.replace(path)


def task_deadlines(args, gate):
    return ((15,17,25) if gate=='closed' and getattr(args,'closed_long_deadline',19)==25
            else episode.DEADLINES)


def build_demonstrations(env, args):
    if getattr(args,'closed_long_deadline',19)==19:
        return episode.build_expert_trials(env,args)
    return [episode.collect_expert_trial(env,gate,deadline,
                args.seed+100_000+gate_index*100+deadline,args)
            for gate_index,gate in enumerate(('open','closed'))
            for deadline in task_deadlines(args,gate)]


@torch.no_grad()
def evaluate_changed_tasks(model, args, mode):
    """Evaluate the same six gate-conditioned tasks used by the new sampler."""
    model.eval()
    _,collect,_,_ = backend(mode)
    env = episode.make_env(args)
    records = []
    try:
        for gate in ('open','closed'):
            for repeat in range(args.eval_trials_per_gate):
                for deadline in task_deadlines(args,gate):
                    trial = collect(model,env,gate,args.eval_seed+repeat,args.total_steps,args,
                                    torch.device('cpu'),deterministic=True,trial_deadline=deadline)
                    for index,e in enumerate(trial.episodes,1):
                        records.append(dict(gate_state=gate,trial=repeat,episode=index,
                            deadline=e.deadline,success=e.success,**{'return':e.episode_return},
                            steps=e.steps,route=e.route,probed_gate=e.probed_gate,
                            deadline_sequence=[deadline]*3))
    finally:
        env.close()
    aggregate = {'open':{},'closed':{}}
    for gate in aggregate:
        for deadline in task_deadlines(args,gate):
            rows = [r for r in records if r['gate_state']==gate and r['deadline']==deadline and r['episode']==3]
            aggregate[gate][str(deadline)] = dict(
                success_rate=float(np.mean([r['success'] for r in rows])),
                mean_return=float(np.mean([r['return'] for r in rows])),
                mean_steps=float(np.mean([r['steps'] for r in rows])),
                route_rates={route:float(np.mean([r['route']==route for r in rows]))
                             for route in ('direct','medium','detour','other')})
    return dict(aggregate=aggregate,episodes=records,
                trial_protocol='fixed gate and deadline across all three episodes')


def diagnostic(result):
    ideal = {('open',15):(.86,'direct',True), ('open',17):(.86,'direct',True),
             ('open',19):(.86,'direct',True), ('closed',15):(-1.15,'detour',False),
             ('closed',17):(.64,'medium',True), ('closed',19):(.82,'detour',True),
             ('closed',25):(.82,'detour',True)}
    rows = [r for r in result['episodes'] if r['episode'] > 1]
    correct = []
    for row in rows:
        reward, route, success = ideal[row['gate_state'],row['deadline']]
        correct.append(row['route']==route and row['success']==success
                       and abs(row['return']-reward)<1e-5)
    feasible = [r for r in rows if (r['gate_state'],r['deadline']) != ('closed',15)]
    return dict(mean_return_ep23=float(np.mean([r['return'] for r in rows])),
                optimal_fraction_ep23=float(np.mean(correct)),
                feasible_success_ep23=float(np.mean([r['success'] for r in feasible])))


def evaluate_preserving_rng(model, args, mode):
    state = capture_rng()
    was_training = model.training
    try:
        _, _, _, evaluator = backend(mode)
        result = (evaluate_changed_tasks(model,args,mode)
                  if getattr(args,'closed_long_deadline',19)==25
                  else evaluator(model,args,torch.device('cpu')))
        result['selection_score'] = float(np.mean([r['return'] for r in result['episodes'] if r['episode']==3]))
        result['diagnostics'] = diagnostic(result)
        return result
    finally:
        restore_rng(state)
        model.train(was_training)


def run(options):
    torch.set_num_threads(1)
    output = options.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Output directory is not empty; original data will not be overwritten')
    source = (options.checkpoint or DEFAULT_CHECKPOINTS[options.mode]).resolve()
    checkpoint,args,model,optimizer,steps,updates,trial_count = load_runtime(source,options.mode)
    if options.total_steps <= steps:
        raise ValueError('--total-steps is the ABSOLUTE target and must exceed checkpoint steps')
    if options.eval_interval_steps<=0 or options.checkpoint_interval<=0 or options.log_interval<=0:
        raise ValueError('All intervals must be positive')
    start_steps,start_updates,start_trials = steps,updates,trial_count
    args.total_steps = options.total_steps
    args.output_dir = str(output)
    args.device = 'cpu'
    args.continuation_mode = options.mode
    args.resume_checkpoint = str(source)
    args.resume_seed = options.resume_seed
    args.eval_interval_steps = options.eval_interval_steps
    args.checkpoint_interval = options.checkpoint_interval
    args.log_interval = options.log_interval
    if getattr(options,'closed_long_deadline',None) is not None:
        args.closed_long_deadline = options.closed_long_deadline
    args.task_deadlines = {gate:list(task_deadlines(args,gate)) for gate in ('open','closed')}
    args.rng_resume = 'restored' if 'rng_state' in checkpoint else 'new_seed_old_checkpoint_has_no_rng'
    args.hidden_update = ('after_every_transition_including_terminal' if options.mode=='step'
                          else 'after_completed_episode')
    output.mkdir(parents=True, exist_ok=True)
    (output/'checkpoints').mkdir()
    (output/'config.json').write_text(json.dumps(vars(args), indent=2))
    env = episode.make_env(args)
    demonstrations = build_demonstrations(env,args)
    torch.save([asdict(t) for t in demonstrations], output/'expert_trials.pt')
    # Initialization and demonstration construction must not consume continuation RNG.
    if 'rng_state' in checkpoint:
        restore_rng(checkpoint['rng_state'])
    else:
        random.seed(options.resume_seed)
        np.random.seed(options.resume_seed)
        torch.manual_seed(options.resume_seed)
    provenance = dict(source_checkpoint=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      start_steps=start_steps, start_updates=start_updates, start_trials=start_trials,
                      optimizer_restored=True, rng_resume=args.rng_resume, resume_seed=options.resume_seed,
                      start_bc_weight=episode.bc_coefficient(steps,args), total_steps_target=args.total_steps,
                      evaluation_interval=args.eval_interval_steps,task_deadlines=args.task_deadlines)
    (output/'resume_info.json').write_text(json.dumps(provenance,indent=2))
    fields = ['update','trial','total_steps','gate_state','trial_return']
    for name in ('return','success','route','deadline','steps','probe'):
        fields += [f'episode_{i}_{name}' for i in (1,2,3)]
    fields += ['loss','actor_loss','critic_loss','bc_loss','bc_weight','entropy','clip_fraction']
    _, collect, update, _ = backend(options.mode)
    next_eval = (steps//args.eval_interval_steps+1)*args.eval_interval_steps
    best_score = -float('inf')
    best_steps = None
    started = time.monotonic()
    progress_path = output/'progress.json'
    with (output/'progress.log').open('w', buffering=1) as log:
        def emit(message):
            print(message,flush=True)
            log.write(str(message)+'\n')

        def write_progress(status):
            value = dict(status=status,mode=options.mode,total_steps=steps,updates=updates,trials=trial_count,
                         start_steps=start_steps,target_steps=args.total_steps,
                         bc_weight=episode.bc_coefficient(steps,args),elapsed_seconds=time.monotonic()-started)
            temporary = progress_path.with_suffix('.pending')
            temporary.write_text(json.dumps(value,indent=2))
            temporary.replace(progress_path)

        def evaluate_and_save():
            nonlocal best_score,best_steps
            result = evaluate_preserving_rng(model,args,options.mode)
            result.update(total_steps=steps,mode=options.mode)
            (output/f'evaluation_{steps}.json').write_text(json.dumps(result,indent=2))
            emit(f'evaluation steps={steps} bc={episode.bc_coefficient(steps,args):.5f} '
                 f'diagnostics={json.dumps(result["diagnostics"])} '
                 f'closed={json.dumps(result["aggregate"]["closed"])}')
            if result['selection_score'] > best_score:
                best_score,best_steps = result['selection_score'],steps
                save_checkpoint(output/'best_model.pt',model,optimizer,args,steps,updates,trial_count)
                (output/'best_evaluation.json').write_text(json.dumps(result,indent=2))
            return result

        try:
            emit('RESUME '+json.dumps(provenance))
            write_progress('running')
            save_checkpoint(output/'initial_model.pt',model,optimizer,args,steps,updates,trial_count)
            result = evaluate_and_save()
            with (output/'train_trials.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle,fieldnames=fields)
                writer.writeheader()
                while steps < args.total_steps:
                    gates = ['open']*(args.trials_per_update//2)+['closed']*(args.trials_per_update//2)
                    random.shuffle(gates)
                    trials = []
                    model.train()
                    for gate in gates:
                        if getattr(args,'closed_long_deadline',19)==25:
                            deadline = int(np.random.choice(task_deadlines(args,gate)))
                            trial = collect(model,env,gate,args.seed+trial_count,steps,args,
                                            torch.device('cpu'),trial_deadline=deadline)
                        else:
                            trial = collect(model,env,gate,args.seed+trial_count,steps,args,torch.device('cpu'))
                        trials.append(trial)
                        steps += sum(e.steps for e in trial.episodes)
                        trial_count += 1
                    targets,advantages = episode.build_meta_targets(trials,args.gamma,args.beta_trial)
                    metrics = update(model,optimizer,trials,targets,advantages,args,torch.device('cpu'),demonstrations,steps)
                    updates += 1
                    for offset,trial in enumerate(trials):
                        row = dict(update=updates,trial=trial_count-len(trials)+offset,total_steps=steps,
                                   gate_state=trial.gate_state,trial_return=sum(e.episode_return for e in trial.episodes),**metrics)
                        for i,e in enumerate(trial.episodes,1):
                            values = {'return':e.episode_return,'success':e.success,'route':e.route,
                                      'deadline':e.deadline,'steps':e.steps,'probe':e.probed_gate}
                            row.update({f'episode_{i}_{name}':value for name,value in values.items()})
                        writer.writerow(row)
                    handle.flush()
                    write_progress('running')
                    if updates % args.log_interval == 0:
                        emit(f'mode={options.mode} update={updates} steps={steps} '
                             f'bc={metrics["bc_loss"]:.4f}*{metrics["bc_weight"]:.5f} '
                             f'elapsed={time.monotonic()-started:.1f}s')
                    if updates % args.checkpoint_interval == 0:
                        save_checkpoint(output/'checkpoints'/f'trial_ppo_{updates}.pt',model,optimizer,args,steps,updates,trial_count)
                    if steps >= next_eval or steps >= args.total_steps:
                        save_checkpoint(output/'checkpoints'/f'steps_{steps}.pt',model,optimizer,args,steps,updates,trial_count)
                        result = evaluate_and_save()
                        next_eval = (steps//args.eval_interval_steps+1)*args.eval_interval_steps
            save_checkpoint(output/'final_model.pt',model,optimizer,args,steps,updates,trial_count)
            (output/'evaluation.json').write_text(json.dumps(result,indent=2))
            summary = dict(mode=options.mode,total_steps=steps,updates=updates,trials=trial_count,
                           start_steps=start_steps,additional_steps=steps-start_steps,
                           additional_updates=updates-start_updates,additional_trials=trial_count-start_trials,
                           bc_weight=episode.bc_coefficient(steps,args),best_steps=best_steps,
                           best_selection_score=best_score,evaluation=result['aggregate'],
                           diagnostics=result['diagnostics'],elapsed_seconds=time.monotonic()-started,
                           rng_resume=args.rng_resume)
            (output/'summary.json').write_text(json.dumps(summary,indent=2))
            write_progress('complete')
            emit('COMPLETE '+json.dumps(summary))
            return summary
        except BaseException as error:
            write_progress('failed')
            emit(f'FAILED {type(error).__name__}: {error}')
            raise
        finally:
            env.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=('episode','step'))
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--total-steps',type=int,default=645_000)
    parser.add_argument('--resume-seed',type=int,default=20261011)
    parser.add_argument('--eval-interval-steps',type=int,default=25_000)
    parser.add_argument('--checkpoint-interval',type=int,default=20)
    parser.add_argument('--log-interval',type=int,default=100)
    parser.add_argument('--closed-long-deadline',type=int,choices=(19,25),
                        help='Replace only closed D19 with D25; preserve all other tasks and schedules')
    parser.add_argument('--output-dir',type=Path)
    args = parser.parse_args()
    if args.closed_long_deadline is None and args.checkpoint is None:
        args.closed_long_deadline = 25 if args.mode=='episode' else 19
    if args.output_dir is None:
        args.output_dir = HERE.parent/'release_0_64p5w/runs'/args.mode
    return args


if __name__ == '__main__':
    run(parse_args())

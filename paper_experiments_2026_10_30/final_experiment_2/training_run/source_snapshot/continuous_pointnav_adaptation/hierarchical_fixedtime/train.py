"""Fresh high-level trial policy, physically executed by a frozen low actor."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from continuous_pointnav_adaptation.hierarchical_fixedtime import scenario
from continuous_pointnav_adaptation.hierarchical_fixedtime.core import (
    TrialPolicy,bc_weight,bc_step_weights,collect_trial,demonstration_loss,episode_records,update,
)
from continuous_pointnav_adaptation.hierarchical_fixedtime.environment import make_env,LOW_CHECKPOINT,DEADLINES

HERE=Path(__file__).parent
CASES=[(gate,d) for gate in ('open','closed') for d in DEADLINES]


def training_cases(include_infeasible=False):
    # Exclude an entire case only if calibration expects no successful episode
    # in its trial. A failed probing episode in an otherwise feasible trial is
    # still included, so the meta-learning objective is preserved.
    return [case for case in CASES if include_infeasible or
            any(scenario.expected_success(*case,index) for index in range(3))]


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def checkpoint(path,model,optimizer,config,steps,updates):
    tmp=path.with_suffix('.tmp')
    torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
        config=config,total_steps=steps,updates=updates,python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),torch_rng=torch.random.get_rng_state()),tmp)
    tmp.replace(path)


def load_model(path):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=TrialPolicy(saved['config']['observation_spec']['obs_dim'])
    model.load_state_dict(saved['model_state_dict']);model.eval()
    return model,saved


@torch.no_grad()
def evaluate(model,envs,steps,*,zero_memory=False):
    state=random.getstate(),np.random.get_state(),torch.random.get_rng_state()
    records=[]
    try:
        model.eval()
        for gate,d in CASES:
            trial=collect_trial(model,envs[gate],gate,d,20000,deterministic=True,zero_memory=zero_memory)
            records.extend(episode_records(trial))
        last=[r for r in records if r['episode']==3]
        return dict(total_steps=steps,episodes=records,zero_memory=zero_memory,
            third_episode_success_rate=float(np.mean([r['success'] for r in last])),
            third_episode_mean_return=float(np.mean([r['return_'] for r in last])),
            **{f'episode_{i}_success_rate':float(np.mean([r['success'] for r in records if r['episode']==i])) for i in (1,2,3)})
    finally:
        random.setstate(state[0]);np.random.set_state(state[1]);torch.random.set_rng_state(state[2])


_worker_envs=_worker_model=None


def worker_init():
    global _worker_envs,_worker_model
    torch.set_num_threads(1)
    _worker_envs={g:make_env(fixed_gate=g) for g in ('open','closed')}
    _worker_model=TrialPolicy(63);_worker_model.eval()


def worker_collect(payload):
    weights,gate,deadline,seed=payload
    _worker_model.load_state_dict({k:torch.from_numpy(v) for k,v in weights.items()})
    return scenario.collect_trial(_worker_model,_worker_envs[gate],gate,deadline,seed)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=HERE/'runs/seed0_wide_left_v3_D145bonus002_D230normal_BCturn5win2_500k')
    p.add_argument('--include-infeasible-training',action='store_true',
                   help='baseline override: include cases with no feasible episode in PPO and BC')
    for k,v,t in [('total-steps',500000,int),('workers',2,int),('seed',0,int),
        ('eval-steps',10000,int),('bc-warmup-updates',20,int),('bc-decay-steps',400000,int),
        ('bc-initial-weight',1.,float),('bc-min-weight',.1,float),('learning-rate',3e-4,float),
        ('bc-critical-weight',5.,float),('bc-turn-window',2,int),
        ('gamma',1.,float),('ppo-epochs',4,int),('clip-range',.2,float),('entropy-coef',.005,float),
        ('value-coef',.5,float),('max-grad-norm',1.,float)]:p.add_argument('--'+k,type=t,default=v)
    args=p.parse_args()
    bc_step_weights(torch.tensor([1]),args.bc_critical_weight,args.bc_turn_window)
    args.return_objective='weighted_episodes';args.episode_weights=(.2,.5,1.)
    return args


def main():
    args=parse_args();output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    train_cases=training_cases(args.include_infeasible_training)
    torch.set_num_threads(1);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    started=time.monotonic();steps=updates=0;phase='demonstrations';pool=None
    low_hash=hashlib.sha256(LOW_CHECKPOINT.read_bytes()).hexdigest()
    envs={g:make_env(fixed_gate=g) for g in ('open','closed')}
    config=vars(args)|dict(output_dir=str(output),scenario='physical_fixedtime',absolute_xy=True,
        deadlines=list(DEADLINES),cases=CASES,episodes_per_trial=3,action_dim=3,
        objective_episode_weights=list(args.episode_weights),
        context_update='completed episodes only; fixed within episode; reset each trial',
        observation_spec=envs['open'].observation_spec,
        low_checkpoint=str(LOW_CHECKPOINT.resolve()),low_sha256=low_hash,low_frozen=True,
        step_penalty=envs['open'].step_penalty,collision_penalty=envs['open'].collision_penalty,
        danger_penalty=envs['open'].danger_penalty,goal_reward=1.,timeout_penalty=1.,
        first_episode_danger_bonus=envs['open'].first_episode_danger_bonus,
        first_episode_bonus_deadlines=envs['open'].observation_spec['first_episode_bonus_deadlines'],
        reward_version=envs['open'].observation_spec['reward_version'],
        bc_sample_weighting='expert left/right turn and +/- window commands: critical weight; other steps: 1; weighted sum / sum weights',
        input_normalization='frozen demonstration mean/std min 0.1, clamp +-5; time unchanged',
        memory_rewards='raw environment rewards; weights affect PPO targets only',
        evaluation_action_selection='deterministic argmax; all 3 episodes for memory',
        checkpoint_selection='third episode success first, third episode mean return as tie breaker',
        initialization='new high network; no prior high checkpoint loaded',
        demonstration_waypoints=scenario.ROUTES,
        probe_upper_return_waypoint=scenario.PROBE_UPPER_RETURN,
        training_cases=train_cases,evaluation_cases=CASES,
        excluded_training_cases=[c for c in CASES if c not in train_cases],
        exclusion_scope='both online PPO collection and auxiliary BC demonstrations; evaluation retains all cases')
    write(output/'config.json',config)
    sources=[HERE/n for n in ('environment.py','scenario.py','core.py','train.py','calibrate.py')]
    sources+=[scenario.CALIBRATION_FILE,LOW_CHECKPOINT,HERE.parent/'lowlevel_heading_speed/policy.py',
        HERE.parent/'discrete_waypoint_16/six_case_calibrated/environment.py',
        HERE.parent/'discrete_waypoint_16/trial_bc_radar/environment.py',
        HERE.parent/'discrete_waypoint_16/environment.py',HERE.parent/'envs/point_gate_goal.py',
        HERE.parent/'naive_training/environment.py']
    for source in sources:
        dest=output/'source_snapshot'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
    write(output/'source_manifest.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})

    def progress(status='running',**kwargs):
        write(output/'progress.json',dict(status=status,phase=phase,total_steps=steps,updates=updates,
            target_steps=args.total_steps,elapsed_seconds=time.monotonic()-started,
            bc_weight=bc_weight(steps,args),**kwargs))

    try:
        progress();demonstrations=[]
        for gate,d in train_cases:
            trial=scenario.collect_trial(None,envs[gate],gate,d,100000+args.seed,teacher=True)
            for index,episode in enumerate(trial.episodes):
                assert episode.collisions==0,(gate,d,index,episode.collisions)
                assert episode.success==scenario.expected_success(gate,d,index),(gate,d,index,episode.success)
                assert episode.observed_gate==gate
            demonstrations.append(trial)
            print('EXPERT',gate,d,[(e.steps,e.success,e.route) for e in trial.episodes],flush=True)
        torch.save(demonstrations,output/'expert_trials.pt')
        write(output/'expert_summary.json',[r for t in demonstrations for r in episode_records(t)])
        stage_summary=[]
        for trial in demonstrations:
            for index,episode in enumerate(trial.episodes):
                actions=torch.as_tensor(episode.actions)
                weights=bc_step_weights(actions,args.bc_critical_weight,args.bc_turn_window)
                stage_summary.append(dict(gate=trial.gate,deadline=trial.deadline,episode=index+1,
                    steps=episode.steps,turn_steps=int((actions!=1).sum()),
                    upweighted_steps=int((weights>1).sum()),weight_sum=float(weights.sum())))
        write(output/'bc_stage_weights.json',stage_summary)
        model=TrialPolicy(63)
        obs=np.concatenate([e.observations for t in demonstrations for e in t.episodes])
        model.obs_mean.copy_(torch.from_numpy(obs.mean(0)))
        model.obs_scale.copy_(torch.from_numpy(np.maximum(obs.std(0),.1)))
        model.obs_mean[-1]=0.;model.obs_scale[-1]=1.
        optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
        checkpoint(output/'initial_model.pt',model,optimizer,config,0,0)
        result=evaluate(model,envs,0);result['phase']='random_initialization'
        write(output/'initial_evaluation.json',result)
        best=(result['third_episode_success_rate'],result['third_episode_mean_return'])
        checkpoint(output/'best_model.pt',model,optimizer,config,0,0);write(output/'best_selection.json',result)
        print('INITIAL ep3',result['third_episode_success_rate'],result['third_episode_mean_return'],flush=True)
        with (output/'evaluation_curve.csv').open('w',newline='') as curve,(output/'training.csv').open('w',newline='') as trainlog:
            curve_writer=csv.DictWriter(curve,fieldnames=['total_steps','third_episode_success_rate',
                'third_episode_mean_return','episode_1_success_rate','episode_2_success_rate',
                'episode_3_success_rate','bc_weight','phase']);curve_writer.writeheader()

            def record(result):
                curve_writer.writerow({k:result[k] if k in result else bc_weight(steps,args) for k in curve_writer.fieldnames});curve.flush()

            record(result);phase='bc_warmup';progress();losses=[]
            for warmup in range(args.bc_warmup_updates):
                model.train();loss=demonstration_loss(model,demonstrations,
                    critical_weight=args.bc_critical_weight,turn_window=args.bc_turn_window)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm);optimizer.step()
                losses.append(float(loss.detach()))
            write(output/'bc_warmup.json',losses);checkpoint(output/'bc_model.pt',model,optimizer,config,0,0)
            result=evaluate(model,envs,0);result['phase']='post_bc';record(result);write(output/'post_bc_evaluation.json',result)
            score=(result['third_episode_success_rate'],result['third_episode_mean_return'])
            if score>best:
                best=score;checkpoint(output/'best_model.pt',model,optimizer,config,0,0);write(output/'best_selection.json',result)
            checkpoint(output/'latest_model.pt',model,optimizer,config,0,0)
            print('POST_BC ep3',result['third_episode_success_rate'],result['third_episode_mean_return'],flush=True)
            phase='ppo_bc';progress()
            metric_fields=['total_steps','updates','loss','actor_loss','critic_loss','entropy','bc_loss','bc_weight','clip_fraction','grad_norm']
            train_writer=csv.DictWriter(trainlog,fieldnames=metric_fields);train_writer.writeheader()
            pool=ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=worker_init) if args.workers>1 else None
            next_eval=args.eval_steps
            while steps<args.total_steps:
                phase='ppo_bc';weights={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
                payloads=[(weights,g,d,args.seed+1000000+updates*100+i) for i,(g,d) in enumerate(train_cases)]
                model.eval()
                trials=(list(pool.map(worker_collect,payloads)) if pool else
                        [scenario.collect_trial(model,envs[g],g,d,seed) for _,g,d,seed in payloads])
                steps+=sum(e.steps for t in trials for e in t.episodes)
                model.train();metrics=update(model,optimizer,trials,demonstrations,steps,args);updates+=1
                train_writer.writerow(dict(total_steps=steps,updates=updates,**metrics));trainlog.flush()
                progress(latest_metrics=metrics)
                if updates%5==0:print('PPO',steps,'updates',updates,'BC',round(metrics['bc_weight'],3),flush=True)
                if steps>=next_eval or steps>=args.total_steps:
                    result=evaluate(model,envs,steps);result['phase']='online';record(result)
                    write(output/f'evaluation_{steps}.json',result)
                    score=(result['third_episode_success_rate'],result['third_episode_mean_return'])
                    if score>best:
                        best=score;checkpoint(output/'best_model.pt',model,optimizer,config,steps,updates);write(output/'best_selection.json',result)
                    checkpoint(output/'latest_model.pt',model,optimizer,config,steps,updates)
                    progress(latest_evaluation={k:v for k,v in result.items() if k!='episodes'})
                    print('EVAL',steps,'ep3',round(result['third_episode_success_rate'],4),'return',round(result['third_episode_mean_return'],4),flush=True)
                    next_eval=(steps//args.eval_steps+1)*args.eval_steps
            checkpoint(output/'final_model.pt',model,optimizer,config,steps,updates)
            write(output/'final_evaluation.json',result)
            write(output/'final_zero_memory.json',evaluate(model,envs,steps,zero_memory=True))
            assert hashlib.sha256(LOW_CHECKPOINT.read_bytes()).hexdigest()==low_hash
            phase='complete';progress(status='completed',final_evaluation={k:v for k,v in result.items() if k!='episodes'})
    except BaseException as error:
        progress(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error));raise
    finally:
        if pool:pool.shutdown(wait=True,cancel_futures=True)
        for env in envs.values():env.close()


if __name__=='__main__':main()

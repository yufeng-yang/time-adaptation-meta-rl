"""Fresh input ablations using the archived physical environment and PPO engine."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import time
import numpy as np
import torch
from continuous_paper_comparison.runtime import (
    ARCHIVE,ROOT,AblationPolicy,archive_config,make_env,core,scenario,assert_frozen_imports)

WORKER_ENVS=None
WORKER_MODEL=None


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def worker_init(mode):
    global WORKER_ENVS,WORKER_MODEL
    torch.set_num_threads(1)
    WORKER_ENVS={g:make_env(g) for g in ('open','closed')}
    WORKER_MODEL=AblationPolicy(mode=mode);WORKER_MODEL.eval()


def worker_collect(payload):
    weights,gate,deadline,seed=payload
    WORKER_MODEL.load_state_dict({k:torch.from_numpy(v) for k,v in weights.items()})
    return scenario.collect_trial(WORKER_MODEL,WORKER_ENVS[gate],gate,deadline,seed)


def evaluate(model,envs,steps):
    states=random.getstate(),np.random.get_state(),torch.get_rng_state()
    model.eval();records=[]
    try:
        for gate in ('open','closed'):
            for d in (75,145,230):
                trial=scenario.collect_trial(model,envs[gate],gate,d,20000,deterministic=True)
                records+=core.episode_records(trial)
        last=[r for r in records if r['episode']==3]
        return dict(steps=steps,episodes=records,
                    final_success=float(np.mean([r['success'] for r in last])),
                    final_return=float(np.mean([r['return_'] for r in last])))
    finally:
        random.setstate(states[0]);np.random.set_state(states[1]);torch.set_rng_state(states[2])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('state','time','context'),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=5)
    parser.add_argument('--steps',type=int,default=500000);parser.add_argument('--eval-steps',type=int,default=10000)
    opt=parser.parse_args();out=opt.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=archive_config();cfg.update(policy_mode=opt.mode,output_dir=str(out),total_steps=opt.steps,
                                  workers=opt.workers,eval_steps=opt.eval_steps,
                                  baseline_initialization='fresh random weights')
    args=argparse.Namespace(**cfg)
    args.return_objective='weighted_episodes';args.episode_weights=(.2,.5,1.)
    torch.set_num_threads(1);random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    write(out/'config.json',cfg);started=time.monotonic();steps=updates=0
    envs={g:make_env(g) for g in ('open','closed')}
    demos=[]
    for gate,d in cfg['training_cases']:
        trial=scenario.collect_trial(None,envs[gate],gate,d,100000,teacher=True)
        for i,ep in enumerate(trial.episodes):
            assert ep.collisions==0 and ep.success==scenario.expected_success(gate,d,i)
        demos.append(trial)
    torch.save(demos,out/'expert_trials.pt');write(out/'expert_summary.json',[r for t in demos for r in core.episode_records(t)])
    model=AblationPolicy(mode=opt.mode)
    obs=np.concatenate([e.observations for t in demos for e in t.episodes])
    model.obs_mean.copy_(torch.from_numpy(obs.mean(0)))
    model.obs_scale.copy_(torch.from_numpy(np.maximum(obs.std(0),.1)))
    model.obs_mean[-1]=0;model.obs_scale[-1]=1
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    def save(name):
        torch.save(dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                        config=cfg,total_steps=steps,updates=updates,
                        python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state()),out/name)
    save('initial_model.pt')
    initial=evaluate(model,envs,0);write(out/'initial_validation.json',initial)
    best=(initial['final_success'],initial['final_return']);save('best_model.pt')
    for _ in range(args.bc_warmup_updates):
        model.train();loss=core.demonstration_loss(model,demos,critical_weight=args.bc_critical_weight,turn_window=args.bc_turn_window)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm);optimizer.step()
    save('bc_model.pt')
    val=evaluate(model,envs,0);write(out/'bc_validation.json',val)
    score=(val['final_success'],val['final_return'])
    if score>best:best=score;save('best_model.pt')
    print('BC',opt.mode,val['final_success'],val['final_return'],flush=True)
    fields=['steps','update','seconds','final_success','final_return']
    with (out/'validation_curve.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        next_eval=opt.eval_steps
        with ProcessPoolExecutor(max_workers=opt.workers,mp_context=mp.get_context('spawn'),
                                 initializer=worker_init,initargs=(opt.mode,)) as pool:
            while steps<opt.steps:
                model.eval();weights={k:v.detach().numpy().copy() for k,v in model.state_dict().items()}
                payload=[(weights,g,d,cfg['seed']+1000000+updates*100+i) for i,(g,d) in enumerate(cfg['training_cases'])]
                trials=list(pool.map(worker_collect,payload))
                steps+=sum(e.steps for t in trials for e in t.episodes);updates+=1
                model.train();metrics=core.update(model,optimizer,trials,demos,steps,args)
                write(out/'progress.json',dict(status='running',steps=steps,updates=updates,
                      elapsed_seconds=time.monotonic()-started,metrics=metrics))
                if updates%5==0:print('PPO',opt.mode,steps,'seconds',round(time.monotonic()-started),flush=True)
                if steps>=next_eval or steps>=opt.steps:
                    val=evaluate(model,envs,steps);write(out/f'validation_{steps}.json',val)
                    writer.writerow(dict(steps=steps,update=updates,seconds=time.monotonic()-started,
                                         final_success=val['final_success'],final_return=val['final_return']));f.flush()
                    score=(val['final_success'],val['final_return'])
                    if score>best:best=score;save('best_model.pt');write(out/'best_selection.json',val)
                    save('latest_model.pt');next_eval=(steps//opt.eval_steps+1)*opt.eval_steps
                    print('VALIDATION',opt.mode,steps,val['final_success'],val['final_return'],flush=True)
    save('final_model.pt');write(out/'summary.json',dict(steps=steps,updates=updates,
        elapsed_seconds=time.monotonic()-started,final_validation=val,selection='best checkpoint by fixed validation; no test selection'))
    for env in envs.values():env.close()
    assert_frozen_imports()


if __name__=='__main__':main()

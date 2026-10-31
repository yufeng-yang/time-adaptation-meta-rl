"""Three-episode diagnostic controller in real Genesis, not trained weights."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from layout import Config, PATTERNS, pattern_name
from environment import GateTrial
from simulation import make_batch, step_batch
from teacher import HistoryController


def run_trials(envs,patterns,deadline):
    world=envs[0].backend.world
    controllers=[HistoryController(env.config) for env in envs]
    deadlines=np.broadcast_to(np.asarray(deadline,dtype=float),(len(envs),))
    for env,p,d in zip(envs,patterns,deadlines): env.reset(deadline=d,doors=p)
    traces=[[] for _ in envs];events=[[] for _ in envs]
    for episode in range(3):
        for i,(env,controller) in enumerate(zip(envs,controllers)):
            if episode: env.next_episode()
            controller.start_episode()
            traces[i].append([])
        done=np.zeros(len(envs),bool)
        steps=0
        while not done.all():
            scan=world.sim.lidar_scan()
            actions=[controller.choose(env.backend.pose,env.carrying,{k:v[i] for k,v in scan.items()},env.elapsed,env.deadline)
                     if not done[i] else np.zeros(2)
                     for i,(env,controller) in enumerate(zip(envs,controllers))]
            out=step_batch(envs,actions,~done)
            for i,(env,result) in enumerate(zip(envs,out)):
                if result is None: continue
                traces[i][-1].append(env.backend.pose[:2].tolist())
                _,_,finished,info=result
                if finished:
                    done[i]=True
                    env.records[-1]['route']=controllers[i].route
                    env.records[-1]['known_gates_after']=controllers[i].known.copy()
            steps+=1
            if steps%40==0:
                print(f'Trial episode {episode+1}, simulated t={max(e.elapsed for e in envs):.0f}s, done={int(done.sum())}/{len(envs)}',flush=True)
        for i,c in enumerate(controllers): events[i].append(c.events.copy())
    return [dict(doors=pattern_name(p),deadline=float(d),records=e.records,gate_observations=v,trajectories=t)
            for e,p,d,v,t in zip(envs,patterns,deadlines,events,traces)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--doors',choices=('00','01','10','11','all'),default='01')
    parser.add_argument('--deadline',type=float,default=160.)
    parser.add_argument('--seed',type=int,default=10)
    parser.add_argument('--viewer',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Refusing overwrite')
    torch.set_num_threads(1)
    patterns=PATTERNS if args.doors=='all' else [tuple(c=='1' for c in args.doors)]
    backends=make_batch([args.seed]*len(patterns),show_viewer=args.viewer)
    envs=[GateTrial(b,seed=args.seed) for b in backends]
    start=time.monotonic()
    result=dict(metadata=envs[0].metadata(),controller='hand-specified waypoint controller with local-ray gate evidence and cross-episode memory; no learned high-level policy',
                trials=run_trials(envs,patterns,args.deadline),wall_seconds=time.monotonic()-start)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in t.items() if k!='trajectories'} for t in result['trials']],indent=2),flush=True)


if __name__=='__main__': main()

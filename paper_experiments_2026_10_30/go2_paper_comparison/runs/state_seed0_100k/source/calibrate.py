"""Measure physical route timings; oracle waypoints are calibration, not training."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from layout import Config, ROUTES, pattern_name, footprint_clearance
from simulation import GateWorld, goal_commands


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Refusing overwrite')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    config=Config()
    cases=[(route,doors,seed) for seed in range(args.repeats) for route,doors in
           [('fallback',(False,False)),('middle',(True,False)),('upper',(False,True)),('lower',(False,False))]]
    start=time.monotonic()
    world=GateWorld(len(cases),config)
    for i,(_,doors,seed) in enumerate(cases):
        world.set_doors(i,doors)
        world.reset(i,seed)
    paths=[]
    for route,_,_ in cases:
        out=list(ROUTES[route])
        paths.append(out+out[-2::-1]+[config.base])
    indices=np.zeros(len(cases),int)
    done=np.zeros(len(cases),bool)
    contacts=np.zeros(len(cases),int)
    failures=np.zeros(len(cases),bool)
    clearance=np.full(len(cases),np.inf)
    pickup=[None]*len(cases);arrival=[None]*len(cases)
    completion_pose=[None]*len(cases)
    traces=[[] for _ in cases]
    for step in range(320):
        goals=np.array([path[min(j,len(path)-1)] for path,j in zip(paths,indices)])
        commands=goal_commands(world.state['pose'],goals)
        commands[done]=0
        world.advance(commands,config.dt)
        for i,(route,doors,seed) in enumerate(cases):
            if done[i]: continue
            contacts[i]+=int(world.contacts[i])
            failures[i]|=world.state['failed'][i]
            for frame in world.trace:
                point=frame[i]
                traces[i].append(point.copy())
                clearance[i]=min(clearance[i],footprint_clearance(point,doors))
            distance=np.linalg.norm(world.state['pose'][i,:2]-goals[i])
            final_target=indices[i] in (len(ROUTES[route])-1,len(paths[i])-1)
            if distance < (.4 if final_target else .2):
                if indices[i]==len(ROUTES[route])-1: pickup[i]=(step+1)*config.dt
                indices[i]+=1
                if indices[i]==len(paths[i]):
                    done[i]=True
                    arrival[i]=(step+1)*config.dt
                    completion_pose[i]=world.state['pose'][i].copy()
            if failures[i]:
                done[i]=True
                completion_pose[i]=world.state['pose'][i].copy()
        if (step+1)%40==0:
            print(f'Physical route check: {(step+1)*config.dt:.0f}s simulated; {sum(v is not None for v in arrival)}/{len(cases)} returned',flush=True)
        if done.all(): break
    records=[]
    for i,(route,doors,seed) in enumerate(cases):
        record=dict(route=route,doors=pattern_name(doors),seed=seed,pickup_seconds=pickup[i],
                    return_seconds=arrival[i],motion_failed=bool(failures[i]),
                    wall_contact_decisions=int(contacts[i]),minimum_circle_clearance_m=float(clearance[i]),
                    final_pose=(completion_pose[i] if completion_pose[i] is not None else world.state['pose'][i]).tolist())
        records.append(record)
        print(record,flush=True)
    np.savez_compressed(args.output/'trajectories.npz',**{f'route_{i}':np.asarray(t) for i,t in enumerate(traces)})
    clean=all(r['return_seconds'] is not None and not r['motion_failed'] and r['wall_contact_decisions']==0 for r in records)
    result=dict(config=asdict(config),records=records,all_clean=clean,wall_seconds=time.monotonic()-start,
                provenance='Actual Genesis rigid dynamics and frozen Full_8000; no position snapping; no policy training',
                waypoint_tolerance_m=.2,final_tolerance_m=.4,
                caveat='Measured scripted-route feasibility, not globally shortest paths or real-world timing bounds',
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    if clean:
        maxima={route:max(r['return_seconds'] for r in records if r['route']==route) for route in ROUTES}
        result['route_max_seconds']=maxima
    (args.output/'calibration.json').write_text(json.dumps(result,indent=2)+'\n')
    if not clean: raise SystemExit('Route calibration found failures/contacts; repair geometry/controller before choosing deadlines')


if __name__=='__main__': main()

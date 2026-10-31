"""Actual physical gate visibility, collision and three-episode protocol checks."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from layout import Config, PATTERNS, GATES, PROBE_X, pattern_name
from environment import GateTrial
from simulation import make_batch, step_batch
from teacher import observe_gate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Refusing overwrite')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    config=Config()
    backends=make_batch([0]*4,config)
    envs=[GateTrial(b,config,i) for i,b in enumerate(backends)]
    initial=[env.reset(deadline=.5,doors=p) for env,p in zip(envs,PATTERNS)]
    for obs in initial[1:]:
        for key in initial[0]: np.testing.assert_array_equal(initial[0][key],obs[key])
    world=backends[0].world
    initial_scan=world.sim.lidar_scan()
    initial_visibility=[]
    for index,pattern in enumerate(PATTERNS):
        measured=[observe_gate({k:v[index] for k,v in initial_scan.items()},g,config.sensor_range) for g in range(2)]
        assert measured==[None,None],(pattern,measured)
        initial_visibility.append(measured)
    for index in range(1,4): np.testing.assert_array_equal(initial_scan['distances'][0],initial_scan['distances'][index])
    np.savez_compressed(args.output/'initial_input.npz',state=initial[0]['state'],grid=initial[0]['grid'],
                        **{k:v[0] for k,v in initial_scan.items()})
    visibility=[]
    for gate_index,box in enumerate(GATES):
        world.sim.robot.set_pos((PROBE_X,box.xy[1],.42))
        world.refresh_sensor()
        world._invalidate()
        scan=world.sim.lidar_scan()
        row=[]
        for index,p in enumerate(PATTERNS):
            measured=observe_gate({k:v[index] for k,v in scan.items()},gate_index,config.sensor_range)
            assert measured==p[gate_index],(index,gate_index,measured,p)
            row.append(measured)
        visibility.append(row)
        np.savez_compressed(args.output/f'probe_gate{gate_index}.npz',
                            **{k:v for k,v in scan.items()})
    # Three explicit episode resets preserve the trial task and refresh scans.
    for e,p in zip(envs,PATTERNS): e.reset(deadline=.5,doors=p)
    for episode in range(3):
        out=step_batch(envs,np.zeros((4,2)))
        assert all(done and reward==0. for _,reward,done,_ in out)
        assert all(info['trial_done']==(episode==2) for *_,info in out)
        np.testing.assert_array_equal(world.doors,PATTERNS)
        if episode<2:
            reset=[e.next_episode() for e in envs]
            for i,o in enumerate(reset): np.testing.assert_array_equal(o['grid'],initial[i]['grid'])
    # Intentional contact diagnostic: closed gates physically block the robot;
    # open gates allow passage. This is separate from clean navigation checks.
    for e,p in zip(envs,PATTERNS): e.reset(deadline=10.,doors=p)
    world.sim.robot.set_pos((PROBE_X,0.,.42))
    world.refresh_sensor();world._invalidate()
    contacts=np.zeros(4,bool)
    for _ in range(8):
        world.advance(np.tile([.4,0.,0.],(4,1)),.5)
        contacts|=world.contacts
    for i,p in enumerate(PATTERNS):
        if p[0]: assert world.state['pose'][i,0]>GATES[0].xy[0]+.3,world.state['pose']
        else:
            assert contacts[i],contacts
            assert world.state['pose'][i,0]<GATES[0].xy[0],world.state['pose']
    result=dict(config=asdict(config),patterns=[pattern_name(p) for p in PATTERNS],
                initial_state_vectors_identical=True,initial_grids_identical=True,initial_raw_scans_identical=True,
                initial_gate_open_measurements=initial_visibility,
                local_probe_open_measurements=visibility,task_persists_over_three_episodes=True,
                closed_gate_physical_contact=contacts.tolist(),push_test_final_pose=world.state['pose'].tolist(),
                note='Intentional push test is a collision diagnostic; it is not a successful-navigation rollout')
    (args.output/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()

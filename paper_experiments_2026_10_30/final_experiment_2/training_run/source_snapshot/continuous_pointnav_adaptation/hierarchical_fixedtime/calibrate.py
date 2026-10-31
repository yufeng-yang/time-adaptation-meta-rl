"""Verify physical demonstrations and deadline separation before training."""
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np

from continuous_pointnav_adaptation.hierarchical_fixedtime import scenario
from continuous_pointnav_adaptation.hierarchical_fixedtime.core import episode_records
from continuous_pointnav_adaptation.hierarchical_fixedtime.environment import (
    make_env,DEADLINES,COMMAND_DT,CALIBRATION_DIR,MAP_VERSION,MAP_X_LIMITS,SPAWN_POSITION,FIRST_EPISODE_DANGER_BONUS,FIRST_EPISODE_BONUS_DEADLINES,wall_locations,
)


def main():
    output=CALIBRATION_DIR;output.mkdir(exist_ok=True)
    routes=[];records=[];trial_paths=[];initial=[]
    for route in ('direct','upper','lower'):
        gate='open' if route=='direct' else 'closed'
        env=make_env(fixed_gate=gate)
        obs,_=env.reset(seed=20000,options=dict(gate_state=gate,deadline=350,new_trial=True))
        np.testing.assert_allclose(env.pose()[0],SPAWN_POSITION,atol=1e-6)
        teacher=scenario.DemonstrationTeacher(350,gate,route);positions=[env.pose()[0].tolist()]
        for step in range(350):
            obs,_,done,_,info=env.step(teacher.action(env,obs));positions.append(env.pose()[0].tolist())
            assert abs(info['physics_time']-(step+1)*COMMAND_DT)<1e-8
            if done:break
        assert info['success'] and info['collision_events']==0,(route,info)
        routes.append(dict(route=route,steps=step+1,seconds=(step+1)*COMMAND_DT,positions=positions,
                           collisions=info['collision_events'],danger_visits=info['danger_visits']))
        env.close()
    assert routes[0]['steps']<DEADLINES[0]<routes[1]['steps']<DEADLINES[1]<routes[2]['steps']<DEADLINES[2]
    for gate in ('open','closed'):
        env=make_env(fixed_gate=gate)
        obs,_=env.reset(seed=20000,options=dict(gate_state=gate,deadline=DEADLINES[1],new_trial=True));initial.append(obs)
        for deadline in DEADLINES:
            trial=scenario.collect_trial(None,env,gate,deadline,20000,teacher=True)
            for index,e in enumerate(trial.episodes):
                np.testing.assert_allclose(e.positions[0],SPAWN_POSITION,atol=1e-6)
                assert e.collisions==0,(gate,deadline,index,e.collisions)
                assert e.success==scenario.expected_success(gate,deadline,index),(gate,deadline,index,e.success)
                assert e.observed_gate==gate
                danger_reward=(FIRST_EPISODE_DANGER_BONUS if index==0 and deadline in FIRST_EPISODE_BONUS_DEADLINES else -env.danger_penalty)*e.danger_visits
                expected_return=(-env.step_penalty*e.steps+env.goal_reward*e.success
                                 -env.timeout_penalty*(not e.success)+danger_reward)
                np.testing.assert_allclose(e.rewards.sum(),expected_return,atol=1e-6)
                trial_paths.append(dict(gate=gate,deadline=deadline,episode=index+1,positions=e.positions.tolist()))
            result=episode_records(trial);records.extend(result)
            print('CALIBRATED',gate,deadline,[(r['steps'],r['success'],r['route']) for r in result],flush=True)
        env.close()
    np.testing.assert_allclose(initial[0],initial[1],atol=1e-6)
    # Extra-time diagnostic measures the complete probing upper route. This
    # does not change the real D=145 limit or its expected first-episode timeout.
    env=make_env(fixed_gate='closed')
    try:
        obs,_=env.reset(seed=20000,options=dict(gate_state='closed',deadline=350,new_trial=True))
        teacher=scenario.DemonstrationTeacher(DEADLINES[1]);positions=[env.pose()[0].tolist()]
        for step in range(350):
            obs,_,done,_,info=env.step(teacher.action(env,obs));positions.append(env.pose()[0].tolist())
            if done:break
        assert info['success'] and info['collision_events']==0
        actual=next(p for p in trial_paths if p['gate']=='closed' and p['deadline']==DEADLINES[1] and p['episode']==1)
        np.testing.assert_allclose(positions[:len(actual['positions'])],actual['positions'],atol=1e-6)
        probe_upper_completion=dict(teacher_deadline=DEADLINES[1],diagnostic_budget=350,
            steps=step+1,seconds=(step+1)*COMMAND_DT,success=True,collisions=0,positions=positions)
        print('PROBE_UPPER_COMPLETION',step+1,(step+1)*COMMAND_DT,flush=True)
    finally:env.close()
    (output/'results.json').write_text(json.dumps(dict(deadlines=DEADLINES,deadline_seconds=[d*.2 for d in DEADLINES],
        routes=routes,episodes=records,trial_paths=trial_paths,initial_heading_degrees=0.,map_version=MAP_VERSION,spawn_position=SPAWN_POSITION,
        demonstration_waypoints=scenario.ROUTES,
        probe_upper_return_waypoint=scenario.PROBE_UPPER_RETURN,
        reward_version='deadline_specific_ep1_danger_bonus_v2',first_episode_danger_bonus=FIRST_EPISODE_DANGER_BONUS,
        first_episode_bonus_deadlines=FIRST_EPISODE_BONUS_DEADLINES,probe_upper_completion=probe_upper_completion,
        gate_label_isolation=True,zero_speed_turns=True,command_period=.2,
        optimality='verified demonstrations, not global shortest-path claims'),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,3,figsize=(10,6),constrained_layout=True)
    for ax,route in zip(axes,routes):
        for x,y in wall_locations():ax.add_patch(Rectangle((x-.26,y-.26),.52,.52,color='#70518b',alpha=.5))
        xy=np.array(route['positions']);ax.plot(xy[:,0],xy[:,1],'.-',ms=2)
        ax.add_patch(plt.Circle((2.5,0),.3,color='green',alpha=.5))
        ax.set(xlim=MAP_X_LIMITS,ylim=(-6,4),title=f"{route['route']}: {route['steps']} commands / {route['seconds']:.1f}s")
        ax.set_aspect('equal')
    fig.suptitle('Physical route calibration | same learned low policy | zero-speed turns')
    fig.savefig(output/'routes.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()

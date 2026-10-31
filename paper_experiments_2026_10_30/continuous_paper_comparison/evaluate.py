"""Frozen-archive comparison on paired uniformly sampled control-step deadlines."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import numpy as np
import torch
from continuous_paper_comparison.runtime import (
    ARCHIVE,AblationPolicy,archive_config,make_env,core,scenario,assert_frozen_imports)

HERE=Path(__file__).resolve().parent
MODES=('state','time','context','full')
MODELS=ENVS=None


def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')


def checkpoint_path(mode):
    return ARCHIVE/'best_model.pt' if mode=='full' else HERE/'runs'/f'{mode}_seed0_matched500k'/'best_model.pt'


def tasks(seed=404132):
    rng=np.random.default_rng(seed)
    return [dict(trial=i,gate='open' if rng.integers(2)==0 else 'closed',
                 deadline=int(rng.integers(75,231)),seed=600000+i) for i in range(100)]


def route_choice(ep):
    route=ep.route
    if route=='other':
        xy=np.asarray(ep.positions)
        if xy[:,1].max()>1.2:return 'upper'
        if xy[:,1].min()<-1.2:return 'lower'
        if xy[:,0].max()>-2.:return 'direct'
    return route


def serialize(trial):
    rows=[]
    for i,ep in enumerate(trial.episodes):
        raw=float(np.sum(ep.rewards,dtype=np.float64));neutral=raw+.002*ep.steps
        cfg=archive_config()
        danger=(cfg['first_episode_danger_bonus'] if i==0 and trial.deadline in cfg['first_episode_bonus_deadlines']
                else -cfg['danger_penalty'])*ep.danger_visits
        expected=(1 if ep.success else -1)+danger-cfg['collision_penalty']*ep.collisions
        assert abs(neutral-expected)<2e-6,(neutral,expected)
        assert ep.steps<=trial.deadline and len(ep.actions)==len(ep.rewards)==ep.steps
        rows.append(dict(episode=i+1,raw_return=raw,deadline_return=neutral,steps=ep.steps,
                         success=ep.success,route=route_choice(ep),collisions=ep.collisions,
                         danger_visits=ep.danger_visits,actions=ep.actions.tolist(),
                         positions=ep.positions.tolist(),rewards=ep.rewards.tolist()))
    return rows


def worker_init():
    global MODELS,ENVS
    torch.set_num_threads(1)
    MODELS={}
    for mode in MODES:
        state=torch.load(checkpoint_path(mode),map_location='cpu',weights_only=False)
        model=AblationPolicy(mode=mode);model.load_state_dict(state['model_state_dict']);model.eval();MODELS[mode]=model
    ENVS={g:make_env(g) for g in ('open','closed')}
    assert_frozen_imports()


def job(payload):
    mode,task=payload
    trial=scenario.collect_trial(MODELS[mode],ENVS[task['gate']],task['gate'],task['deadline'],task['seed'],deterministic=True)
    return mode,dict(**task,episodes=serialize(trial))


def stats(values):
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(91035)
    samples=x[rng.integers(len(x),size=(5000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(samples,[.025,.975]).tolist())


def collect_candidates(reference,records):
    """Empirical feasible route library, never a shortest-time proof.

    Preserve the pre-test archive library, and separately expand it with any
    faster successful test route. Expansion avoids labeling a demonstrated
    feasible safe route as impossible. Policies/checkpoints are never updated.
    """
    bank={}
    for source,trials in [('archive_reference',reference)]+list(records.items()):
        for trial in trials:
            for ep in trial['episodes']:
                route=ep['route']
                if not ep['success'] or ep['collisions'] or route not in ('direct','upper','lower'):continue
                rewards=np.asarray(ep['rewards']);events=np.flatnonzero(np.isclose(np.abs(rewards+.002),.2,atol=1e-5)|
                    np.isclose(np.abs(rewards+.002),.02,atol=1e-5))
                candidate=dict(steps=ep['steps'],danger_step=int(events[0]+1) if ep['danger_visits'] and len(events) else None,
                               source=source,trial=trial.get('trial'),episode=ep['episode'],actions=ep['actions'])
                # A slower danger-free execution can dominate a faster hazardous
                # execution when the budget is sufficient; retain both regimes.
                variants=bank.setdefault(route,dict(variants=[]))['variants']
                same=[v for v in variants if (v['danger_step'] is None)==(candidate['danger_step'] is None)]
                if same and same[0]['steps']<=candidate['steps']:continue
                variants[:]=[v for v in variants if v not in same]+[candidate]
    assert set(bank)=={'direct','upper','lower'},bank
    return bank


def reference_decisions(gate,deadline,bank):
    values={}
    for route,item in bank.items():
        candidates=[]
        for variant in item.get('variants',[item]):
            feasible=variant['steps']<=deadline and (route!='direct' or gate=='open')
            danger=.2 if variant['danger_step'] is not None and variant['danger_step']<=deadline else 0.
            candidates.append((1. if feasible else -1.)-danger)
        values[route]=max(candidates)
    best=max(values.values())
    return [r for r,v in values.items() if abs(v-best)<1e-8],best>0,values


def summary(trials):
    return dict(trials=len(trials),
        trial_return=stats([sum(e['raw_return'] for e in t['episodes']) for t in trials]),
        final_episode_return=stats([t['episodes'][-1]['raw_return'] for t in trials]),
        odr=stats([t['optimal_decision'] for t in trials]),
        pretest_reference_odr=stats([t['pretest_optimal_decision'] for t in trials]),
        deadline_trial_return=stats([sum(e['deadline_return'] for e in t['episodes']) for t in trials]),
        deadline_final_episode_return=stats([t['episodes'][-1]['deadline_return'] for t in trials]),
        episode_returns=[float(np.mean([t['episodes'][i]['raw_return'] for t in trials])) for i in range(3)],
        final_success_rate=float(np.mean([t['episodes'][-1]['success'] for t in trials])))


def verify_archive():
    expected=json.loads((HERE/'archive_before.json').read_text())
    actual={str(p.relative_to(ARCHIVE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ARCHIVE.rglob('*') if p.is_file()}
    assert expected==actual,'Archive contents changed'
    return dict(archive_unchanged=True,verified_files=len(expected))


def evaluate(out,workers):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    verify_archive();test=tasks();write(out/'tasks.json',test)
    state=torch.load(ARCHIVE/'best_model.pt',map_location='cpu',weights_only=False)
    model=AblationPolicy();model.load_state_dict(state['model_state_dict']);model.eval()
    envs={g:make_env(g) for g in ('open','closed')};reference=[]
    for gate in ('open','closed'):
        for d in (75,145,230):
            trial=scenario.collect_trial(model,envs[gate],gate,d,20000,deterministic=True)
            reference.append(dict(gate=gate,deadline=d,episodes=serialize(trial)))
    for e in envs.values():e.close()
    pretest=collect_candidates(reference,{})
    write(out/'pretest_route_library.json',pretest);write(out/'archive_reference_trials.json',reference)
    records={m:[] for m in MODES};count=0
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=worker_init) as pool:
        for mode,trial in pool.map(job,[(m,t) for m in MODES for t in test]):
            records[mode].append(trial);count+=1
            if count%25==0:print('TEST',count,'/400 trials',flush=True)
    bank=collect_candidates(reference,records);write(out/'expanded_route_library.json',bank)
    results={}
    for mode,trials in records.items():
        for t in trials:
            routes,feasible,values=reference_decisions(t['gate'],t['deadline'],bank)
            previous,_,_=reference_decisions(t['gate'],t['deadline'],pretest)
            t.update(optimal_routes=routes,route_reference_values=values,feasible=feasible,
                     optimal_decision=t['episodes'][-1]['route'] in routes,
                     pretest_optimal_decision=t['episodes'][-1]['route'] in previous)
        s=summary(trials);saved=torch.load(checkpoint_path(mode),map_location='cpu',weights_only=False)
        s.update(checkpoint=str(checkpoint_path(mode)),checkpoint_sha256=hashlib.sha256(checkpoint_path(mode).read_bytes()).hexdigest(),
                 selected_training_steps=saved['total_steps'],training_budget=500000,
                 feasible_only=summary([t for t in trials if t['feasible']]),
                 gates={g:summary([t for t in trials if t['gate']==g]) for g in ('open','closed')})
        results[mode]=s;write(out/f'{mode}_trials.json',trials)
        print(mode,{k:s[k] for k in ('trial_return','final_episode_return','odr')},flush=True)
    verification=verify_archive();write(out/'verification.json',verification)
    write(out/'results.json',dict(methods=results,deadline_range=[75,230],deadline_unit='0.2s high-level commands',
        test_trials=100,training_seeds=1,task_seed=404132,map='Fixed archived geometry; no randomized gate positions',
        selection='Archive selected checkpoint unchanged; baselines selected by same six-case validation success/return criterion every10k steps',
        reward='Original archived rewards; episode1 danger bonus+0.02 only at D145, otherwise danger penalty0.2; step cost0.002',
        odr='Empirical validated-route reference, NOT a globally optimal physical-control oracle. Reference is expanded with successful test trajectories only for scoring; both pretest and expanded ODR are reported.',
        confidence='95% trial bootstrap,5000 samples; not independent training-seed uncertainty',verification=verification))
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['policy','trial_return','final_episode_return','validated_route_ODR','pretest_reference_ODR'])
        for mode,s in results.items():w.writerow([mode,s['trial_return']['mean'],s['final_episode_return']['mean'],s['odr']['mean'],s['pretest_reference_odr']['mean']])
    plot(out,results)


def plot(out,results):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/continuous_comparison_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['State-only','Time-aware','Context-only','Deadline-Aware (archive)']
    fig,ax=plt.subplots(figsize=(11,3));ax.axis('off');rows=[]
    for label,s in zip(labels,results.values()):
        cells=[label]
        for key in ('trial_return','final_episode_return','odr'):
            v=s[key];scale=100 if key=='odr' else 1
            cells.append(f"{v['mean']*scale:.3f} [{v['ci95'][0]*scale:.3f}, {v['ci95'][1]*scale:.3f}]")
        rows.append(cells)
    table=ax.table(cellText=rows,colLabels=['Policy','Trial return','Final-episode return','Validated-route ODR (%)'],
                   loc='center',cellLoc='center',colWidths=[.24,.25,.25,.26])
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.8)
    ax.set_title('Continuous navigation:100 paired trials | deadlines uniform75-230',pad=14)
    fig.text(.5,.04,'Means [95% trial-bootstrap CI]; fixed map; one training seed; original reward retained.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(out/'results_table.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for mode,label in zip(MODES,labels):
        path=ARCHIVE/'training_run/evaluation_curve.csv' if mode=='full' else HERE/'runs'/f'{mode}_seed0_matched500k'/'validation_curve.csv'
        with path.open() as f:points=list(csv.DictReader(f))
        xkey,ykey=('total_steps','third_episode_mean_return') if mode=='full' else ('steps','final_return')
        ax.plot([float(p[xkey])/1000 for p in points],[float(p[ykey]) for p in points],label=label)
    ax.set(xlabel='Collected high-level transitions (thousands)',ylabel='Mean final-episode raw return',
           title='Continuous navigation: fixed six-case validation')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=180);plt.close(fig)
    from continuous_pointnav_adaptation.hierarchical_fixedtime.environment import wall_locations
    from matplotlib.patches import Rectangle
    all_trials={m:json.loads((out/f'{m}_trials.json').read_text()) for m in MODES}
    closed=[t for t in all_trials['full'] if t['gate']=='closed']
    chosen=[min(closed,key=lambda t:abs(t['deadline']-d)) for d in (145,230)]
    fig,axes=plt.subplots(2,4,figsize=(14,9),sharex=True,sharey=True)
    for row,task in enumerate(chosen):
        for col,(mode,label) in enumerate(zip(MODES,labels)):
            ax=axes[row,col];trial=next(t for t in all_trials[mode] if t['trial']==task['trial'])
            for x,y in wall_locations():ax.add_patch(Rectangle((x-.26,y-.26),.52,.52,color='.8'))
            ax.add_patch(Rectangle((-.26,-.52),.52,1.04,color='.4'))
            ax.add_patch(Rectangle((-.49,2.75-.49),.98,.98,color='#cc9c8b',alpha=.6))
            ax.add_patch(plt.Circle((2.5,0),.3,color='#9cbfa8'))
            for ep in trial['episodes']:
                xy=np.asarray(ep['positions']);ax.plot(xy[:,0],xy[:,1],lw=1.2,
                    label=f"Ep{ep['episode']}: {ep['raw_return']:.2f}")
            ax.set(xlim=(-5.05,3.55),ylim=(-6,4),aspect='equal',title=f"{label}\nClosed, D={task['deadline']}",xlabel='x (m)')
            ax.legend(fontsize=7,loc='lower left')
            if col==0:ax.set_ylabel('y (m)')
    fig.tight_layout();fig.savefig(out/'closed_trial_paths.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();evaluate(a.output,a.workers)

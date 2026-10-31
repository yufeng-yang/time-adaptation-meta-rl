"""Paired random-deadline evaluation with an exact ideal-option target oracle."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from pick_place_paper_comparison.runtime import ARCHIVE,AblationPolicy,archived_env,engine,assert_frozen_imports

HERE=Path(__file__).resolve().parent
MODES=('state','time','context','full')


def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')


def tasks(seed=404133):
    rng=np.random.default_rng(seed)
    return [dict(trial=i,deadline=int(rng.integers(30,151)),seed=700000+i) for i in range(100)]


def oracle(env):
    # At most one ball can be delivered per episode. Direct delivery minimizes
    # duration for each target: inspections add waits and vertical travel, while
    # horizontal detours cannot beat the direct segment (triangle inequality).
    durations=[env.duration(4+i,env.home) for i in range(5)]
    feasible=[i for i,cost in enumerate(durations) if cost<=env.deadline]
    best=max(float(env._scores[i]) for i in feasible)
    return dict(direct_delivery_steps=durations,max_score=best,max_return=best/env.config.score_scale,
                optimal_targets=[archived_env.LABELS[i] for i in feasible if abs(env._scores[i]-best)<1e-9])


def selected_target(record):
    return next((a.removeprefix('deliver_') for a in reversed(record['actions']) if a.startswith('deliver_')),None)


def stats(values):
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(91036)
    draws=a[rng.integers(len(a),size=(5000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),ci95=np.quantile(draws,[.025,.975]).tolist())


def summary(trials):
    return dict(trials=len(trials),
        trial_return=stats([sum(e['return'] for e in t['episodes']) for t in trials]),
        final_episode_return=stats([t['episodes'][-1]['return'] for t in trials]),
        odr=stats([t['optimal_decision'] for t in trials]),
        trial_score=stats([sum(e['score'] for e in t['episodes']) for t in trials]),
        final_episode_score=stats([t['episodes'][-1]['score'] for t in trials]),
        delivered_optimal_rate=stats([t['delivered_optimal'] for t in trials]),
        episode_returns=[float(np.mean([t['episodes'][i]['return'] for t in trials])) for i in range(3)],
        final_delivery_rate=float(np.mean([t['episodes'][-1]['delivered'] for t in trials])))


def check_archive():
    before=json.loads((HERE/'archive_before.json').read_text())
    after={str(q.relative_to(ARCHIVE)):hashlib.sha256(q.read_bytes()).hexdigest() for q in ARCHIVE.rglob('*') if q.is_file()}
    assert before==after,'Archive changed'
    return dict(archive_unchanged=True,verified_files=len(before))


def checkpoint_path(mode):
    return ARCHIVE/'half_bc/best.pt' if mode=='full' else HERE/'runs'/f'{mode}_seed0_200updates'/'best.pt'


def evaluate(out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);check_archive()
    test=tasks();write(out/'tasks.json',test);results={};records={}
    for mode in MODES:
        path=checkpoint_path(mode);saved=torch.load(path,map_location='cpu',weights_only=False)
        model=AblationPolicy(mode);model.load_state_dict(saved['model']);model.eval();trials=[]
        for task in test:
            env=archived_env.PickPlaceTrial(seed=task['seed'])
            trial=engine.collect_trial(model,env,task['deadline'],deterministic=True)
            truth=oracle(env);eps=[]
            for ep,record in zip(trial['episodes'],trial['records']):
                value=float(sum(ep['rewards']))
                assert value==record['score']/env.config.score_scale
                assert sum(ep['durations'])==record['low_steps']<=task['deadline']
                eps.append(dict(**record,**{'return':value},target=selected_target(record),
                                action_indices=ep['actions'],durations=ep['durations'],rewards=ep['rewards']))
            assert len(eps)==3
            trials.append(dict(**task,scores=trial['scores'],positions=trial['positions'],oracle=truth,episodes=eps,
                optimal_decision=eps[-1]['target'] in truth['optimal_targets'],
                delivered_optimal=eps[-1]['delivered'] and eps[-1]['ball'] in truth['optimal_targets']))
        s=summary(trials);s.update(checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                 selected_phase=saved['phase'],selected_update=saved['update'],selected_low_steps=saved['low_steps'])
        s['deadline_bands']={}
        for name,low,high in [('short',30,36),('medium',37,90),('long',91,150)]:
            subset=[t for t in trials if low<=t['deadline']<=high]
            if subset:s['deadline_bands'][name]=summary(subset)
        write(out/f'{mode}_trials.json',trials);results[mode]=s;records[mode]=trials
        print(mode,{k:s[k] for k in ('trial_return','final_episode_return','odr')},flush=True)
    assert_frozen_imports();verified=check_archive();write(out/'verification.json',verified)
    write(out/'results.json',dict(methods=results,deadline_range=[30,150],test_trials=100,task_seed=404133,
        training_seeds=1,environment='Archived nine ideal Cartesian options; not physical manipulation or the newer8-action primitive environment',
        reward='Delivered score/4, gamma1; raw score companions also provided',
        selection='Original half_bc/best.pt; three fresh baselines use same validation mean-trial-score criterion',
        oracle='Full-information maximum score among targets directly deliverable from episode home within D; only evaluation sees hidden scores',
        odr='Final attempted delivery target belongs to oracle optimal target set; no delivery choice is incorrect. Delivered-optimal rate is separately reported.',
        uncertainty='95% paired-task bootstrap intervals over100 trials; not training-seed uncertainty',verification=verified))
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['policy','trial_return','final_episode_return','ODR','trial_raw_score','final_raw_score'])
        for mode,s in results.items():w.writerow([mode,s['trial_return']['mean'],s['final_episode_return']['mean'],s['odr']['mean'],s['trial_score']['mean'],s['final_episode_score']['mean']])
    differences={}
    for mode in ('state','time','context'):
        differences[mode]=dict(
            trial_return=stats([sum(e['return'] for e in a['episodes'])-sum(e['return'] for e in b['episodes']) for a,b in zip(records['full'],records[mode])]),
            final_episode_return=stats([a['episodes'][-1]['return']-b['episodes'][-1]['return'] for a,b in zip(records['full'],records[mode])]),
            odr=stats([int(a['optimal_decision'])-int(b['optimal_decision']) for a,b in zip(records['full'],records[mode])]))
    write(out/'paired_full_minus_baseline.json',differences);plot(out,results,records)


def plot(out,results,records):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/pick_place_comparison_matplotlib')
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
    table=ax.table(cellText=rows,colLabels=['Policy','Trial return','Final-episode return','ODR (%)'],loc='center',cellLoc='center',colWidths=[.24,.25,.25,.26])
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.8)
    ax.set_title('Pick-and-place: 100 paired trials | deadlines uniform30-150',pad=14)
    fig.text(.5,.04,'Means [95% trial-bootstrap CI]; reward = delivered value /4; nine ideal high-level options; one seed.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(out/'results_table.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for mode,label in zip(MODES,labels):
        path=ARCHIVE/'half_bc/progress.json' if mode=='full' else HERE/'runs'/f'{mode}_seed0_200updates'/'progress.json'
        rows=json.loads(path.read_text());x=[];y=[]
        if mode=='full':
            for row in rows:
                validation=ARCHIVE/'half_bc'/f"validation_{row['update']:04d}.json"
                if validation.exists():
                    values=json.loads(validation.read_text())
                    x.append(row['low_steps']);y.append(float(np.mean([v['episodes']['3']['score']/4 for v in values.values()])))
        else:
            for row in rows:
                if 'validation' in row:
                    x.append(row['steps']);y.append(float(np.mean([v['episodes']['3']['score']/4 for v in row['validation'].values()])))
        ax.plot(np.asarray(x)/1000,y,label=label)
    ax.set(xlabel='Collected low-level steps (thousands)',ylabel='Mean final-episode reward',title='Pick-and-place: fixed-deadline validation')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(12,3.5),sharey=True)
    for ax,(mode,s),label in zip(axes,results.items(),labels):
        xs=[t['deadline'] for t in records[mode]];ys=[t['episodes'][-1]['score'] for t in records[mode]]
        ax.scatter(xs,ys,alpha=.6,s=18);ax.set(title=label,xlabel='Deadline (steps)',ylim=(-.1,4.2))
    axes[0].set_ylabel('Final delivered value');fig.tight_layout();fig.savefig(out/'deadline_outcomes.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);evaluate(p.parse_args().output)

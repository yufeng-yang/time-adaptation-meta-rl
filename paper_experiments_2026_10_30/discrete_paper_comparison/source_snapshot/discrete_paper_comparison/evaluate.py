"""Paired 100-trial evaluation; preserve original rewards and audit route ODR."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from discrete_paper_comparison.train import AblationPolicy, engine

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CASES=[(gate,d) for gate in ('open','closed') for d in (15,17,25)]
FULL=ROOT/'tidyup_plan2/baseline3_gate_meta_rl/episode_transformer_bc/release_0_80w/final_model.pt'


def write(path,data): path.write_text(json.dumps(data,indent=2)+'\n')


def task_set(seed=404131,count=100):
    rng=np.random.default_rng(seed)
    return [dict(trial=k,gate=CASES[int(i)][0],deadline=CASES[int(i)][1],seed=500000+k)
            for k,i in enumerate(rng.integers(0,len(CASES),count))]


def selected_route(positions):
    """Last departure from fork, invalidated on return; no action-value claims."""
    route=None
    for previous,current in zip(positions,positions[1:]):
        previous,current=tuple(previous),tuple(current)
        if current==(3,4) and previous!=(3,4): route=None
        if previous==(3,4) and current!=previous:
            route={(4,4):'direct',(3,3):'medium',(3,5):'detour'}.get(current)
    return route


def oracle_routes(args):
    """Oracle over three canonical route plans, using time-neutral task returns.

    A known blocked direct route stops before collision and waits out the budget.
    Thus infeasible closed-D15 has direct/safe tied at -1, danger at -1.2.
    This is task-level route optimality, not globally optimal primitive control.
    """
    env=engine.make_env(args); result={}
    for gate,deadline in CASES:
        values={}
        for route in ('direct','medium','detour'):
            env.reset(seed=123,options=dict(new_trial=True,gate_state=gate,deadline=deadline))
            script=engine.base.route_actions(route);raw=0.;step=0;done=False;waiting=False
            while not done:
                action=script[step] if step<len(script) and not waiting else step%2
                base=env.unwrapped
                if action==2:
                    dx,dy=((1,0),(0,1),(-1,0),(0,-1))[int(base.agent_dir)]
                    if not base._is_traversable(base.agent_pos[0]+dx,base.agent_pos[1]+dy):
                        waiting=True;action=step%2
                _,reward,done,trunc,info=env.step(action);done=done or trunc;raw+=reward;step+=1
            values[route]=round(raw+args.step_penalty*step,8)
        maximum=max(values.values())
        result[f'{gate}_D{deadline}']=dict(route_values=values,
            optimal_routes=[r for r,v in values.items() if abs(v-maximum)<1e-7],
            feasible=maximum>0)
    env.close();return result


def audit_episode(env,task,ep,index,step_penalty):
    env.reset(seed=task['seed'] if index==0 else None,
              options=dict(new_trial=index==0,gate_state=task['gate'],deadline=task['deadline']))
    positions=[list(map(int,env.unwrapped.agent_pos))];rewards=[];collisions=0;done=False
    for action in ep.actions:
        _,reward,done,truncated,info=env.step(int(action));done=done or truncated
        positions.append(list(map(int,env.unwrapped.agent_pos)));rewards.append(reward)
        collisions+=int(info['collision'])
    assert done and len(rewards)==ep.steps and abs(sum(rewards)-ep.episode_return)<2e-6
    score=sum(rewards)+step_penalty*ep.steps
    event_score=(1 if info['success'] else -1)-info['danger_cost']-.05*collisions
    assert abs(score-event_score)<1e-7
    return dict(episode=index+1,raw_return=sum(rewards),deadline_return=score,
                success=ep.success,steps=ep.steps,route=ep.route,
                selected_route=selected_route(positions),probed_gate=ep.probed_gate,
                collisions=collisions,danger_cost=info['danger_cost'],
                actions=ep.actions.tolist(),positions=positions,rewards=rewards)


def statistics(values):
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(91034)
    sample=x[rng.integers(len(x),size=(5000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(sample,[.025,.975]).tolist())


def summarize(trials):
    return dict(trials=len(trials),
        trial_return=statistics([sum(e['raw_return'] for e in t['episodes']) for t in trials]),
        final_episode_return=statistics([t['episodes'][-1]['raw_return'] for t in trials]),
        odr=statistics([t['optimal_decision'] for t in trials]),
        deadline_trial_return=statistics([sum(e['deadline_return'] for e in t['episodes']) for t in trials]),
        deadline_final_episode_return=statistics([t['episodes'][-1]['deadline_return'] for t in trials]),
        per_episode_return=[float(np.mean([t['episodes'][i]['raw_return'] for t in trials])) for i in range(3)],
        final_success_rate=float(np.mean([t['episodes'][-1]['success'] for t in trials])))


def evaluate(output):
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    tasks=task_set();write(output/'tasks.json',tasks)
    args=engine.parse_args([]);oracle=oracle_routes(args);write(output/'oracle.json',oracle)
    summaries={};all_records={}
    for mode in ('state','time','context','full'):
        path=FULL if mode=='full' else HERE/'runs'/f'{mode}_seed0_800k'/'final_model.pt'
        state=torch.load(path,map_location='cpu',weights_only=False)
        cfg=argparse.Namespace(**state['config']);cfg.device='cpu'
        assert (cfg.gamma,cfg.step_penalty,cfg.bc_decay_steps)==(.99,.01,600000)
        model=AblationPolicy(cfg.hidden_dim,cfg.transformer_layers,cfg.attention_heads,mode)
        model.load_state_dict(state['model_state_dict']);model.eval()
        env=engine.make_env(cfg);audit_env=engine.make_env(cfg);records=[]
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        with torch.no_grad():
            for task in tasks:
                trial=engine.base.collect_trial(model,env,task['gate'],task['seed'],
                    state['total_steps'],cfg,torch.device('cpu'),deterministic=True,
                    trial_deadline=task['deadline'])
                eps=[audit_episode(audit_env,task,ep,i,cfg.step_penalty) for i,ep in enumerate(trial.episodes)]
                route=eps[-1]['selected_route'];reference=oracle[f"{task['gate']}_D{task['deadline']}"]
                records.append(dict(**task,episodes=eps,optimal_routes=reference['optimal_routes'],
                                    feasible=reference['feasible'],optimal_decision=route in reference['optimal_routes']))
        env.close();audit_env.close()
        assert hashlib.sha256(path.read_bytes()).hexdigest()==before
        summary=summarize(records)
        summary.update(checkpoint=str(path),checkpoint_sha256=before,training_steps=state['total_steps'],
                       training_seed=cfg.seed,
                       cases={f'{g}_D{d}':summarize([t for t in records if t['gate']==g and t['deadline']==d])
                              for g,d in CASES if any(t['gate']==g and t['deadline']==d for t in records)},
                       feasible_only=summarize([t for t in records if t['feasible']]))
        write(output/f'{mode}_trials.json',records);summaries[mode]=summary;all_records[mode]=records
        print(mode,{k:summary[k] for k in ('trial_return','final_episode_return','odr')},flush=True)
    paired_differences={}
    full=all_records['full']
    for mode in ('state','time','context'):
        other=all_records[mode]
        paired_differences[mode]=dict(
            trial_return=statistics([sum(e['raw_return'] for e in a['episodes'])-
                                     sum(e['raw_return'] for e in b['episodes']) for a,b in zip(full,other)]),
            final_episode_return=statistics([a['episodes'][-1]['raw_return']-b['episodes'][-1]['raw_return']
                                             for a,b in zip(full,other)]),
            odr=statistics([int(a['optimal_decision'])-int(b['optimal_decision']) for a,b in zip(full,other)]))
    write(output/'paired_full_minus_baseline.json',paired_differences)
    write(output/'results.json',dict(methods=summaries,case_counts=dict(Counter(f"{t['gate']}_D{t['deadline']}" for t in tasks)),
        task_seed=404131,training_seeds=1,selection='Fixed final checkpoints; no test selection',
        primary_return='Undiscounted original rewards including 0.01 per-action cost',
        deadline_return='Separate post-hoc score = raw return + 0.01*steps; original rewards remain inputs to context',
        odr='Last departure from fork in final episode; oracle ties over direct/danger/safe canonical routes using time-neutral score. No departure is incorrect. Infeasible closed-D15 included, with feasible-only results also saved.',
        ci='5000 trial-level bootstrap resamples; task-mixture uncertainty, not independent training-seed uncertainty',
        limitation='Fixed map and six configurations only; 100 sampled trials do not imply 100 different maps. Missing inputs masked in actor and trajectory encoder; inactive context modules retained for shape compatibility.'))
    with (output/'metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['method','trial_return','final_episode_return','ODR','training_steps'])
        for mode,s in summaries.items():writer.writerow([mode,s['trial_return']['mean'],s['final_episode_return']['mean'],s['odr']['mean'],s['training_steps']])
    plot(output,summaries,all_records)


def plot(output,summaries,records):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/discrete_paper_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['State-only','Time-aware','Context-only','Deadline-Aware']
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key,title in zip(axes,('trial_return','final_episode_return','odr'),
                            ('Trial return','Final-episode return','Optimal decision rate')):
        data=[s[key] for s in summaries.values()];means=np.array([s['mean'] for s in data])
        low=np.array([s['ci95'][0] for s in data]);high=np.array([s['ci95'][1] for s in data])
        ax.bar(range(4),means,color=['.65','.65','.65','#39775c'])
        ax.errorbar(range(4),means,yerr=[np.maximum(0,means-low),np.maximum(0,high-means)],fmt='none',color='black',capsize=3)
        ax.set_xticks(range(4),labels,rotation=25,ha='right');ax.set_title(title);ax.axhline(0,color='.3',lw=.6)
        ax.set_ylabel('Fraction of trials' if key=='odr' else 'Undiscounted original reward')
        if key=='odr': ax.set_ylim(0,1.05)
    fig.suptitle('Discrete navigation | 100 paired test trials | one training seed')
    fig.text(.5,.015,'Error bars: 95% trial-bootstrap CI. Returns retain step costs; ODR uses time-neutral route optimality.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.93));fig.savefig(output/'comparison.png',dpi=180);fig.savefig(output/'comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(13,3.6),sharey=True)
    colors={'direct':'#39775c','medium':'#bf7848','detour':'#587daa',None:'.7'}
    for ax,(mode,trials),label in zip(axes,records.items(),labels):
        for j,(g,d) in enumerate(CASES):
            subset=[t for t in trials if t['gate']==g and t['deadline']==d]
            counts=Counter(t['episodes'][-1]['selected_route'] for t in subset);bottom=0
            for route in ('direct','medium','detour',None):
                value=counts[route]/len(subset);ax.bar(j,value,bottom=bottom,color=colors[route],label=route or 'none' if j==0 else None);bottom+=value
        ax.set_xticks(range(6),[f'{g[0].upper()}{d}' for g,d in CASES],rotation=45)
        ax.set_title(label);ax.set_xlabel('Gate / deadline configuration')
    axes[0].set_ylabel('Final-episode decision frequency');axes[-1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'route_choices.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,2.7));ax.axis('off')
    rows=[]
    for label,s in zip(labels,summaries.values()):
        cells=[label]
        for key in ('trial_return','final_episode_return','odr'):
            value=s[key];scale=100 if key=='odr' else 1
            cells.append(f"{value['mean']*scale:.3f} [{value['ci95'][0]*scale:.3f}, {value['ci95'][1]*scale:.3f}]")
        rows.append(cells)
    table=ax.table(cellText=rows,colLabels=['Policy','Trial return','Final-episode return','ODR (%)'],
                   loc='center',cellLoc='center',colWidths=[.2,.27,.27,.26])
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.65)
    ax.set_title('Discrete navigation: 100 paired random trials',pad=12)
    fig.text(.5,.04,'Means [95% trial-bootstrap CI]; one training seed. Original step costs retained in returns.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(output/'results_table.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for mode,label in zip(summaries,labels):
        path=FULL.parent/'evaluation_curve.csv' if mode=='full' else HERE/'runs'/f'{mode}_seed0_800k'/'evaluation_curve.csv'
        with path.open() as f: points=list(csv.DictReader(f))
        ax.plot([int(p['total_steps'])/1000 for p in points],
                [float(p['third_episode_mean_return']) for p in points],label=label)
    ax.axvline(600,color='.5',linestyle=':',lw=1,label='BC reaches zero')
    ax.set(xlabel='Collected environment steps (thousands)',ylabel='Mean final-episode raw return',
           title='Discrete navigation: fixed six-case validation during training')
    ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'training_curves.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    evaluate(parser.parse_args().output)

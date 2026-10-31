"""Paired held-out comparison; no unsupported global-optimum labels."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime_source/gates'))
from layout import Config
from model import MetaPolicy
from train import make_env_batch,collect_batch,save_json
ARCHIVE=ROOT.parent/'deadline_go2_gates/runs/half_bc_steps100k_seed0'
def main():
    torch.set_num_threads(1)
    out=ROOT/'evaluation_100trials_seed404134'
    out.mkdir(exist_ok=False)
    rng=np.random.default_rng(404134)
    tasks=[dict(deadline=float(rng.integers(36,201)/2),doors=(rng.random(2)<.5).tolist(),seed=800000+i) for i in range(100)]
    save_json(out/'tasks.json',tasks)
    results={}
    for mode in ('state','time','context','full'):
        path=ARCHIVE/'best.pt' if mode=='full' else ROOT/'runs'/f'{mode}_seed0_100k'/'best.pt'
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=MetaPolicy(mode=mode).cuda();model.load_state_dict(saved['model']);model.eval()
        records=[]
        # Keep the original 12-environment physical batching, including the final padded batch.
        for start in range(0,len(tasks),12):
            block=tasks[start:start+12];padded=block+[block[-1]]*(12-len(block))
            envs=make_env_batch(Config(),[t['seed'] for t in padded])
            trials=collect_batch(model,envs,[t['deadline'] for t in padded],patterns=[t['doors'] for t in padded],deterministic=True)
            for task,trial in zip(block,trials):
                for ep,record in zip(trial['episodes'],trial['records']):
                    assert abs(sum(ep['rewards'])-record['delivered_value']/4)<1e-6
                records.append(dict(**task,records=trial['records'],steps=trial['steps']))
            print('EVAL',mode,len(records),flush=True)
        save_json(out/f'{mode}_trials.json',records)
        trial=np.array([sum(r['reward'] for r in t['records']) for t in records])
        final=np.array([t['records'][-1]['reward'] for t in records])
        boot=np.random.default_rng(404135).integers(0,100,(5000,100))
        row={}
        for name,values in [('trial_return',trial),('final_episode_return',final)]:
            row[name]=dict(mean=float(values.mean()),ci95=np.quantile(values[boot].mean(1),[.025,.975]).tolist())
        row['final_delivery_rate']=float(np.mean([t['records'][-1]['delivered'] for t in records]))
        row['checkpoint']=str(path);row['selected_phase']=saved['phase'];row['selected_update']=saved['update']
        results[mode]=row
        save_json(out/'results.json',results)
    # Independently retain the original three training deadlines on all four patterns.
    from train import evaluate
    fixed={}
    for mode in ('state','time','context','full'):
        path=ARCHIVE/'best.pt' if mode=='full' else ROOT/'runs'/f'{mode}_seed0_100k'/'best.pt'
        model=MetaPolicy(mode=mode).cuda();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);model.eval()
        fixed[mode],_=evaluate(model,Config(),seed=900000)
    save_json(out/'fixed_deadline_results.json',fixed)
    before=json.loads((ROOT/'archive_before.json').read_text())
    current={str(p.relative_to(ARCHIVE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ARCHIVE.rglob('*') if p.is_file()}
    assert before==current,'Protected archive changed'
    save_json(out/'verification.json',dict(archive_unchanged=True,verified_files=len(before),audited_episodes=1200))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    labels=['State-only','Time-aware','Context-only','Deadline-Aware']
    for ax,key,title in zip(axes,['trial_return','final_episode_return','final_delivery_rate'],['Trial return','Final-episode return','Final delivery rate']):
        vals=[results[m][key]['mean'] if isinstance(results[m][key],dict) else results[m][key] for m in results]
        ax.bar(range(4),vals,color=['#99a5b5','#6c9bd2','#81b7a0','#d89b47'])
        if key!='final_delivery_rate':
            cis=np.array([results[m][key]['ci95'] for m in results]);ax.errorbar(range(4),vals,yerr=[np.array(vals)-cis[:,0],cis[:,1]-np.array(vals)],fmt='none',c='black',capsize=3)
        ax.set_xticks(range(4),labels,rotation=22,ha='right');ax.set_title(title)
    fig.suptitle('Go2 gate delivery: 100 paired trials, 3 episodes/trial, single training seed')
    fig.tight_layout();fig.savefig(out/'results.png',dpi=180)
if __name__=='__main__':main()

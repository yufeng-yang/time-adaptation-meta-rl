"""Fresh matched ablations; never modify the archive's policies or runtime."""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
from pick_place_paper_comparison.runtime import (
    ARCHIVE,AblationPolicy,archived_env,engine,assert_frozen_imports)


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('state','time','context'),required=True)
    parser.add_argument('--output',type=Path,required=True);opt=parser.parse_args()
    cfg=json.loads((ARCHIVE/'half_bc/config.json').read_text());out=opt.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cfg.update(policy_mode=opt.mode,output_dir=str(out),bc_aux_initial_weight=.2,bc_decay_updates=100,
               learning_rate_bc=1e-3,learning_rate_ppo=3e-4,ppo_epochs=4,ppo_minibatches=2,
               entropy_coefficient=.01,value_coefficient=.5,clip_range=.2,max_grad_norm=.5,
               seed=0,baseline_initialization='fresh weights; missing time/context/history masked')
    write(out/'config.json',cfg)
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0);random.seed(0);rng=np.random.default_rng(0)
    task=archived_env.TaskConfig();started=time.monotonic();steps=0
    experts=[engine.collect_trial(None,archived_env.PickPlaceTrial(task,100000+i),task.deadlines[i%3],teacher=True)
             for i in range(cfg['expert_trials'])]
    write(out/'expert_summary.json',engine.summarize(experts))
    write(out/'expert_trials.json',[{k:t[k] for k in ('deadline','scores','positions','records')} for t in experts])
    model=AblationPolicy(opt.mode);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    def save(name,phase,update):
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),phase=phase,update=update,
                        low_steps=steps,config=cfg,rng_state=dict(torch=torch.get_rng_state(),numpy=rng.bit_generator.state,
                                                               python=random.getstate())),out/name)
    save('initial.pt','initial',0)
    for update in range(cfg['bc_updates']):
        demos=[experts[int(i)] for i in rng.integers(len(experts),size=cfg['batch_trials'])]
        replay=engine.replay(model,demos)
        loss=F.cross_entropy(replay['logits'],replay['actions'])+.1*F.mse_loss(replay['values'],replay['returns'])
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
        if (update+1)%100==0:print('BC',opt.mode,update+1,round(time.monotonic()-started),flush=True)
    save('bc.pt','bc',cfg['bc_updates'])
    summary,_=engine.evaluate(model,task,cfg['eval_trials'],200000)
    write(out/'bc_validation.json',summary);best=float(np.mean([v['mean_trial_score'] for v in summary.values()]))
    save('best.pt','bc',cfg['bc_updates']);write(out/'best_selection.json',dict(phase='bc',update=cfg['bc_updates'],summary=summary))
    for group in optimizer.param_groups:group['lr']=3e-4
    progress=[]
    for update in range(cfg['ppo_updates']):
        trials=[engine.collect_trial(model,archived_env.PickPlaceTrial(task,int(rng.integers(1000000000))))
                for _ in range(cfg['batch_trials'])]
        steps+=sum(t['low_steps'] for t in trials)
        adv=np.concatenate([e['advantages'] for t in trials for e in t['episodes']]);mean,std=adv.mean(),adv.std()+1e-8
        for t in trials:
            for ep in t['episodes']:ep['advantages']=(ep['advantages']-mean)/std
        weight=.2*max(0.,1-(update+1)/100)
        for _ in range(4):
            for ids in np.array_split(rng.permutation(len(trials)),2):
                replay=engine.replay(model,[trials[int(i)] for i in ids]);dist=Categorical(logits=replay['logits'])
                ratio=(dist.log_prob(replay['actions'])-replay['old_log_probs']).exp()
                actor=-torch.minimum(ratio*replay['advantages'],ratio.clamp(.8,1.2)*replay['advantages']).mean()
                loss=actor+.5*F.mse_loss(replay['values'],replay['returns'])-.01*dist.entropy().mean()
                if weight:
                    demos=[experts[int(i)] for i in rng.integers(len(experts),size=max(4,len(ids)//2))]
                    d=engine.replay(model,demos);loss+=weight*F.cross_entropy(d['logits'],d['actions'])
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
        row=dict(update=update+1,steps=steps,bc_weight=weight,elapsed_seconds=time.monotonic()-started)
        if (update+1)%cfg['eval_every']==0 or update+1==cfg['ppo_updates']:
            summary,_=engine.evaluate(model,task,cfg['eval_trials'],200000);row['validation']=summary
            write(out/f'validation_{update+1:04d}.json',summary)
            metric=float(np.mean([v['mean_trial_score'] for v in summary.values()]))
            if metric>best:
                best=metric;save('best.pt','ppo',update+1);write(out/'best_selection.json',dict(phase='ppo',update=update+1,summary=summary))
            save('latest.pt','ppo',update+1)
            print('PPO',opt.mode,update+1,steps,'validation',round(metric,4),'seconds',round(time.monotonic()-started),flush=True)
        progress.append(row);write(out/'progress.json',progress)
    save('final.pt','ppo',cfg['ppo_updates'])
    write(out/'summary.json',dict(updates=cfg['ppo_updates'],steps=steps,elapsed_seconds=time.monotonic()-started,
                                 selection='same mean-trial-score validation as archive; no test selection'))
    assert_frozen_imports()


if __name__=='__main__':main()

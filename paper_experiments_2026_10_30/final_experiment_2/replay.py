"""Reproduce the selected best checkpoint using only this archive's local code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
RUNTIME=HERE/'runtime_source'
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
sys.dont_write_bytecode=True
sys.path.insert(0,str(RUNTIME))

import numpy as np
import torch
from continuous_pointnav_adaptation.hierarchical_fixedtime.core import TrialPolicy,collect_trial,episode_records
from continuous_pointnav_adaptation.hierarchical_fixedtime.environment import make_env


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Optional JSON evaluation output.')
    args=parser.parse_args();torch.set_num_threads(1)
    metadata=json.loads((HERE/'experiment.json').read_text())
    for filename,key in [('best_model.pt','selected_high_sha256'),('frozen_low_actor.pt','frozen_low_sha256')]:
        assert hashlib.sha256((HERE/filename).read_bytes()).hexdigest()==metadata[key],filename
    saved=torch.load(HERE/'best_model.pt',map_location='cpu',weights_only=False)
    config=saved['config'];model=TrialPolicy(config['observation_spec']['obs_dim'])
    model.load_state_dict(saved['model_state_dict']);model.eval()
    kwargs={key:config[key] for key in ('step_penalty','collision_penalty','danger_penalty','goal_reward','timeout_penalty')}
    kwargs.update(deadlines=tuple(config['deadlines']),time_scale=config['observation_spec']['time_scale'],
        first_episode_danger_bonus=config['first_episode_danger_bonus'],
        first_episode_bonus_deadlines=config['first_episode_bonus_deadlines'],
        low_checkpoint=HERE/'frozen_low_actor.pt')
    records=[]
    for gate in ('open','closed'):
        env=make_env(fixed_gate=gate,**kwargs)
        try:
            # Training's fixed-gate scene first caches qpos during demonstration
            # collection. Reproduce that initialization before evaluation resets.
            first_deadline=next(d for g,d in config['training_cases'] if g==gate)
            env.reset(seed=100000+config['seed'],options=dict(gate_state=gate,deadline=first_deadline,new_trial=True))
            for deadline in config['deadlines']:
                trial=collect_trial(model,env,gate,deadline,20000,deterministic=True)
                records.extend(episode_records(trial))
        finally:env.close()
    expected=json.loads((HERE/'best_evaluation.json').read_text())['episodes']
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(checkpoint_steps=saved['total_steps'],episodes=records),indent=2))
    assert len(records)==len(expected)==18
    for actual,reference in zip(records,expected):
        for key in reference:
            if key=='return_':np.testing.assert_allclose(actual[key],reference[key],atol=1e-6,rtol=0)
            else:assert actual[key]==reference[key],(key,actual,reference)
    # Verify no local project module was resolved from the changing workspace.
    for name,module in tuple(sys.modules.items()):
        if name.startswith('continuous_pointnav_adaptation') and getattr(module,'__file__',None):
            assert Path(module.__file__).resolve().is_relative_to(RUNTIME),module.__file__
    last=[r for r in records if r['episode']==3]
    result=dict(experiment='final_experiment_2',checkpoint_steps=saved['total_steps'],
        matched_archived_evaluation=True,third_episode_successes=sum(r['success'] for r in last),
        evaluated_cases=len(last),episodes=records)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))


if __name__=='__main__':main()

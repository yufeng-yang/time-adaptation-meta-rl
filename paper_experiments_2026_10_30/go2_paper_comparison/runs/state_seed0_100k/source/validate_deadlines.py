"""All gate patterns and deadlines, using locally acquired gate history only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from layout import Config,PATTERNS
from simulation import make_batch
from environment import GateTrial
from demo import run_trials


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Refusing overwrite')
    torch.set_num_threads(1)
    config=Config()
    patterns=list(PATTERNS)*len(config.deadlines)
    deadlines=np.repeat(config.deadlines,len(PATTERNS))
    backends=make_batch([10]*len(patterns),config)
    envs=[GateTrial(b,config,10) for b in backends]
    start=time.monotonic()
    trials=run_trials(envs,patterns,deadlines)
    no_contacts=all(not r['wall_contact_decisions'] and not r['motion_failed'] for t in trials for r in t['records'])
    expected={18.:{'11':[1,1,1],'10':[1,1,1],'01':[1,1,1],'00':[1,1,1]},
              48.:{'11':[4,4,4],'10':[4,4,4],'01':[1,4,4],'00':[1,1,1]},
              100.:{'11':[4,4,4],'10':[4,4,4],'01':[4,4,4],'00':[4,4,4]}}
    strategic_pattern=all([r['delivered_value'] for r in t['records']]==expected[t['deadline']][t['doors']] for t in trials)
    result=dict(metadata=envs[0].metadata(),trials=trials,all_clean=no_contacts,expected_strategy_pattern=strategic_pattern,
        controller='Hand-specified static-layout waypoint controller with actual ray evidence; no learned policy',
        wall_seconds=time.monotonic()-start,
        caveat='Single demonstration seed for deadline checks; measured scripted feasibility, not shortest-path bounds',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict(doors=t['doors'],deadline=t['deadline'],
                          deliveries=[r['delivered'] for r in t['records']],
                          values=[r['delivered_value'] for r in t['records']],
                          seconds=[r['seconds'] for r in t['records']]) for t in trials],indent=2),flush=True)
    if not no_contacts: raise SystemExit('Contact/fall found in trial check')
    if not strategic_pattern: raise SystemExit('Measured strategy pattern differs from design; inspect records before training')


if __name__=='__main__': main()

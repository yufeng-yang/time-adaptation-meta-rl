"""Sequential training queue with persistent logs, failure state and automatic evaluation."""
import os,sys,subprocess,json,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
os.chdir(R.parent)
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_CACHE_DIR='/tmp/go2_comparison_numba',MPLCONFIGDIR='/tmp/time_adapt_mpl',GS_CACHE_FILE_PATH='/tmp/time_adapt_genesis_cache',XDG_CACHE_HOME='/tmp/time_adapt_cache',PYTHONDONTWRITEBYTECODE='1')
def status(**kw):(R/'queue_status.json').write_text(json.dumps(dict(timestamp=time.time(),**kw),indent=2))
completed=[]
for mode in ('state','time','context'):
    status(phase='training',current=mode,completed=completed)
    cmd=[sys.executable,'-B',str(R/'runtime_source/gates/train.py'),'--output',str(R/'runs'/f'{mode}_seed0_100k'),'--mode',mode,'--device','cuda','--eval-every','5']
    with (R/f'{mode}_seed0.log').open('x') as log:
        result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        status(phase='failed',current=mode,completed=completed,exit_code=result.returncode);sys.exit(result.returncode)
    completed.append(mode)
status(phase='evaluation',completed=completed)
with (R/'evaluation.log').open('x') as log:
    result=subprocess.run([sys.executable,'-B',str(R/'evaluate.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
status(phase='complete' if result.returncode==0 else 'evaluation_failed',completed=completed,exit_code=result.returncode)
sys.exit(result.returncode)

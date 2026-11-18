"""Sequential fresh-process campaign. Every primary and diagnostic job is retained."""
import hashlib,json,os,random,subprocess,sys,time
from pathlib import Path
from case_config import cases,SEEDS

HERE=Path(__file__).resolve().parent
def cid(c):return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:16]
def jobs():
    result=[]
    for rep in range(3):
        suites=['dynamic','burst','dimension','capacity','grid','threshold','yago_threshold'];random.Random(81550+rep).shuffle(suites)
        for suite in suites:result.append(dict(suite=suite,repeat=rep))
    for i,seed in enumerate([11,29,47]):result.append(dict(suite='large',repeat=i,seed=seed))
    # Balanced order of matched uninstrumented/instrumented control processes.
    for rep in range(2):
        for diagnostic in ([False,True] if rep==0 else [True,False]):
            result.append(dict(suite='phase_control',repeat=rep,diagnostic=diagnostic,output=f'{"diagnostics" if diagnostic else "observer_controls"}/phase_control_p{rep}'))
    result.append(dict(suite='dynamic',repeat=0,diagnostic=True))
    for suite in ['burst','dimension','capacity','grid','threshold','yago_threshold','large']:
        result.append(dict(suite=suite,repeat=0,seed=11,diagnostic=True))
    corner=next(c for c in cases('grid') if c['degree']==32 and c['length']==6 and c['budget']==1048576)
    for seed in SEEDS[1:]:result.append(dict(suite='grid',repeat=0,seed=seed,diagnostic=True,only_case=cid(corner),output=f'diagnostics/grid_high_s{seed}'))
    return result

def main():
    statepath=HERE/'campaign_status.json';logdir=HERE/'campaign_logs';logdir.mkdir(exist_ok=True)
    if not (HERE/'case_config_frozen.json').exists():raise RuntimeError('Freeze configuration first')
    frozen=json.loads((HERE/'case_config_frozen.json').read_text())
    assert frozen['campaign_jobs']==jobs()
    state=json.loads(statepath.read_text()) if statepath.exists() else dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),jobs=[])
    done={r['index'] for r in state['jobs'] if r.get('exit_code')==0};env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def save():
        temp=statepath.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(statepath)
    state.update(supervisor_pid=os.getpid(),complete=False);save()
    for index,job in enumerate(jobs()):
        if index in done:continue
        command=[sys.executable,str(HERE/'benchmark_main.py')]
        for key,value in job.items():
            flag='--'+key.replace('_','-')
            if isinstance(value,bool):
                if value:command.append(flag)
            else:command.extend([flag,str(HERE/value) if key=='output' else str(value)])
        log=logdir/f'{index:02d}_{job["suite"]}.log'
        record=dict(index=index,job=job,command=command,log=log.name,started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),start_epoch=time.time())
        with log.open('a',buffering=1) as f:
            p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env);record['pid']=p.pid;state['jobs'].append(record);state['current_job']=index;save();print('START',index,job,'pid',p.pid,flush=True);record['exit_code']=p.wait()
        record['elapsed_s']=time.time()-record['start_epoch'];save();print('FINISH',index,'exit',record['exit_code'],'seconds',round(record['elapsed_s'],1),flush=True)
        if record['exit_code']:raise RuntimeError('Campaign stopped on failed job; completed result files remain unchanged')
    state.update(complete=True,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()

if __name__=='__main__':main()

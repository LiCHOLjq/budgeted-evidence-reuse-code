"""Resume only missing diagnostic jobs, preserving all original status/results."""
import json,os,subprocess,sys,time
from pathlib import Path
import run_campaign
HERE=Path(__file__).resolve().parent

def main():
    original=json.loads((HERE/'campaign_status.json').read_text())
    assert {r['index'] for r in original['jobs'] if r.get('exit_code')==0}==set(range(31))
    assert original['jobs'][-1]['index']==31 and original['jobs'][-1]['exit_code']==1
    path=HERE/'campaign_recovery_status.json'
    state=json.loads(path.read_text()) if path.exists() else dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),original_status_sha256=run_campaign.hashlib.sha256((HERE/'campaign_status.json').read_bytes()).hexdigest(),jobs=[])
    state.update(supervisor_pid=os.getpid(),complete=False)
    def save():
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(path)
    done={r['index'] for r in state['jobs'] if r.get('exit_code')==0}
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for index,planned in enumerate(run_campaign.jobs()):
        if index<31 or index in done:continue
        job=dict(planned)
        if index==31:job['output']='diagnostics/capacity_p0_recovery'
        command=[sys.executable,str(HERE/'benchmark_recovery_main.py')]
        for key,value in job.items():
            flag='--'+key.replace('_','-')
            if isinstance(value,bool):
                if value:command.append(flag)
            else:command.extend([flag,str(HERE/value) if key=='output' else str(value)])
        log=HERE/'campaign_logs'/f'recovery_{index:02d}_{job["suite"]}.log'
        row=dict(index=index,job=job,command=command,start_epoch=time.time(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),log=log.name)
        with log.open('a') as f:
            p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env);row['pid']=p.pid;state['jobs'].append(row);state['current_job']=index;save();row['exit_code']=p.wait()
        row['elapsed_s']=time.time()-row['start_epoch'];save()
        if row['exit_code']:raise RuntimeError('Recovery failed; preserve all evidence')
    state.update(complete=True,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save()

if __name__=='__main__':main()

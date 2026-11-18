"""Additive recovery: only post-query nullable snapshot aggregation changes."""
import inspect,json,shutil
from pathlib import Path
import benchmark_main as b

HERE=Path(__file__).resolve().parent
OLD="if key not in ['phase','qid','footprint_kind'] and isinstance(ss[0][key],(int,float)):su['sample_mean_'+key]=float(np.mean([z[key] for z in ss]))"
NEW="""if key not in ['phase','qid','footprint_kind']:
                    valid=[z[key] for z in ss if isinstance(z[key],(int,float))]
                    if valid:su['sample_mean_'+key]=float(np.mean(valid))
                    if key=='python_to_logical_ratio':
                        su['sample_nonnull_count_'+key]=len(valid)
                        su['sample_null_count_'+key]=len(ss)-len(valid)"""

def install():
    frozen=json.loads((HERE/'RECOVERY_FROZEN_MAIN.json').read_text())
    for name,h in frozen['source_sha256'].items():assert b.sha(HERE/name)==h,(name,'recovery source changed')
    original=inspect.getsource(b.run_case);assert original.count(OLD)==1
    # Only the post-query summary statement is replaced. The untouched measured
    # functions, workload generator, executor and source map remain original.
    exec(compile(original.replace(OLD,NEW),str(__file__),'exec'),b.__dict__)
    original_write=b.json_write
    def write(path,value):
        if isinstance(value,dict) and ('source_sha256' in value):
            value=dict(value,recovery_provenance=dict(kind='nullable_snapshot_summary_only',manifest_sha256=b.sha(HERE/'RECOVERY_FROZEN_MAIN.json'),source_sha256=frozen['source_sha256']))
        original_write(path,value)
        if path.name=='environment_start.json':
            for name in list(frozen['source_sha256'])+['RECOVERY_FROZEN_MAIN.json']:
                dest=path.parent/'recovery_source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(HERE/name,dest)
    b.json_write=write

if __name__=='__main__':install();b.main()

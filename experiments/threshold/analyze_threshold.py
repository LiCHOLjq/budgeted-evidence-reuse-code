"""threshold paired threshold, construction-work and diagnostic-state audit."""
import csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'recomputed'/'threshold'
OUT.mkdir(parents=True, exist_ok=True)
TAUS=[16,64,256,1000000000]
def write(name,rows):
    with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    summary=[];states={};phase=defaultdict(list);errors=[];count=0;environments=[];construction=[]
    expected=json.loads((HERE/'FROZEN_THRESHOLD.json').read_text())
    old={x['seed']:x['workload_sha256'] for x in json.loads((HERE/'stream_hashes.json').read_text())}
    observed_pairs=set();observed_methods=set();raw_request_count=0
    for folder in sorted((HERE/'results').glob('*')):
        if not folder.is_dir():continue
        env=json.loads((folder/'environment.json').read_text());environments.append(env);count+=env['queries'];summary.extend(json.loads((folder/'summary.json').read_text()))
        pair=(env['mode'],env['repeat'])
        if pair in observed_pairs:errors.append('Duplicate process block')
        observed_pairs.add(pair)
        wm=json.loads((folder/'workload_manifest.json').read_text())
        if {r['seed'] for r in wm}!=set(old) or len(wm)!=len(old) or any(not r['matched_ablation'] or r['workload_sha256']!=old[r['seed']] or r['queries']!=480 for r in wm):errors.append('Retained stream manifest mismatch')
        if env['source_sha256']!=expected['source_sha256']:errors.append('Source set differs from frozen threshold experiment')
        for file in sorted(folder.glob('s*_tau*.csv')):
            with file.open(newline='') as f:rows=list(csv.DictReader(f))
            raw_request_count+=len(rows)
            if len(rows)!=480 or any(int(r['error']) for r in rows):errors.append('Bad request ledger '+str(file))
            if any(float(r['used'])>65536 or not math.isclose(float(r['service_us']),float(r['query_us'])+float(r['maintenance_us']),abs_tol=1e-8) for r in rows):errors.append('Bad capacity/service ledger '+str(file))
            seed,tau=int(rows[0]['seed']),int(rows[0]['tau'])
            key=(env['mode'],env['repeat'],seed,tau)
            if key in observed_methods or [int(r['qid']) for r in rows]!=list(range(480)):errors.append('Duplicate method or invalid request order')
            observed_methods.add(key)
            zz=[x for x in summary if x['mode']==env['mode'] and x['repeat']==env['repeat'] and x['seed']==seed and x['tau']==tau]
            if len(zz)!=1:errors.append('Missing or duplicate method summary')
            else:
                for field in ['service_us','query_us','maintenance_us']:
                    if not math.isclose(float(np.mean([float(r[field]) for r in rows])),zz[0]['mean_'+field],rel_tol=1e-10,abs_tol=1e-8):errors.append('Summary arithmetic mismatch')
                if zz[0]['workload_sha256']!=old[seed] or zz[0]['ablation_workload_sha256']!=old[seed]:errors.append('Summary stream mismatch')
            if env['mode']!='primary':
                for r in rows:states[env['mode'],env['repeat'],seed,tau,int(r['qid'])]=r['persistent_state_sha256']
                for field in ['lookup_validation_ns','graph_ns','certificate_admission_ns','ranking_ns','residual_ns']:
                    phase[env['mode'],env['repeat'],tau,field].append(float(np.mean([float(r[field]) for r in rows])))
            if env['mode']=='counted':
                for field in ['existing_dependency_key_tests','candidate_bucket_tests','fine_assignments','fine_deletions','coarse_assignments','already_coarse_skips']:
                    construction.append(dict(repeat=env['repeat'],seed=seed,tau=tau,metric=field,total=sum(int(r[field]) for r in rows),per_query=float(np.mean([int(r[field]) for r in rows]))))
    if count!=expected['total_executions']:errors.append('Incomplete threshold count')
    if observed_pairs!={('primary',r) for r in range(5)}|{(m,r) for m in ['profile','counted'] for r in [0,1]}:errors.append('Incomplete process coverage')
    if observed_methods!={(m,r,s,t) for m,r in observed_pairs for s in old for t in TAUS} or raw_request_count!=count:errors.append('Incomplete raw method/request coverage')
    primary=[r for r in summary if r['mode']=='primary'];lookup={(r['repeat'],r['seed'],r['tau']):r for r in primary};seeds=sorted({r['seed'] for r in primary});ratios=[]
    rng=np.random.default_rng(948180)
    for rep in range(5):
        baseline=np.array([lookup[rep,s,16]['mean_service_us'] for s in seeds])
        for tau in TAUS:
            target=np.array([lookup[rep,s,tau]['mean_service_us'] for s in seeds]);draw=rng.integers(0,len(seeds),size=(10000,len(seeds)));boot=target[draw].mean(axis=1)/baseline[draw].mean(axis=1)
            ratios.append(dict(repeat=rep,tau=tau,seeds=len(seeds),service_mean_us=float(target.mean()),service_relative_to_tau16=float(target.mean()/baseline.mean()),seed_ci_low=float(np.quantile(boot,.025)),seed_ci_high=float(np.quantile(boot,.975))))
    process=[]
    for tau in TAUS:
        rr=[r for r in ratios if r['tau']==tau];x=np.array([r['service_relative_to_tau16'] for r in rr]);sd=float(x.std(ddof=1));half=2.7764451051977987*sd/math.sqrt(5)
        process.append(dict(tau=tau,processes=5,mean_relative_latency=float(x.mean()),sd=sd,minimum=float(x.min()),maximum=float(x.max()),descriptive_t_low=float(x.mean()-half),descriptive_t_high=float(x.mean()+half),mean_service_us=float(np.mean([r['service_mean_us'] for r in rr]))))
    # Report every pair, so the same-state 64/256 contrast is not selectively
    # added while other contrasts disappear. These are descriptive supplements;
    # tau16 remains the frozen primary reference.
    pair_ratios=[];pair_summary=[]
    for i,baseline_tau in enumerate(TAUS):
        for target_tau in TAUS[i+1:]:
            values=[]
            for rep in range(5):
                baseline=np.mean([lookup[rep,s,baseline_tau]['mean_service_us'] for s in seeds]);target=np.mean([lookup[rep,s,target_tau]['mean_service_us'] for s in seeds]);ratio=float(target/baseline);values.append(ratio)
                pair_ratios.append(dict(repeat=rep,baseline_tau=baseline_tau,target_tau=target_tau,relative_latency=ratio))
            x=np.asarray(values);sd=float(x.std(ddof=1));half=2.7764451051977987*sd/math.sqrt(5)
            pair_summary.append(dict(baseline_tau=baseline_tau,target_tau=target_tau,processes=5,mean_relative_latency=float(x.mean()),sd=sd,minimum=float(x.min()),maximum=float(x.max()),descriptive_t_low=float(x.mean()-half),descriptive_t_high=float(x.mean()+half)))
    paired_state_checks=threshold_checks=0;threshold_differences=defaultdict(int);pair_comparisons=[]
    for rep in [0,1]:
        for seed in seeds:
            for tau in TAUS:
                for q in range(480):
                    paired_state_checks+=1
                    if states['profile',rep,seed,tau,q]!=states['counted',rep,seed,tau,q]:errors.append('Counter profile changed retained state')
            for mode in ['profile','counted']:
                for tau in [64,256]:
                    for q in range(480):
                        threshold_checks+=1
                        if states[mode,rep,seed,tau,q]!=states[mode,rep,seed,16,q]:threshold_differences[str(tau)]+=1
    for left,right in [(16,64),(16,256),(64,256)]:
        comparisons=differences=0
        for rep in [0,1]:
            for mode in ['profile','counted']:
                for seed in seeds:
                    for q in range(480):
                        comparisons+=1;differences+=states[mode,rep,seed,left,q]!=states[mode,rep,seed,right,q]
        pair_comparisons.append(dict(left_tau=left,right_tau=right,comparisons=comparisons,differences=differences))
    observer=[]
    for rep in [0,1]:
        for tau in TAUS:
            plain=[r for r in summary if r['mode']=='profile' and r['repeat']==rep and r['tau']==tau];counted=[r for r in summary if r['mode']=='counted' and r['repeat']==rep and r['tau']==tau]
            observer.append(dict(repeat=rep,tau=tau,counted_over_original_profile_service=float(np.mean([r['mean_service_us'] for r in counted])/np.mean([r['mean_service_us'] for r in plain]))))
    phase_rows=[dict(mode=k[0],repeat=k[1],tau=k[2],phase=k[3],mean_ns=float(np.mean(v))) for k,v in phase.items()]
    construction_checks=defaultdict(set)
    for row in construction:construction_checks[row['seed'],row['tau'],row['metric']].add(row['total'])
    if any(len(v)!=1 for v in construction_checks.values()):errors.append('Construction counts changed across deterministic diagnostic repeats')
    write('primary_ratios.csv',ratios);write('process_summary.csv',process);write('construction.csv',construction);write('observer.csv',observer);write('phases.csv',phase_rows);write('state_comparisons.csv',pair_comparisons);write('threshold_pair_process_ratios.csv',pair_ratios);write('threshold_pair_summary.csv',pair_summary)
    facts=dict(status='PASS' if not errors else 'FAIL',total_executions=count,primary_executions=sum(e['queries'] for e in environments if e['mode']=='primary'),diagnostic_executions=sum(e['queries'] for e in environments if e['mode']!='primary'),workload_seeds=seeds,requests_per_stream=480,all_workloads_match_old_ablation=True,counted_vs_original_state_comparisons=paired_state_checks,small_threshold_state_comparisons=threshold_checks,small_threshold_state_differences_against_tau16=dict(threshold_differences),process_summary=process,observer=observer,errors=errors,ratio_direction='tau service / tau16 service; >1 means slower than tau16',inference_scope='five sequential fresh processes on one host/session; t intervals descriptive; new results do not causally decompose old6.4%spread')
    facts['state_pairs']=pair_comparisons
    facts['construction_repeat_pairs_verified']=len(construction_checks)
    facts['descriptive_all_threshold_pairs']=pair_summary
    (OUT/'key_facts.json').write_text(json.dumps(facts,indent=2));print(json.dumps(facts,indent=2));raise SystemExit(bool(errors))
if __name__=='__main__':main()

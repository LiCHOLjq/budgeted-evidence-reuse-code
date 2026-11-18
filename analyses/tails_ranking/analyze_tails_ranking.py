"""Read-only main reanalysis of tail latency and ranking.

Run from any directory with Python + NumPy. Writes to the selected output directory.
No measured source, retained result, manuscript, or server job is changed.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description='Recompute latency and ranking tables from retained measurements')
parser.add_argument('--output',type=Path,default=ROOT/'recomputed/tails_ranking')
OUT=parser.parse_args().output.resolve()
OUT.mkdir(parents=True,exist_ok=True)
MAIN=ROOT/'experiments'
INPUTS={}
STRICT=json.loads((MAIN/'analysis_final/strict_integrity.json').read_text(encoding='utf8'))
AUDIT=json.loads((MAIN/'analysis_final/analysis_audit.json').read_text(encoding='utf8'))
assert STRICT['status']=='PASS' and AUDIT['status']=='pass_observed_records'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def bind(path):
    path=Path(path);relative=path.relative_to(ROOT).as_posix();digest=sha(path)
    if path.parent==MAIN/'analysis_final' and path.name in AUDIT['output_sha256']:
        assert digest==AUDIT['output_sha256'][path.name],relative
    elif relative.startswith('experiments/results/') or relative.startswith('experiments/diagnostics/'):
        assert digest==STRICT['files'][path.relative_to(MAIN).as_posix()],relative
    INPUTS[relative]=digest
    return path

def read_csv(path):
    with bind(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def write_csv(name,rows):
    assert rows,name
    fields=list(rows[0])
    with (OUT/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def write_json(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')

def f(row,key):return float(row[key])
def true(row):return row['diagnostic']=='True'
def mean(v):return float(np.mean(v)) if len(v) else None
def quantile(v,q):return float(np.percentile(v,q,method='linear')) if len(v) else None

def one(rows,**spec):
    result=[r for r in rows if all(str(r[k])==str(v) for k,v in spec.items())]
    assert len(result)==1,(spec,len(result))
    return result[0]

def latex_table(name,caption,label,head,rows,align,legend=''):
    lines=[r'\begingroup',r'\footnotesize',r'\setlength{\tabcolsep}{3pt}',
           r'\begin{longtable}{'+align+'}',r'\caption{'+caption+r'}\label{'+label+r'}\\',
           legend+r'\\' if legend else '',r'\toprule',' & '.join(head)+r'\\',r'\midrule',r'\endfirsthead',
           r'\multicolumn{'+str(len(head))+r'}{l}{\textit{Continued from preceding page}}\\',
           r'\toprule',' & '.join(head)+r'\\',r'\midrule',r'\endhead',r'\midrule',
           r'\multicolumn{'+str(len(head))+r'}{r}{\textit{Continued on next page}}\\',r'\endfoot',r'\bottomrule',r'\endlastfoot']
    lines += [' & '.join(str(x) for x in row)+r'\\' for row in rows]
    lines += [r'\end{longtable}',r'\endgroup']
    (OUT/name).write_text('\n'.join(lines)+'\n',encoding='utf8')

def main():
    for p in [MAIN/'analysis_final/analysis_audit.json',MAIN/'analysis_final/strict_integrity.json']:
        bind(p)
    summary=read_csv(MAIN/'analysis_final/configuration_summary.csv')
    runs=read_csv(MAIN/'analysis_final/run_summary.csv')
    ratios=read_csv(MAIN/'analysis_final/process_ratios.csv')
    ids={key:f'C{i+1:03d}' for i,key in enumerate(sorted({(r['suite'],r['case_id']) for r in summary}))}
    assert len(ids)==225
    primary=[r for r in summary if not true(r) and r['phase']=='steady']
    assert len(primary)==647
    run_groups=defaultdict(list)
    for r in runs:
        if not true(r) and r['phase']=='steady':run_groups[(r['suite'],r['case_id'],r['method'])].append(r)
    tails=[]
    short={'EagerMemo':'Memo','LocalLRU':'LRU','LocalLRUMemo':'LRU+m','LocalHybrid':'Hybrid','EagerIndexLRU':'Index','AdaptiveGate':'Gate'}
    for r in sorted(primary,key=lambda r:(ids[(r['suite'],r['case_id'])],r['method'])):
        rr=run_groups[(r['suite'],r['case_id'],r['method'])]
        for metric in ['query_p95_us','service_p95_us']:
            assert np.isclose(mean([f(x,metric) for x in rr]),f(r,metric),rtol=1e-12,atol=1e-9)
        tails.append(dict(config_id=ids[(r['suite'],r['case_id'])],suite=r['suite'],case_id=r['case_id'],method=r['method'],
                          runs=len(rr),seed_count=r['workload_seeds'],process_count=r['process_repeat_labels'],
                          requests_per_run=int(r['count'])-int(r['warmup']),
                          mean_query_p95_us=f(r,'query_p95_us'),mean_service_p95_us=f(r,'service_p95_us'),
                          mean_service_us=f(r,'mean_service_us'),definition='Arithmetic mean of per-run steady-window percentile; not pooled p95'))
    write_csv('service_query_tails_all_647.csv',tails)
    latex_table('service_query_tails_all_647.tex',
      'Complete primary query and complete-service tails. Each entry is the arithmetic mean of per-run steady-window 95th percentiles, in microseconds; these are not pooled percentiles. IDs match the configuration registry. Small-graph rows have 15 matched seed/process runs, LiveJournal rows three seed/process-confounded runs. Service includes maintenance and query.',
      'analysis-tab:tails-all',['ID','Policy',r'Q95 $\mu$s',r'S95 $\mu$s'],
      [[r['config_id'],short[r['method']],f"{r['mean_query_p95_us']:,.2f}",f"{r['mean_service_p95_us']:,.2f}"] for r in tails],
      'llrr')

    # Stage accounting is diagnostic, with seed 11 only in these families.
    phases=[]
    for r in summary:
        if r['suite'] not in ('large','dimension') or not true(r) or r['phase']!='steady':continue
        lookup=f(r,'mean_phase_lookup_validation_ns')/1000
        graph=f(r,'mean_phase_graph_ns')/1000;cert=f(r,'mean_phase_certificate_admission_ns')/1000
        rank=f(r,'mean_phase_ranking_ns')/1000;residual=f(r,'mean_phase_residual_ns')/1000
        outer=f(r,'mean_query_us')-f(r,'mean_phase_total_ns')/1000
        maintenance=f(r,'mean_maintenance_us');service=f(r,'mean_service_us')
        assert np.isclose(lookup+graph+cert+rank+residual+outer+maintenance,service,rtol=1e-10)
        phases.append(dict(config_id=ids[(r['suite'],r['case_id'])],suite=r['suite'],case_id=r['case_id'],method=r['method'],
                           length=int(r['length']),dimension=int(r['dimension']),hot_count=int(r['hot_count']),
                           diagnostic_runs=int(r['runs']),diagnostic_seed_count=int(r['workload_seeds']),diagnostic_seed_ids='11',
                           mean_eligible_count=f(r,'mean_eligible_count'),lookup_validation_us=lookup,graph_us=graph,
                           certificate_admission_us=cert,ranking_us=rank,residual_us=residual,outer_query_us=outer,
                           maintenance_us=maintenance,service_us=service,ranking_fraction_of_service=rank/service,
                           ranking_fraction_of_query=rank/f(r,'mean_query_us')))
    write_csv('diagnostic_ranking_phases.csv',phases)
    large=[r for r in phases if r['suite']=='large'];assert len(large)==10
    write_csv('livejournal_phases_C081_C082.csv',large)
    latex_table('livejournal_phases.tex',
      'LiveJournal diagnostic complete-service decomposition. One separate seed-11 replay per configuration, 10,000 steady requests. Times are mean microseconds. Lookup includes validation; Cert. combines certificate construction, admission, and eviction. Other combines residual and outer query work. Ranking fraction divides ranking time by complete service. This instrumented replay does not causally decompose primary timing or establish population variability.',
      'analysis-tab:live-phases',['ID','Policy','Lookup','Graph','Cert.','Rank','Other','Maint.','Service',r'Rank\%'],
      [[r['config_id'],short[r['method']],*[f'{r[k]:.2f}' for k in ['lookup_validation_us','graph_us','certificate_admission_us','ranking_us']],
        f"{r['residual_us']+r['outer_query_us']:.2f}",f"{r['maintenance_us']:.2f}",f"{r['service_us']:.2f}",f"{100*r['ranking_fraction_of_service']:.1f}"]
       for r in sorted(large,key=lambda r:(r['config_id'],r['method']))],'llrrrrrrrr')
    dimensions=[]
    for h in [3,4]:
        for d in [32,128,384,768]:
            a=one(summary,suite='dimension',diagnostic='False',phase='steady',length=h,dimension=d,method='EagerMemo')
            b=one(summary,suite='dimension',diagnostic='False',phase='steady',length=h,dimension=d,method='LocalLRU')
            ratio=one(ratios,suite='dimension',diagnostic='False',phase='steady',length=h,dimension=d,method='LocalLRU',comparator='EagerMemo')
            da=one(phases,suite='dimension',length=h,dimension=d,method='EagerMemo')
            db=one(phases,suite='dimension',length=h,dimension=d,method='LocalLRU')
            dimensions.append(dict(config_id=ids[('dimension',a['case_id'])],case_id=a['case_id'],length=h,dimension=d,
              primary_memo_mean_us=f(a,'mean_service_us'),primary_lru_mean_us=f(b,'mean_service_us'),
              primary_absolute_saving_us=f(a,'mean_service_us')-f(b,'mean_service_us'),
              primary_process_mean_ratio=f(ratio,'mean_ratio'),ratio_of_grand_means=f(a,'mean_service_us')/f(b,'mean_service_us'),
              diagnostic_seed=11,diagnostic_memo_rank_service_fraction=da['ranking_fraction_of_service'],
              diagnostic_lru_rank_service_fraction=db['ranking_fraction_of_service'],
              diagnostic_memo_rank_us=da['ranking_us'],diagnostic_lru_rank_us=db['ranking_us']))
    write_csv('dimension_savings_and_ranking.csv',dimensions)
    latex_table('dimension_savings.tex',
      'Within-dimension-family primary service and separately measured ranking fractions. Primary means average 15 runs; M/P averages three matched-seed process ratios. Ranking fractions are rank/service in a separate seed-11 diagnostic replay, not a decomposition of the primary means.',
      'analysis-tab:dimension-cost',['$h$','$d$','Memo','LRU','Saved',r'M/P',r'Memo rank\%',r'LRU rank\%'],
      [[r['length'],r['dimension'],*[f'{r[k]:.2f}' for k in ['primary_memo_mean_us','primary_lru_mean_us','primary_absolute_saving_us']],
        f"{r['primary_process_mean_ratio']:.3f}",f"{100*r['diagnostic_memo_rank_service_fraction']:.1f}",f"{100*r['diagnostic_lru_rank_service_fraction']:.1f}"] for r in dimensions], 'rrrrrrrr')

    # Primary tail membership is selected separately in each recorded run.
    # Diagnostic costs are attached by seed/qid only after exact state/work checks.
    case=one(summary,suite='dynamic',diagnostic='False',phase='steady',length=4,graph_rate=0.1,update_locality='uniform',method='LocalLRU')['case_id']
    paired=[];per_run=[];diag_unique=[]
    aligned_fields=['charged_bytes','resident_entries','evicted','invalidated','edge_visits','eligible_count','effective_edits']
    for seed in [11,29,47,61,73]:
        diag=read_csv(MAIN/f'diagnostics/dynamic_p0/{case}_s{seed}_LocalLRU_traces.csv')
        dm={int(x['qid']):x for x in diag}
        diag_unique += [dict(x,seed=seed) for x in diag if x['phase']=='steady']
        for repeat in range(3):
            lr=read_csv(MAIN/f'results/dynamic_p{repeat}/{case}_s{seed}_LocalLRU_traces.csv')
            mm=read_csv(MAIN/f'results/dynamic_p{repeat}/{case}_s{seed}_EagerMemo_traces.csv')
            lr=[x for x in lr if x['phase']=='steady'];mm={int(x['qid']):x for x in mm}
            cutoff=quantile([f(x,'service_us') for x in lr],95)
            memo_p95=quantile([f(mm[int(x['qid'])],'service_us') for x in lr],95)
            per_run.append(dict(seed=seed,process_repeat=repeat,lru_service_p95_us=cutoff,memo_service_p95_us=memo_p95,
                                p95_difference_us=cutoff-memo_p95,steady_requests=len(lr)))
            for x in lr:
                qid=int(x['qid']);d=dm[qid];m=mm[qid]
                for key in aligned_fields:assert f(x,key)==f(d,key),(seed,repeat,qid,key)
                assert f(x,'eligible_count')==f(m,'eligible_count')
                rec=dict(config_id=ids[('dynamic',case)],suite='dynamic',case_id=case,seed=seed,process_repeat=repeat,qid=qid,primary_lru_tail=f(x,'service_us')>=cutoff,
                         lru_service_us=f(x,'service_us'),memo_service_us=f(m,'service_us'),
                         paired_service_excess_us=f(x,'service_us')-f(m,'service_us'),
                         effective_edit=f(x,'effective_edits')>0,eligible_count=f(x,'eligible_count'),edge_visits=f(x,'edge_visits'),
                         evicted=f(x,'evicted'),invalidated=f(x,'invalidated'),resident_entries=f(x,'resident_entries'))
                for key in ['diag_validation_tokens','diag_validation_ns','diag_query_misses','diag_prefix_hits','diag_final_hits',
                            'diag_validation_failures','phase_graph_ns','phase_certificate_admission_ns','phase_ranking_ns','phase_lookup_validation_ns']:
                    rec['matched_'+key]=f(d,key)
                paired.append(rec)
    assert len(paired)==5760 and len(diag_unique)==1920
    write_csv('h4_primary_tail_matched_positions.csv',paired)
    write_csv('h4_primary_per_run_p95.csv',per_run)
    tail_groups=[]
    predicates={
      'all':lambda r:True,'tail':lambda r:r['primary_lru_tail'],'non_tail':lambda r:not r['primary_lru_tail'],
      'post_effective_edit':lambda r:r['effective_edit'],'no_effective_edit':lambda r:not r['effective_edit'],
      'tail_post_effective_edit':lambda r:r['primary_lru_tail'] and r['effective_edit'],
      'tail_no_effective_edit':lambda r:r['primary_lru_tail'] and not r['effective_edit']}
    metrics=['lru_service_us','memo_service_us','paired_service_excess_us','eligible_count','edge_visits','evicted','invalidated','resident_entries',
             'matched_diag_validation_tokens','matched_diag_validation_ns','matched_phase_graph_ns','matched_phase_certificate_admission_ns',
             'matched_phase_ranking_ns','matched_phase_lookup_validation_ns']
    for label,predicate in predicates.items():
        subset=[r for r in paired if predicate(r)]
        base=dict(group=label,observations=len(subset),unique_seed_qid=len({(r['seed'],r['qid']) for r in subset}),
                  effective_edit_fraction=mean([r['effective_edit'] for r in subset]),
                  full_miss_fraction=mean([r['matched_diag_query_misses'] for r in subset]),
                  prefix_hit_fraction=mean([r['matched_diag_prefix_hits'] for r in subset]),
                  final_hit_fraction=mean([r['matched_diag_final_hits'] for r in subset]),
                  eviction_request_fraction=mean([r['evicted']>0 for r in subset]),
                  validation_failure_fraction=mean([r['matched_diag_validation_failures']>0 for r in subset]))
        for metric in metrics:
            values=[r[metric] for r in subset]
            base[metric+'_mean']=mean(values);base[metric+'_p95']=quantile(values,95)
        tail_groups.append(base)
    write_csv('h4_tail_conditionals.csv',tail_groups)
    chosen=[r for r in tail_groups if r['group'] in ['all','tail','post_effective_edit','tail_post_effective_edit']]
    names={'all':'All','tail':'Primary tail','post_effective_edit':'After edit','tail_post_effective_edit':'Tail after edit'}
    latex_table('tail_mechanism_table.tex',
      'Wiki-Vote h=4, uniform updates at probability 0.1: primary-tail membership and matched diagnostic work. Tail membership is selected separately within each primary run; N counts repeated memberships, not independent observations. Validation, graph and certificate/admission times are from a separate diagnostic replay at the same seed/request and are not a primary-time decomposition. Diagnostic values can repeat across primary processes. Full miss is request-level; evictions are entries per request.',
      'analysis-tab:tail-mechanism',['Subset','$N$',r'Miss\%','Evict./q',r'Valid. $\mu$s',r'Graph $\mu$s',r'Cert. $\mu$s'],
      [[names[r['group']],r['observations'],f"{100*r['full_miss_fraction']:.1f}",f"{r['evicted_mean']:.2f}",
        f"{r['matched_diag_validation_ns_mean']/1000:.1f}",f"{r['matched_phase_graph_ns_mean']/1000:.1f}",
        f"{r['matched_phase_certificate_admission_ns_mean']/1000:.1f}"] for r in chosen],'lrrrrrr')
    # Secondary deduplicated view: union of seed/qid positions selected as a tail
    # in any primary process. Each diagnostic observation contributes only once.
    grouped=defaultdict(list)
    for row in paired:grouped[(row['seed'],row['qid'])].append(row)
    unique=[]
    for key,rr in sorted(grouped.items()):
        row=dict(rr[0]);row.pop('process_repeat');row['tail_memberships_across_three_processes']=sum(x['primary_lru_tail'] for x in rr)
        row['primary_lru_tail']=any(x['primary_lru_tail'] for x in rr)
        for metric in ['lru_service_us','memo_service_us','paired_service_excess_us']:
            row[metric]=mean([x[metric] for x in rr])
        unique.append(row)
    write_csv('h4_unique_seed_positions.csv',unique)
    dedup=[]
    for label in ['all_unique','tail_union_unique','tail_union_postedit_unique']:
        subset=[r for r in unique if label=='all_unique' or (r['primary_lru_tail'] and (label!='tail_union_postedit_unique' or r['effective_edit']))]
        row=dict(group=label,unique_seed_qid=len(subset),effective_edit_fraction=mean([r['effective_edit'] for r in subset]),
                 full_miss_fraction=mean([r['matched_diag_query_misses'] for r in subset]),prefix_hit_fraction=mean([r['matched_diag_prefix_hits'] for r in subset]),
                 final_hit_fraction=mean([r['matched_diag_final_hits'] for r in subset]),eviction_request_fraction=mean([r['evicted']>0 for r in subset]))
        for metric in metrics:
            row[metric+'_mean']=mean([r[metric] for r in subset]);row[metric+'_p95']=quantile([r[metric] for r in subset],95)
        dedup.append(row)
    write_csv('h4_unique_tail_union_conditionals.csv',dedup)
    conditions=[]
    for seed in [11,29,47,61,73]:
        for kind in ['all','post_effective_edit','no_effective_edit']:
            values=[r for r in diag_unique if r['seed']==seed and (kind=='all' or (f(r,'effective_edits')>0)==(kind=='post_effective_edit'))]
            conditions.append(dict(seed=seed,condition=kind,requests=len(values),
              validation_mean_us=mean([f(r,'diag_validation_ns')/1000 for r in values]),
              validation_p95_us=quantile([f(r,'diag_validation_ns')/1000 for r in values],95),
              token_mean=mean([f(r,'diag_validation_tokens') for r in values]),token_p95=quantile([f(r,'diag_validation_tokens') for r in values],95)))
    write_csv('h4_diagnostic_validation_conditionals.csv',conditions)
    fact=dict(status='PASS',source='retained audited main only; no new timings',all_primary_rows=647,
              trace_alignment=dict(primary_positions=5760,unique_diagnostic_seed_positions=1920,exact_fields=aligned_fields,failures=0),
              tail_definition='For each of 15 primary runs, service_us >= its linear-interpolated steady-window p95; 20/384 positions per run. Matched diagnostic measurements are selected by the same seed/qid, not substituted into primary timing.',
              primary_h4_p95=dict(lru_mean_us=mean([r['lru_service_p95_us'] for r in per_run]),memo_mean_us=mean([r['memo_service_p95_us'] for r in per_run]),
                                 runs_with_lru_p95_greater=sum(r['p95_difference_us']>0 for r in per_run),runs=len(per_run)),
              groups={r['group']:r for r in tail_groups},deduplicated_groups={r['group']:r for r in dedup},dimension=dimensions,livejournal=large,
              caveats=['No request-position independence assumed.','Tail selection by observed LocalLRU primary latency is descriptive and selection-conditioned.',
                       'Diagnostic seed/qid values repeat across three primary processes in membership-weighted tables; separate unique-position union tables count each diagnostic position once. No confidence interval treats the 300 memberships as independent.',
                       'Diagnostic phases are separately instrumented and cannot causally allocate primary tail excess.',
                       'Certificate/admission phase includes eviction bookkeeping; it does not separately time copies and evictions.',
                       'Ranking scans eligible C, not every graph vertex. An ANN implementation changes work and potentially answer semantics; no restored ratio is predicted.',
                       'LiveJournal phase estimates and dimension ranking fractions use one diagnostic seed only.'])
    write_json('key_findings.json',fact)
    write_json('input_hashes.json',INPUTS)
    write_snippets(fact)
    outputs={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ['OUTPUT_MANIFEST.json']}
    write_json('OUTPUT_MANIFEST.json',dict(status='PASS',script_sha256=sha(Path(__file__)),inputs=INPUTS,outputs=outputs,
                                         modifications_outside_output_directory=False))
    print(json.dumps({'status':'PASS','all_tail_rows':647,'matched_positions':5760,'tail_positions':fact['groups']['tail']['observations'],
                      'primary_h4_p95':fact['primary_h4_p95'],'files':len(outputs)}))

def write_snippets(fact):
    t=fact['groups']['tail'];allr=fact['groups']['all'];te=fact['groups']['tail_post_effective_edit'];post=fact['groups']['post_effective_edit']
    text=(r'\paragraph{Complete-service tails and miss reconstruction.} '+
      'Table~\\ref{analysis-tab:tails-all} reports both query and complete-service p95 for every configuration/policy. Each value averages per-run steady-window percentiles rather than pooling requests. '+
      f"At the uniform-update Wiki-Vote $h=4$ reference, LocalLRU's mean run p95 is {fact['primary_h4_p95']['lru_mean_us']:,.1f} microseconds versus {fact['primary_h4_p95']['memo_mean_us']:,.1f} for EagerMemo, despite its lower mean service. "+
      f"Selecting requests at or above each LocalLRU run's own p95 yields {t['observations']} of {allr['observations']:,} repeated request positions ({t['unique_seed_qid']} distinct seed/request pairs). Diagnostic seed/request measurements can therefore be selected in more than one of the three primary processes; these membership-weighted summaries are not independent samples and receive no confidence interval. A separate deduplicated union table counts each diagnostic position once. "+
      f"Matched diagnostic route counters show {100*t['full_miss_fraction']:.1f}\\% complete cache misses, {100*t['prefix_hit_fraction']:.1f}\\% prefix hits and no full hits. These requests traverse {t['edge_visits_mean']:,.0f} edges and evict {t['evicted_mean']:.2f} entries on average, versus {allr['edge_visits_mean']:,.0f} and {allr['evicted_mean']:.2f} over all steady requests. "+
      f"Only {te['observations']}/{t['observations']} tail positions follow an effective graph edit. Their matched diagnostic validation averages {t['matched_diag_validation_ns_mean']/1000:.1f} microseconds, below the all-request {allr['matched_diag_validation_ns_mean']/1000:.1f}; graph expansion and certificate/admission average {t['matched_phase_graph_ns_mean']/1000:,.1f} and {t['matched_phase_certificate_admission_ns_mean']/1000:,.1f} microseconds. "+
      f"Across post-edit positions, validation's pooled diagnostic p95 is {post['matched_diag_validation_ns_p95']/1000:.1f} microseconds, while the tail/post-edit intersection has p95 {te['matched_diag_validation_ns_p95']/1000:.1f}. "+
      'The evidence associates the upper tail with expensive miss reconstruction and materialization/eviction work, rather than unusually long validation loops. It does not establish that eviction causes a miss or separately identify copying cost. Diagnostic phases come from separate instrumented replays; matching state/work counters supports the association but cannot causally allocate the primary latency excess.\n')
    (OUT/'tail_mechanism.tex').write_text(text,encoding='utf8')
    d0=next(x for x in fact['dimension'] if x['length']==3 and x['dimension']==32)
    d1=next(x for x in fact['dimension'] if x['length']==3 and x['dimension']==768)
    text=(r'\paragraph{Sensitivity to the common ranking stage.} '+
      f"Within the dimension family, $h=3$ complete-service means change from {d0['primary_memo_mean_us']:.1f}/{d0['primary_lru_mean_us']:.1f} microseconds (EagerMemo/LocalLRU) at $d=32$ to {d1['primary_memo_mean_us']:.1f}/{d1['primary_lru_mean_us']:.1f} at $d=768$. Absolute savings are {d0['primary_absolute_saving_us']:.1f} and {d1['primary_absolute_saving_us']:.1f} microseconds; the matched process-mean ratio falls from {d0['primary_process_mean_ratio']:.3f} to {d1['primary_process_mean_ratio']:.3f}. "+
      f"In the separate seed-11 diagnostic replay at $d=768$, ranking occupies {100*d1['diagnostic_memo_rank_service_fraction']:.1f}\\% of EagerMemo service and {100*d1['diagnostic_lru_rank_service_fraction']:.1f}\\% of LocalLRU service. There is no method-independent measured 87\\% fraction. "+
      'This is sensitivity to exhaustive ranking-stage cost on generated vectors, not identification of a production embedding operating point. The executor ranks the eligible set $C$, not all graph vertices. A cheaper common ranking stage would increase the ratio if all other costs and answers were fixed. ANN candidate generation or reranking could instead change index maintenance, recall, candidate sets and execution semantics; no ANN baseline is measured, so neither a return to the low-dimensional ratio nor exact-output equivalence is claimed.\n')
    (OUT/'ranking_scope.tex').write_text(text,encoding='utf8')
    lines=[r'\paragraph{Selective LiveJournal ranking.} '+
      'The full topology contains 4,847,571 vertices, but exhaustive ranking visits only the eligible result of the two-hop predicate. Primary mean eligible sets are approximately 230 and 873 vertices for hot pools 8 and 800. Table~\\ref{analysis-tab:live-phases} adds the separate seed-11 decomposition, whose own eligible means are 188.8 and 772.6; its seed scope differs from the three primary runs.']
    for hot in [8,800]:
        a=next(x for x in fact['livejournal'] if x['hot_count']==hot and x['method']=='EagerMemo')
        b=next(x for x in fact['livejournal'] if x['hot_count']==hot and x['method']=='LocalLRU')
        lines.append(f"For hot pool {hot}, ranking takes {a['ranking_us']:.1f}/{b['ranking_us']:.1f} microseconds and {100*a['ranking_fraction_of_service']:.1f}\\%/{100*b['ranking_fraction_of_service']:.1f}\\% of EagerMemo/LocalLRU diagnostic service; graph expansion takes {a['graph_us']:.1f}/{b['graph_us']:.1f} microseconds.")
    lines.append('Ranking is therefore a substantial common stage even for these selective queries, especially after successful reuse reduces traversal. This one-seed instrumented decomposition is not a population bottleneck estimate and does not causally decompose the three primary seed/process observations.\n')
    (OUT/'livejournal_ranking.tex').write_text(' '.join(lines),encoding='utf8')
    (OUT/'replication_scope.tex').write_text(
      r'\paragraph{Replication and trace-length tradeoff.} '+
      'The selected replication experiment used ten seeds in five sequential fresh processes, with 240 requests for the synthetic corner and 120 requests for each Wiki-Vote length. The main experiment broadened the configuration grid and extended small-graph traces to 512 requests (128 warmup, 384 steady), while reducing replication to five seeds in three processes. Descriptive process-mean Student-$t$ intervals therefore use two degrees of freedom in the main experiment, versus four in that selected replication experiment; the main experiment bands show observed ranges, not those intervals. This trade improves trace-duration and configuration coverage while weakening estimation of process variability. The earlier sorting and other historical suites have their own designs and are not all ten-by-five experiments. The separate threshold exact-stream follow-up retained ten original ablation streams and five primary processes. Large main cases instead use three seeds confounded with three processes. Cross-family repeats share a host/session and may share workload inputs; their spread describes sensitivity to separately scheduled timing families, not independent deployments or a complete uncertainty budget.\n',encoding='utf8')

if __name__=='__main__':main()

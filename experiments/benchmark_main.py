"""Prospective main server campaign: matched streams, fresh processes, full service.

Run --suite pilot before freezing the protocol. Reported output is immutable;
resumption skips only complete case/method files whose source hashes match.
"""
import argparse,csv,gc,hashlib,inspect,json,os,pickle,platform,random,shutil,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'artifact'))
import engine_comparisons,engine_main
from case_config import SEEDS,METHODS,cases,methods
from large_graph import CSRGraph

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for z in iter(lambda:f.read(8<<20),b''):h.update(z)
    return h.hexdigest()

def json_write(p,v):p.write_text(json.dumps(v,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else float(x)),encoding='utf8')
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def endpoint_hash(s):return hashlib.sha256(np.asarray(sorted(s),dtype='<u8').tobytes()).hexdigest()

def source_hashes():
    sources=[HERE/'benchmark_main.py',HERE/'case_config.py',HERE/'large_graph.py',HERE/'engine_main.py',HERE.parent/'artifact'/'engine_comparisons.py',HERE.parent/'artifact'/'engine_replication.py']
    sources.extend(p for p in [HERE/'engine_protocol.md',HERE/'EXPERIMENT_PROTOCOL_MAIN.md',HERE/'case_config_frozen.json',HERE/'run_campaign.py'] if p.exists())
    return {str(p.relative_to(HERE.parent)):sha(p) for p in sources}

def vectors(c,seed):
    # Query/vector coordinate streams use the first d coordinates of one 768D stream.
    if c['dataset']=='livejournal':
        out=HERE/'data'/f'livejournal_vectors_s{seed}_d{c["dimension"]}.npy'
        if not out.exists():
            z=np.lib.format.open_memmap(out,mode='w+',dtype=np.float64,shape=(c['n'],c['dimension']))
            rng=np.random.default_rng(seed)
            for a in range(0,c['n'],10000):z[a:a+10000]=rng.normal(size=(min(10000,c['n']-a),c['dimension']))
            z.flush();del z
        return np.load(out,mmap_mode='r')
    return np.random.default_rng(seed).normal(size=(c['n'],768))[:,:c['dimension']].copy()

def graph_input(c,seed):
    if c['dataset']=='livejournal':return HERE/'data'/'livejournal',vectors(c,seed)
    a=[[set() for _ in range(c['n'])] for _ in range(2)]
    if c['dataset'] in ['wiki_vote','yago']:
        p=HERE/'data'/('wiki_vote.json' if c['dataset']=='wiki_vote' else 'yago_subclass_2400.json')
        data=json.loads(p.read_text())
        assert data['n']==c['n']
        for u,v in data['edges']:a[0][u].add(v);a[1][v].add(u)
    else:
        rng=random.Random(seed)
        for r in [0,1]:
            for u in range(c['n']):
                start=u//200*200;a[r][u].update(rng.sample(range(start,start+200),c['degree']))
    return a,vectors(c,seed)

def graph(c,a,v):return CSRGraph(a,v) if c['dataset']=='livejournal' else engine_main.Graph(a,v)

def fresh(g,a,path,q):
    # Independent set-comprehension traversal; shared float64 ranking kernel.
    f={a};fronts=[1];edges=[];read=set();seenfront=[frozenset(f)]
    for rel in path:
        read.update(rel*g.n+int(u) for u in f)
        edges.append(sum(len(g.adj[rel][u]) for u in f))
        f={int(v) for u in f for v in g.adj[rel][u]}
        fronts.append(len(f));seenfront.append(frozenset(f))
    stable=next((j for j in range(len(path)) if all(z==seenfront[-1] for z in seenfront[j:])),None)
    f.discard(a);ranked=engine_main.rank(g.vectors,f,q,10)
    return dict(topk=ranked,count=len(f),endpoint_sha256=endpoint_hash(f),frontier_sizes=fronts,edge_visits_by_hop=edges,distinct_read_buckets=len(read),stable_start=stable)

def workload(c,seed,a,v):
    # The shadow-domain reference is always LocalLRU at 1MiB, regardless of
    # tested policy, budget or hybrid threshold. Warmup/query shape is fixed.
    generator_hash=hashlib.sha256(''.join(inspect.getsource(f) for f in [workload,fresh,graph_input,vectors]).encode()).hexdigest()
    input_file=HERE/'data'/('wiki_vote.json' if c['dataset']=='wiki_vote' else 'yago_subclass_2400.json' if c['dataset']=='yago' else 'livejournal/manifest.json')
    input_hash=sha(input_file) if c['dataset']!='synthetic' else 'synthetic_generator_in_source'
    wc={k:v for k,v in c.items() if k not in ['budget','hybrid_tau']};identity=dict(case=wc,seed=seed,generator_hash=generator_hash,input_hash=input_hash,numpy=np.__version__);wid=digest(identity)
    dest=HERE/'workloads'/f'{wid}.pkl';dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        with dest.open('rb') as f:pack=pickle.load(f)
        assert pack['identity']==identity
        return pack
    gr=random.Random(30000+seed);qr=random.Random(10000+seed);vr=random.Random(20000+seed);nr=np.random.default_rng(40000+seed)
    hot=qr.sample(range(c['n']),c['hot_count']);g=graph(c,a,v);ref=engine_main.EvidenceCache(g,'LocalLRU',1048576)
    output=[];hh=hashlib.sha256();d=c['dimension'];paths=[tuple(([s,1-s]*8)[:c['length']-1]+[tail]) for s in [0,1] for tail in [0,1]]
    for i in range(c['count']):
        anchor=qr.choice(hot) if qr.random()<c['query_locality'] else qr.randrange(c['n']);path=qr.choice(paths)
        q=nr.normal(size=768)[:d].copy();vu=nr.normal(size=768)[:d].copy();vv=vr.randrange(c['n']);vector=(vv,vu) if vr.random()<c['vector_rate'] else None
        trigger=gr.random();number=c['burst'] if c['burst'] and i and i%c['burst_period']==0 else (0 if c['burst'] else int(trigger<c['graph_rate']))
        edits=[];hot_requested=hot_selected=0
        for j in range(max(1,number)):
            rel,src,dst,insert=gr.randrange(2),gr.randrange(c['n']),gr.randrange(c['n']),bool(gr.randrange(2));hotdraw=gr.random()
            if j>=number:continue
            if c['update_locality']=='hot':
                hot_requested+=1;domain=sorted({b for e in ref.entries.values() for b in e.deps if b>=0})
                if domain:
                    bucket=domain[min(int(hotdraw*len(domain)),len(domain)-1)];rel,src=divmod(bucket,c['n']);hot_selected+=1
            bucket=g.adj[rel][src]
            if not insert and len(bucket):dst=int(sorted(bucket)[int(hotdraw*len(bucket))]);insert=False
            else:
                insert=True
                while dst in bucket:dst=(dst+1)%c['n']
            edit=(rel,src,dst,insert);assert engine_main.apply_edge(ref,*edit);edits.append(edit)
        if vector is not None:g.vector(*vector)
        ref.synchronize();ans=ref.query(anchor,path,q);truth=fresh(g,anchor,path,q)
        assert tuple(ans[0])==tuple(truth['topk']) and len(ans[1])==truth['count'] and endpoint_hash(ans[1])==truth['endpoint_sha256']
        row=dict(qid=i,phase='warmup' if i<c['warmup'] else 'steady',anchor=anchor,path=path,q=q,edits=edits,vector=vector,hot_requested=hot_requested,hot_selected=hot_selected,truth=truth)
        hh.update(str((i,anchor,path,edits,vv if vector else None)).encode());hh.update(q.tobytes())
        if vector:hh.update(vu.tobytes())
        output.append(row)
    pack=dict(workload_id=wid,workload_sha256=hh.hexdigest(),generator_sha256=generator_hash,identity=identity,case=wc,seed=seed,rows=output,graph_edits=sum(len(r['edits']) for r in output),graph_update_requests=sum(bool(r['edits']) for r in output),vector_updates=sum(r['vector'] is not None for r in output))
    temp=dest.with_suffix('.tmp')
    with temp.open('wb') as f:pickle.dump(pack,f,protocol=5)
    temp.replace(dest);return pack

def apply(cache,r):
    effective=0
    for e in r['edits']:effective+=int(engine_main.apply_edge(cache,*e))
    if r['vector'] is not None:cache.g.vector(*r['vector'])
    cache.synchronize();return effective

def run_case(c,seed,suite,rep,ci,out,diagnostic=False):
    cid=digest(c)[:16];a,v=graph_input(c,seed);pack=workload(c,seed,a,v);requests=pack['rows']
    order=[m for m in methods(suite) if not diagnostic or m!='AdaptiveGate'];random.Random(701003+rep*1009+seed*101+ci).shuffle(order)
    for mi,method in enumerate(order):
        label=f'{cid}_s{seed}_{method}';target=out/(label+'.json')
        if target.exists():
            old=json.loads(target.read_text());assert old['source_sha256']==source_hashes();continue
        g=graph(c,a,v);cache=engine_main.EvidenceCache(g,method,c['budget'],hybrid_tau=c['hybrid_tau'],profile=diagnostic and method!='AdaptiveGate',diagnostics=diagnostic and method!='AdaptiveGate',gate_seed=seed)
        rows=[];snapshots=[];overlaps=[];gate=[];start=time.time()
        for r in requests:
            before=cache.stats.copy()
            if diagnostic and r['edits']:overlaps.append(dict(qid=r['qid'],phase=r['phase'],**engine_main.overlap_audit(cache,[rel*g.n+src for rel,src,dst,ins in r['edits']])))
            t0=time.perf_counter_ns();effective=apply(cache,r);t1=time.perf_counter_ns();answer=cache.query(r['anchor'],r['path'],r['q']);t2=time.perf_counter_ns()
            truth=r['truth'];error=int(tuple(answer[0])!=tuple(truth['topk']) or len(answer[1])!=truth['count'] or endpoint_hash(answer[1])!=truth['endpoint_sha256'])
            z=dict(qid=r['qid'],phase=r['phase'],query_us=(t2-t1)/1000,maintenance_us=(t1-t0)/1000,service_us=(t2-t0)/1000,error=error,eligible_count=truth['count'],charged_bytes=cache.used,resident_entries=len(cache.entries),graph_edits=len(r['edits']),effective_edits=effective,vector_update=int(r['vector'] is not None),dependency_count=cache.last_dependency_count,memo_peak_bytes=cache.last_memo_peak_bytes)
            underlying=cache.cache if method=='AdaptiveGate' else cache
            z['entry_charge_bytes']=cache.used-getattr(underlying,'index_bytes',0)-(256 if method=='AdaptiveGate' else 0)
            z['mean_entry_charge']=z['entry_charge_bytes']/len(cache.entries) if cache.entries else 0
            z['mean_resident_charge']=cache.used/len(cache.entries) if cache.entries else 0
            z.update({k:cache.stats[k]-before.get(k,0) for k in cache.stats})
            if diagnostic:
                z.update({f'diag_{k}':v for k,v in getattr(cache,'last_diagnostics',{}).items()})
                z.update({f'phase_{k}':v for k,v in cache.last_profile_ns.items()})
            rows.append(z)
            # Primary bookkeeping is O(1). Every full resident inventory,
            # certificate scan and object footprint belongs to separate replay.
            if diagnostic and ((r['qid']+1)%32==0 or r['qid'] in [c['warmup']-1,len(requests)-1]):
                snap=engine_main.cache_snapshot(cache,footprint=diagnostic)
                assert snap['used_bytes']==snap['entry_charge_bytes']+snap['index_bytes']+snap['gate_bytes']
                snapshots.append(dict(qid=r['qid'],phase=r['phase'],**snap))
            if method=='AdaptiveGate':gate.append(dict(qid=r['qid'],phase=r['phase'],**cache.last_gate))
            if error or cache.used>c['budget'] or effective!=len(r['edits']):raise AssertionError((label,r['qid'],error,cache.used,effective,len(r['edits'])))
        def csvout(name,data):
            if not data:return
            keys=list(dict.fromkeys(k for z in data for k in z))
            with (out/(label+'_'+name+'.csv')).open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
        for name,data in [('traces',rows),('snapshots',snapshots),('overlaps',overlaps),('gate',gate)]:csvout(name,data)
        summaries=[]
        for phase in ['all','warmup','steady']:
            rr=rows if phase=='all' else [z for z in rows if z['phase']==phase];ss=snapshots if phase=='all' else [z for z in snapshots if z['phase']==phase]
            su=dict(phase=phase,queries=len(rr),errors=sum(z['error'] for z in rr),query_p95_us=float(np.percentile([z['query_us'] for z in rr],95)))
            for key in rows[0]:
                if key not in ['phase','qid']:su['mean_'+key]=float(np.mean([z.get(key,0) for z in rr]))
            for key in (ss[0] if ss else []):
                if key not in ['phase','qid','footprint_kind'] and isinstance(ss[0][key],(int,float)):su['sample_mean_'+key]=float(np.mean([z[key] for z in ss]))
            summaries.append(su)
        json_write(target,dict(case=c,case_id=cid,suite=suite,process_repeat=rep,seed=seed,method=method,diagnostic=diagnostic,case_order=ci,method_order=mi,elapsed_s=time.time()-start,workload_sha256=pack['workload_sha256'],workload_id=pack['workload_id'],source_sha256=source_hashes(),summary=summaries))
        print(suite,rep,ci,method,seed,'mean_us',round(summaries[0]['mean_service_us'],2),'elapsed',round(time.time()-start,1),flush=True)
        del cache,g,rows;gc.collect()
    # Witnesses and hop sizes are policy-independent, stored once per case.
    witness=out/f'{cid}_s{seed}_witness.json'
    if not witness.exists():json_write(witness,dict(case=c,seed=seed,workload_sha256=pack['workload_sha256'],graph_edits=pack['graph_edits'],vector_updates=pack['vector_updates'],rows=[dict(qid=r['qid'],phase=r['phase'],anchor=r['anchor'],path=r['path'],**r['truth']) for r in requests]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True);p.add_argument('--repeat',type=int,default=0);p.add_argument('--seed',type=int);p.add_argument('--diagnostic',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--output');p.add_argument('--only-case');a=p.parse_args()
    if not a.suite.startswith('pilot') and not (HERE/'EXPERIMENT_PROTOCOL_MAIN.md').exists():raise RuntimeError('Freeze the protocol before reported runs')
    out=Path(a.output) if a.output else HERE/('diagnostics' if a.diagnostic else 'results')/f'{a.suite}_p{a.repeat}'
    out.mkdir(parents=True,exist_ok=True)
    for relative,expected in source_hashes().items():
        src=HERE.parent/relative;dst=out/'source_snapshot'/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert sha(dst)==expected,('Frozen source mismatch',relative)
        else:shutil.copy2(src,dst)
    cc=[(c,s) for c in cases(a.suite) for s in ([a.seed] if a.seed is not None else ([11] if a.suite.startswith('pilot') else SEEDS))]
    if a.only_case:cc=[(c,s) for c,s in cc if digest(c)[:16]==a.only_case]
    random.Random(912701+a.repeat*65537+sum(map(ord,a.suite))).shuffle(cc)
    if a.limit is not None:cc=cc[:a.limit]
    data_files=[HERE/'data'/'wiki_vote.json',HERE/'data'/'yago_subclass_2400.json',HERE/'data'/'livejournal'/'manifest.json']
    meta=dict(suite=a.suite,repeat=a.repeat,diagnostic=a.diagnostic,pid=os.getpid(),python=sys.version,numpy=np.__version__,platform=platform.platform(),cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else [],load_before=os.getloadavg() if hasattr(os,'getloadavg') else [],source_sha256=source_hashes(),data_sha256={str(p.relative_to(HERE)):sha(p) for p in data_files if p.exists()},started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),cases=len(cc))
    json_write(out/'environment_start.json',meta);start=time.time()
    for i,(c,s) in enumerate(cc):run_case(c,s,a.suite,a.repeat,i,out,a.diagnostic)
    meta.update(elapsed_s=time.time()-start,load_after=os.getloadavg() if hasattr(os,'getloadavg') else [],complete=True);json_write(out/'environment.json',meta)

if __name__=='__main__':main()

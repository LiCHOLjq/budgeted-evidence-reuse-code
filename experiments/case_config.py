"""main prospective case definitions. Frozen by protocol before reported runs."""
SEEDS=[11,29,47,61,73]
REPEATS=3
METHODS=['EagerMemo','LocalLRU','LocalLRUMemo','LocalHybrid','EagerIndexLRU','AdaptiveGate']
TAUS=[16,64,256,1000000000]

def base(**kw):
    c=dict(dataset='wiki_vote',n=7115,degree=0,length=3,budget=1048576,dimension=32,graph_rate=.1,vector_rate=.2,update_locality='uniform',query_locality=.9,hot_count=8,burst=0,burst_period=0,count=512,warmup=128,hybrid_tau=64)
    c.update(kw);return c

def cases(suite):
    if suite=='dynamic':return [base(length=h,graph_rate=p,update_locality=l) for h in [3,4] for p in [0,.05,.1,.3,.5,.8] for l in ['uniform','hot']]
    if suite=='burst':return [base(length=h,graph_rate=0,update_locality=l,burst=16,burst_period=p) for h in [3,4] for l in ['uniform','hot'] for p in [32,160]]
    if suite=='dimension':return [base(length=h,dimension=d) for h in [3,4] for d in [32,128,384,768]]
    if suite=='capacity':return [base(length=h,budget=b) for h in [3,4] for b in [65536,262144,1048576,4194304]]
    if suite=='grid':return [base(dataset='synthetic',n=2400,degree=d,length=h,budget=b) for d in [4,8,16,32] for h in [3,4,5,6] for b in [65536,1048576]]
    if suite=='threshold':return [dict(c,hybrid_tau=t) for c in cases('grid')+[base(length=h) for h in [3,4]] for t in TAUS]
    if suite=='yago_threshold':return [base(dataset='yago',n=2400,length=3,budget=65536,hybrid_tau=t) for t in TAUS]
    if suite=='large':return [base(dataset='livejournal',n=4847571,length=2,budget=16777216,count=12000,warmup=2000,hot_count=h) for h in [8,800]]
    if suite=='phase_control':return [base(length=h) for h in [3,4]]+[base(dataset='synthetic',n=2400,degree=32,length=6)]
    if suite=='pilot':return [base(length=h,count=64,warmup=16,dimension=d) for h in [3,4] for d in [32,768]]+[base(dataset='synthetic',n=2400,degree=32,length=6,count=64,warmup=16)]
    if suite=='pilot_large':return [dict(c,count=128,warmup=32) for c in cases('large')]
    raise ValueError(suite)

def methods(suite):return ['LocalHybrid'] if suite in ['threshold','yago_threshold'] else [m for m in METHODS if m!='AdaptiveGate'] if suite=='phase_control' else METHODS

def describe():
    out={}
    for s in ['dynamic','burst','dimension','capacity','grid','threshold','yago_threshold']:
        cs=cases(s);out[s]=dict(cases=cs,methods=methods(s),seeds=SEEDS,processes=REPEATS,requests=sum(c['count'] for c in cs)*len(SEEDS)*REPEATS*len(methods(s)))
    out['large']=dict(cases=cases('large'),methods=METHODS,seeds=[11,29,47],processes='one fresh process per seed',requests=2*12000*len(METHODS)*3)
    return out

if __name__=='__main__':
    import json
    from pathlib import Path
    z=describe();Path(__file__).with_name('case_config_draft.json').write_text(json.dumps(z,indent=2));print({k:v['requests'] for k,v in z.items()});print('total',sum(v['requests'] for v in z.values()))

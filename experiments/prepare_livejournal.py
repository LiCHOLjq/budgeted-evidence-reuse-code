"""Normalize the complete public SNAP LiveJournal graph into read-only CSR.

No edge sampling is performed. Original node IDs are already dense below n.
Two relation arrays retain the original direction and its explicit inverse.
"""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for z in iter(lambda:f.read(8<<20),b''):h.update(z)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path(__file__).parent/'data');a=p.parse_args();out=a.data/'livejournal';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Refusing to overwrite completed normalized dataset')
    raw=a.data/'soc-LiveJournal1.txt.gz';start=time.time();print('Parsing complete original graph',flush=True)
    with gzip.open(raw,'rt') as f:edges=np.loadtxt(f,dtype=np.uint32,comments='#')
    n=int(edges.max())+1;m=len(edges);assert m==68993773 and n==4847571,(n,m)
    unique=np.unique(edges.astype(np.uint64)[:,0]*n+edges[:,1]);assert len(unique)==m;del unique
    files={}
    for relation in [0,1]:
        src,dst=edges[:,relation],edges[:,1-relation]
        order=np.argsort(src,kind='stable');indices=dst[order];ptr=np.r_[np.uint64(0),np.cumsum(np.bincount(src,minlength=n),dtype=np.uint64)]
        # Sort each bucket by destination through a global lexicographic order.
        order=np.lexsort((dst,src));indices=dst[order]
        for kind,arr in [('indptr',ptr),('indices',indices)]:
            target=out/f'r{relation}_{kind}.npy';np.save(target,arr);files[target.name]=dict(sha256=sha(target),bytes=target.stat().st_size,dtype=str(arr.dtype),shape=list(arr.shape))
        print('Completed relation',relation,'elapsed',round(time.time()-start,1),flush=True)
        del src,dst,order,indices,ptr
    meta=dict(source_url='https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz',source_page='https://snap.stanford.edu/data/soc-LiveJournal1.html',raw_sha256=sha(raw),raw_bytes=raw.stat().st_size,n=n,original_directed_edges=m,typed_edges=2*m,relations=['original direction','explicit inverse'],selection='complete graph; no sampling; preserve original dense IDs',self_loops=int(np.count_nonzero(edges[:,0]==edges[:,1])),files=files,elapsed_s=time.time()-start)
    (out/'manifest.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__':main()

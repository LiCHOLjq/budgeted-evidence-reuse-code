"""CSR plus copy-on-first-mutation buckets for the complete-graph scale control.

Every policy, including the oracle, uses the identical adapter. A bucket access
returns its sorted NumPy slice until that bucket is modified, then a Python set.
Overlay lookup is on the measured path; only graph construction is excluded.
"""
import json
from pathlib import Path
import numpy as np

class Relation:
    def __init__(self,indptr,indices):self.indptr,self.indices,self.changed=indptr,indices,{}
    def __getitem__(self,u):
        x=self.changed.get(u)
        return x if x is not None else self.indices[self.indptr[u]:self.indptr[u+1]]
    def __len__(self):return len(self.indptr)-1

class CSRGraph:
    def __init__(self,data,vectors):
        data=Path(data);self.n=len(vectors);self.adj=[Relation(np.load(data/f'r{r}_indptr.npy',mmap_mode='r'),np.load(data/f'r{r}_indices.npy',mmap_mode='r')) for r in [0,1]]
        self.versions=np.zeros(2*self.n,dtype=np.uint64);self.relation_epochs=[0,0];self.graph_epoch=self.vector_epoch=0
        self.vectors=np.asarray(vectors,dtype=np.float64).copy()
    def edge(self,relation,src,dst,insert):
        rel=self.adj[relation];bucket=rel[src]
        present=(dst in bucket)
        if present==insert:return False
        if src not in rel.changed:bucket=set(map(int,bucket));rel.changed[src]=bucket
        (bucket.add if insert else bucket.remove)(dst)
        self.versions[relation*self.n+src]+=1;self.relation_epochs[relation]+=1;self.graph_epoch+=1;return True
    def vector(self,vertex,value):self.vectors[vertex]=value;self.vector_epoch+=1

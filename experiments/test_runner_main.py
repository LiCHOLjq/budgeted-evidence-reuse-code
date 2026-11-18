"""Pre-measurement semantic checks for CSR storage and matched workload streams."""
import tempfile,unittest
from pathlib import Path
import numpy as np
import benchmark_main as b
from case_config import base
from large_graph import CSRGraph

class RunnerTests(unittest.TestCase):
    def test_csr_exact_storage_and_updates(self):
        a=[[{1,2},{2},set(),{0},set()],[set(),{0},{0,1},set(),{3}]];v=np.arange(20).reshape(5,4).astype(float)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            for r in [0,1]:
                np.save(Path(d)/f'r{r}_indptr.npy',np.array([0]+list(np.cumsum([len(x) for x in a[r]])),dtype=np.uint64))
                np.save(Path(d)/f'r{r}_indices.npy',np.array([u for s in a[r] for u in sorted(s)],dtype=np.uint32))
            regular=b.engine_main.Graph(a,v);csr=CSRGraph(d,v)
            left=b.engine_main.EvidenceCache(regular,'EagerIndexLRU',100000);right=b.engine_main.EvidenceCache(csr,'EagerIndexLRU',100000)
            for i in range(100):
                u,r,dst=i%5,i%2,(i*3)%5;insert=dst not in regular.adj[r][u]
                self.assertEqual(b.engine_main.apply_edge(left,r,u,dst,insert),b.engine_main.apply_edge(right,r,u,dst,insert))
                left.synchronize();right.synchronize()
                for path in [(0,1),(1,0,1),(0,0,1,0)]:self.assertEqual(left.query(u,path,np.ones(4)),right.query(u,path,np.ones(4)))
                self.assertEqual(left.used,right.used)
                self.assertEqual(list(regular.versions),list(csr.versions))
            for rel in csr.adj:
                rel.indptr._mmap.close();rel.indices._mmap.close()

    def test_matched_query_and_vector_streams(self):
        configs=[base(dataset='synthetic',n=2400,degree=4,count=40,warmup=8,graph_rate=p,update_locality=l,dimension=d) for p,l,d in [(0,'uniform',32),(.8,'hot',32),(.1,'uniform',128)]]
        packs=[]
        for c in configs:
            a,v=b.graph_input(c,947);packs.append(b.workload(c,947,a,v))
        for rows in zip(*(p['rows'] for p in packs)):
            self.assertEqual(len({(r['anchor'],r['path']) for r in rows}),1)
            for r in rows[1:]:np.testing.assert_array_equal(rows[0]['q'],r['q'][:32])
        self.assertGreater(sum(r['hot_selected'] for r in packs[1]['rows']),0)

if __name__=='__main__':unittest.main()

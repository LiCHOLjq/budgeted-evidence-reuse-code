"""Ablation differential and invariant tests; run python -m unittest -v test_engine_ablation."""
import random
from time import perf_counter_ns
import unittest
import numpy as np
from engine_ablation import Graph, EvidenceCache, METHODS, oracle


class RevisionTests(unittest.TestCase):
    def graph(self, n=8):
        adjacency = [[set() for _ in range(n)] for _ in range(3)]
        adjacency[0][0] = {1, 2, 3}
        adjacency[1][1] = {4}
        adjacency[1][2] = {5}
        adjacency[2][4] = {6}
        return Graph(adjacency, np.arange(n*3).reshape(n, 3))

    def check(self, g, c, anchor=0, path=(0, 1), k=10):
        actual = c.query(anchor, path, np.zeros(3), k)
        self.assertEqual(actual, oracle(g, anchor, path, np.zeros(3), k))
        self.assertEqual(c.used, sum(e.size for e in c.entries.values()))
        self.assertLessEqual(c.used, c.capacity)
        for key, e in c.entries.items():
            self.assertEqual(e.size, c.charge(key, e.endpoints, e.deps))
            for token in (x for x in e.deps if x < 0):
                relation = -1-token
                self.assertFalse(any(relation*g.n <= x < (relation+1)*g.n for x in e.deps))
        return actual

    def test_threshold_is_per_relation_and_strict(self):
        for tau, expected in [(0, {-1, -2}), (2, {0, -2}), (3, {0, 9, 10, 11})]:
            g = self.graph()
            c = EvidenceCache(g, 'LocalHybrid', 4096, hybrid_tau=tau)
            self.check(g, c)
            self.assertEqual(set(c.entries[(0, (0, 1))].deps), expected)

    def test_cumulative_repeated_relation_compression(self):
        g = self.graph()
        c = EvidenceCache(g, 'LocalHybrid', 4096, hybrid_tau=3)
        self.check(g, c, path=(0, 0))
        self.assertEqual(c.entries[(0, (0, 0))].deps, {-1: 0})

    def test_empty_bucket_insertion_and_parent_eviction(self):
        for method in ('LocalLRU', 'LocalHybrid', 'LocalFinal'):
            for tau in (0, 2, 64):
                g = self.graph()
                c = EvidenceCache(g, method, 4096, hybrid_tau=tau)
                self.check(g, c)
                if (0, (0,)) in c.entries:
                    c.drop((0, (0,)), 'evicted')
                g.edge(1, 3, 7, True)  # Previously empty inspected bucket.
                result = self.check(g, c)
                self.assertIn(7, result[1])
                self.assertGreater(c.stats['invalidated'], 0)

    def test_transitive_coarse_certificate_survives_parent_eviction(self):
        g = self.graph()
        c = EvidenceCache(g, 'LocalHybrid', 4096, hybrid_tau=0)
        self.check(g, c, path=(0, 1, 2))
        c.drop((0, (0,)), 'evicted')
        c.drop((0, (0, 1)), 'evicted')
        g.edge(0, 0, 1, False)
        self.assertNotIn(6, self.check(g, c, path=(0, 1, 2))[1])

    def test_same_relation_conservatism_other_relation_reuse(self):
        g = self.graph()
        c = EvidenceCache(g, 'LocalHybrid', 4096, hybrid_tau=2)
        self.check(g, c)
        e = c.entries[(0, (0, 1))]
        g.edge(2, 7, 0, True)
        self.check(g, c)
        self.assertIs(c.entries[(0, (0, 1))], e)
        g.edge(1, 7, 0, True)  # Unread bucket but relation is coarse.
        self.check(g, c)
        self.assertIsNot(c.entries[(0, (0, 1))], e)

    def test_noop_aba_and_vector_domains(self):
        g = self.graph()
        c = EvidenceCache(g, 'LocalHybrid', 4096, hybrid_tau=0)
        self.check(g, c)
        e = c.entries[(0, (0, 1))]
        self.assertFalse(g.edge(0, 0, 1, True))
        self.assertEqual(g.relation_epochs, [0, 0, 0])
        g.vector(5, [0, 0, 0])
        self.assertEqual(self.check(g, c)[0][0], 5)
        self.assertIs(c.entries[(0, (0, 1))], e)
        g.edge(0, 0, 1, False)
        g.edge(0, 0, 1, True)
        self.assertEqual(g.relation_epochs, [2, 0, 0])
        self.check(g, c)
        self.assertIsNot(c.entries[(0, (0, 1))], e)

    def test_stale_inherited_coarse_tag_cannot_be_refreshed(self):
        g = self.graph()
        c = EvidenceCache(g, 'LocalHybrid', 4096, hybrid_tau=0)
        self.check(g, c, path=(0,))
        old = c.entries[(0, (0,))]
        g.edge(0, 0, 4, True)
        # Longer miss must reject stale shorter prefix BEFORE inheriting deps.
        self.assertIn(6, self.check(g, c, path=(0, 2))[1])
        self.assertEqual(old.deps, {-1: 0})
        self.assertEqual(c.entries[(0, (0, 2))].deps[-1], 1)

    def test_huge_tau_equals_local_lru_work_and_content(self):
        graphs = [self.graph(), self.graph()]
        caches = [EvidenceCache(graphs[0], 'LocalLRU', 512),
                  EvidenceCache(graphs[1], 'LocalHybrid', 512, hybrid_tau=10**9)]
        rng = random.Random(834)
        for i in range(150):
            edge = rng.randrange(3), rng.randrange(8), rng.randrange(8), bool(rng.randrange(2))
            for g in graphs:
                g.edge(*edge)
            a, path = rng.randrange(8), tuple(rng.randrange(3) for _ in range(3))
            results = [self.check(g, c, a, path) for g, c in zip(graphs, caches)]
            self.assertEqual(results[0], results[1])
            self.assertEqual(caches[0].stats, caches[1].stats)
            self.assertEqual(caches[0].entries, caches[1].entries)

    def test_profile_matches_unprofiled_and_nonoverlapping(self):
        for method in METHODS:
            graphs = [self.graph(), self.graph()]
            caches = [EvidenceCache(graphs[0], method, 512, hybrid_tau=2),
                      EvidenceCache(graphs[1], method, 512, hybrid_tau=2, profile=True)]
            for i, path in enumerate([(0, 1), (0, 1), (0, 2), (0, 1, 2), (), (0, 0)]):
                if i == 3:
                    for g in graphs:
                        g.edge(1, 3, 7, True)
                actual = self.check(graphs[0], caches[0], path=path)
                outer = perf_counter_ns()
                profiled = caches[1].query(0, path, np.zeros(3))
                outer = perf_counter_ns()-outer
                self.assertEqual(actual, profiled)
                self.assertEqual(caches[0].stats, caches[1].stats)
                self.assertEqual(caches[0].entries, caches[1].entries)
                p = caches[1].last_profile_ns
                names = ['lookup_validation_ns', 'graph_ns', 'certificate_admission_ns', 'ranking_ns']
                self.assertGreaterEqual(p['residual_ns'], 0)
                self.assertEqual(sum(p[x] for x in names)+p['residual_ns'], p['total_ns'])
                self.assertLessEqual(p['total_ns'], outer)

    def test_random_differential_mixed_updates_all_policies(self):
        for method in METHODS:
            for tau in (0, 2, 64):
                for fast in (False, True):
                    rng = random.Random(991)
                    nrng = np.random.default_rng(991)
                    g = self.graph(20)
                    c = EvidenceCache(g, method, 1024, hybrid_tau=tau, epoch_fast_path=fast)
                    for i in range(100):
                        g.edge(rng.randrange(3), rng.randrange(20), rng.randrange(20), bool(rng.randrange(2)))
                        if i % 4 == 0:
                            g.vector(rng.randrange(20), nrng.normal(size=3))
                        a = rng.randrange(20)
                        path = tuple(rng.randrange(3) for _ in range(rng.randrange(1, 5)))
                        self.check(g, c, a, path)

    def test_zero_tiny_capacity_zero_k_and_negative_k(self):
        for method in METHODS:
            for capacity in (0, 1, 128, 512):
                g = self.graph()
                c = EvidenceCache(g, method, capacity)
                for path in [(0,), (0, 1), (), (2,)]:
                    self.assertEqual(self.check(g, c, path=path, k=0)[0], ())
                with self.assertRaises(ValueError):
                    c.query(0, (0,), np.zeros(3), -1)

    def test_cycles_allow_anchor_intermediate_and_exclude_final(self):
        for method in METHODS:
            g = self.graph()
            g.edge(1, 1, 0, True)
            c = EvidenceCache(g, method, 4096)
            result = self.check(g, c, path=(0, 1, 0))
            self.assertEqual(result[1], frozenset({1, 2, 3}))


if __name__ == '__main__':
    unittest.main(verbosity=2)

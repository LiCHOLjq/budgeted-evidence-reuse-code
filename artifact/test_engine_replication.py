"""Run: python -m unittest -v test_engine_comparisons test_engine_replication (from artifact/)."""
import random
from time import perf_counter_ns
import unittest
import numpy as np
import engine_comparisons as comparisons
from engine_replication import EvidenceCache, Graph, METHODS, oracle


class CombinedCacheTests(unittest.TestCase):
    def nested(self):
        return Graph([[{0, 1}, {1, 2}, {2}, set()]], np.zeros((4, 2)))

    def query(self, cache, path, anchor=0, k=10):
        args = anchor, path, np.zeros(2), k
        result = cache.query(*args)
        self.assertEqual(result, oracle(cache.g, *args))
        self.assertEqual(cache.used, sum(e.size for e in cache.entries.values()))
        self.assertLessEqual(cache.used, cache.capacity)
        return result

    def same_persistent_state(self, ordinary, combined):
        self.assertEqual(ordinary.entries, combined.entries)
        self.assertEqual(list(ordinary.entries), list(combined.entries))
        for name in ('used', 'peak', 'clock', 'last_graph', 'last_vector',
                     'last_dependency_count', 'last_endpoint_count'):
            self.assertEqual(getattr(ordinary, name), getattr(combined, name), name)
        for name in ('hits', 'misses', 'invalidated', 'evicted', 'rejected',
                     'validated_buckets', 'flushes'):
            self.assertEqual(ordinary.stats[name], combined.stats[name], name)
        for name in ('edge_visits', 'bucket_expansions'):
            self.assertLessEqual(combined.stats[name], ordinary.stats[name], name)

    def test_old_methods_construct_original_comparisons_executor(self):
        for method in comparisons.METHODS:
            c = EvidenceCache(self.nested(), method, 4096, profile=True, hybrid_tau=2)
            self.assertIs(type(c), comparisons.EvidenceCache)
            self.assertIs(c.evidence.__func__, comparisons.EvidenceCache.evidence)
            self.assertIs(c._profiled_evidence.__func__, comparisons.EvidenceCache._profiled_evidence)
            self.query(c, (0, 0, 0))
        self.assertEqual(METHODS, comparisons.METHODS+['LocalLRUMemo'])
        with self.assertRaises(ValueError):
            EvidenceCache(self.nested(), 'unknown', 1)
        with self.assertRaises(ValueError):
            EvidenceCache(self.nested(), 'LocalLRUMemo', -1)

    def test_exact_and_subset_reuse_preserve_every_prefix(self):
        ordinary = EvidenceCache(self.nested(), 'LocalLRU', 4096)
        combined = EvidenceCache(self.nested(), 'LocalLRUMemo', 4096)
        self.assertEqual(self.query(ordinary, (0,)*4), self.query(combined, (0,)*4))
        self.same_persistent_state(ordinary, combined)
        self.assertEqual(combined.stats['memo_subset_hits'], 2)
        self.assertEqual(combined.stats['memo_subset_saved_buckets'], 3)
        self.assertEqual(combined.stats['memo_hits'], 1)
        self.assertEqual(combined.stats['bucket_expansions'], 3)
        self.assertEqual(combined.stats['edge_visits'], 5)
        self.assertEqual(combined.entries[(0, (0,)*4)].edge_cost, 16)
        self.assertEqual(combined.entries[(0, (0,)*4)].deps, {0: 0, 1: 0, 2: 0})
        self.assertEqual(combined.last_memo_entries, 3)
        self.assertEqual(combined.last_memo_peak_bytes, 376)

    def test_exact_reuse_keeps_empty_bucket_dependency_after_eviction(self):
        g = Graph([[{1, 2}, {1, 2}, set(), set()]], np.zeros((4, 2)))
        c = EvidenceCache(g, 'LocalLRUMemo', 4096)
        path = (0,)*4
        self.query(c, path)
        self.assertEqual(c.stats['memo_hits'], 2)
        self.assertEqual(c.entries[(0, path)].deps, {0: 0, 1: 0, 2: 0})
        for key in list(c.entries):
            if key != (0, path):
                c.drop(key, 'evicted')
        g.edge(0, 2, 3, True)
        self.assertIn(3, self.query(c, path)[1])
        self.assertGreater(c.stats['invalidated'], 0)

    def test_subset_delta_keeps_empty_bucket_and_previous_support(self):
        g = Graph([[{0, 1}, {2}, set(), set()]], np.zeros((4, 2)))
        c = EvidenceCache(g, 'LocalLRUMemo', 4096)
        path = (0,)*3
        self.query(c, path)
        self.assertEqual(c.stats['memo_subset_hits'], 2)
        self.assertEqual(c.stats['bucket_expansions'], 3)
        self.assertEqual(c.entries[(0, path)].deps, {0: 0, 1: 0, 2: 0})
        for key in list(c.entries):
            if key != (0, path):
                c.drop(key, 'evicted')
        g.edge(0, 2, 3, True)
        self.assertIn(3, self.query(c, path)[1])
        g.edge(0, 0, 1, False)
        self.assertEqual(self.query(c, path)[1], frozenset())

    def test_inherited_dependencies_survive_parent_eviction_during_suffix(self):
        g = Graph([[{1, 2}, set(), set(), set()],
                   [set(), {1, 2}, set(), set()]], np.zeros((4, 2)))
        c = EvidenceCache(g, 'LocalLRUMemo', 220)
        self.query(c, (0,))
        path = (0, 1, 1, 1)
        self.query(c, path)
        self.assertEqual(list(c.entries), [(0, path)])
        self.assertEqual(c.stats['memo_hits'], 2)
        self.assertEqual(c.entries[(0, path)].deps, {0: 0, 5: 0, 6: 0})
        g.edge(0, 0, 1, False)
        self.assertEqual(self.query(c, path)[1], frozenset())

    def test_memo_cleared_between_requests_and_final_hits_report_zero(self):
        c = EvidenceCache(self.nested(), 'LocalLRUMemo', 0)
        self.query(c, (0,)*4)
        first, charge = c.stats.copy(), c.last_memo_peak_bytes
        self.query(c, (0,)*4)
        for name in ('memo_hits', 'memo_misses', 'memo_subset_hits', 'edge_visits',
                     'bucket_expansions'):
            self.assertEqual(c.stats[name], 2*first[name])
        self.assertEqual(c.last_memo_peak_bytes, charge)
        self.assertFalse(any(isinstance(x, dict) and ('memo' in name or 'latest' in name)
                             for name, x in vars(c).items()))
        c.capacity = 4096
        self.query(c, (0,)*4)
        self.assertGreater(c.last_memo_peak_bytes, 0)
        self.query(c, (0,)*4)
        self.assertEqual(c.last_memo_peak_bytes, 0)
        self.assertEqual(c.last_memo_entries, 0)
        self.query(c, ())
        self.assertEqual(c.last_memo_peak_bytes, 0)

    def test_memo_is_not_seeded_from_cached_prefix(self):
        c = EvidenceCache(self.nested(), 'LocalLRUMemo', 4096)
        self.query(c, (0, 0))
        before = c.stats.copy()
        self.query(c, (0,)*4)
        self.assertEqual(c.stats['memo_misses']-before['memo_misses'], 1)
        self.assertEqual(c.stats['memo_hits']-before['memo_hits'], 1)
        self.assertEqual(c.stats['bucket_expansions']-before['bucket_expansions'], 3)
        self.assertEqual(c.last_memo_entries, 1)
        self.assertEqual(c.last_memo_peak_bytes, 152)

    def test_unrelated_update_vector_change_noop_and_aba(self):
        g = self.nested()
        c = EvidenceCache(g, 'LocalLRUMemo', 4096)
        path = (0,)*4
        self.query(c, path)
        entry = c.entries[(0, path)]
        g.edge(0, 3, 0, True)  # Not in any dependency.
        self.query(c, path)
        self.assertIs(c.entries[(0, path)], entry)
        g.vector(2, [-2, 0])
        self.assertEqual(self.query(c, path)[0], (1, 2))
        self.assertIs(c.entries[(0, path)], entry)
        self.assertFalse(g.edge(0, 0, 1, True))
        g.edge(0, 0, 1, False)
        g.edge(0, 0, 1, True)
        self.query(c, path)
        self.assertIsNot(c.entries[(0, path)], entry)

    def test_nonnested_cycle_and_relation_sensitive_keys(self):
        g = Graph([[{1}, {0}, set()], [{2}, {1}, set()]], np.zeros((3, 2)))
        c = EvidenceCache(g, 'LocalLRUMemo', 0)
        self.query(c, (0, 0, 0))
        self.assertEqual(c.stats['memo_subset_hits'], 0)
        self.assertEqual(c.stats['memo_hits'], 1)
        self.assertEqual(c.stats['bucket_expansions'], 2)
        g = Graph([[{0}, set()], [{1}, set()]], np.zeros((2, 2)))
        c = EvidenceCache(g, 'LocalLRUMemo', 4096)
        self.assertEqual(self.query(c, (0, 1))[1], frozenset({1}))
        self.assertEqual(c.stats['memo_hits'], 0)

    def test_empty_results_zero_budget_and_k(self):
        g = Graph([[set(), set()]], np.zeros((2, 2)))
        for capacity in (0, 1, 128, 4096):
            c = EvidenceCache(g, 'LocalLRUMemo', capacity)
            self.assertEqual(self.query(c, (0,)*4, k=0), ((), frozenset()))
            self.assertEqual(c.stats['memo_hits'], 2)
            self.assertEqual(c.last_memo_entries, 2)
            self.assertEqual(c.last_memo_peak_bytes, 192)
            with self.assertRaises(ValueError):
                c.query(0, (0,), np.zeros(2), -1)

    def test_mixed_update_differential_persistent_state_and_admission(self):
        # 1,200 query pairs; identical requests/updates, both epoch modes,
        # rejected admissions, frequent evictions and an ample-capacity case.
        for capacity in (0, 128, 320, 1024, 16384):
            for fast in (False, True):
                rng = random.Random(718)
                adjacency = [[set(rng.sample(range(12), 3)) for _ in range(12)]
                             for _ in range(2)]
                graphs = [Graph(adjacency, np.zeros((12, 2))) for _ in range(2)]
                caches = [EvidenceCache(graphs[0], 'LocalLRU', capacity, fast),
                          EvidenceCache(graphs[1], 'LocalLRUMemo', capacity, fast)]
                for i in range(120):
                    if i % 3 == 0:
                        edge = rng.randrange(2), rng.randrange(12), rng.randrange(12), bool(rng.randrange(2))
                        for g in graphs:
                            g.edge(*edge)
                    if i % 7 == 0:
                        vertex, value = rng.randrange(12), [rng.random(), rng.random()]
                        for g in graphs:
                            g.vector(vertex, value)
                    anchor = rng.randrange(3)
                    path = tuple(rng.randrange(2) for _ in range(i % 7))
                    self.assertEqual(self.query(caches[0], path, anchor),
                                     self.query(caches[1], path, anchor))
                    self.same_persistent_state(*caches)

    def test_profile_primary_equivalence_and_nonoverlapping_phases(self):
        caches = [EvidenceCache(self.nested(), 'LocalLRUMemo', 512, profile=p)
                  for p in (False, True)]
        for i in range(40):
            if i % 5 == 0:
                for c in caches:
                    c.g.edge(0, 2, 3, bool(i % 2))
            path = (0,)*(i % 7)
            actual = self.query(caches[0], path)
            start = perf_counter_ns()
            profiled = caches[1].query(0, path, np.zeros(2))
            external = perf_counter_ns()-start
            self.assertEqual(actual, profiled)
            self.assertEqual(caches[0].stats, caches[1].stats)
            self.assertEqual(caches[0].entries, caches[1].entries)
            self.assertEqual(caches[0].last_memo_peak_bytes, caches[1].last_memo_peak_bytes)
            self.assertEqual(caches[0].last_memo_entries, caches[1].last_memo_entries)
            p = caches[1].last_profile_ns
            names = ('lookup_validation_ns', 'graph_ns', 'certificate_admission_ns', 'ranking_ns')
            self.assertGreaterEqual(p['residual_ns'], 0)
            self.assertEqual(sum(p[n] for n in names)+p['residual_ns'], p['total_ns'])
            self.assertLessEqual(p['total_ns'], external)


if __name__ == '__main__':
    unittest.main()

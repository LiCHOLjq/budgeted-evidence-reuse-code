"""Quick semantic tests; no benchmark timings or speed assertions.

Run from experiments/: python -m unittest -v test_engine_main
"""
import random
import unittest
from unittest.mock import patch
import numpy as np
import engine_main as main


class MainExperimentTests(unittest.TestCase):
    def graph(self, n=9):
        adj = [[set() for _ in range(n)] for _ in range(3)]
        adj[0][0] = {1, 2, 3}
        adj[1][1], adj[1][2] = {4}, {5}
        adj[2][4] = {6}
        return main.Graph(adj, np.arange(n*3).reshape(n, 3))

    def invariant(self, cache):
        if isinstance(cache, main.AdaptiveGate):
            self.invariant(cache.cache)
            self.assertEqual(cache.used, cache.cache.used+cache.GATE_BYTES)
        elif hasattr(cache, 'reverse'):
            expected = {}
            for key, entry in cache.entries.items():
                self.assertEqual(entry.size, cache.charge(key, entry.endpoints, entry.deps))
                for bucket in entry.deps:
                    expected.setdefault(bucket, set()).add(key)
            self.assertEqual(cache.reverse, expected)
            expected_index = 64+24*len(expected) if expected else 0
            self.assertEqual(cache.index_bytes, expected_index)
            self.assertEqual(cache.used, sum(e.size for e in cache.entries.values())+expected_index)
        else:
            self.assertEqual(cache.used, sum(e.size for e in cache.entries.values()))
        self.assertLessEqual(cache.used, cache.capacity)
        self.assertGreaterEqual(cache.used, 0)

    def check(self, cache, anchor=0, path=(0, 1), k=10):
        q = np.array([1, 3, 5])
        actual = cache.query(anchor, path, q, k)
        expected = main.oracle(cache.g, anchor, path, q, k)
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0], main.independent_rank(cache.g.vectors, actual[1], q, k))
        self.invariant(cache)
        return actual

    def test_original_controls_are_original_instances(self):
        for method in main.comparisons.METHODS:
            c = main.EvidenceCache(self.graph(), method, 4096)
            self.assertIs(type(c), main.comparisons.EvidenceCache)
        c = main.EvidenceCache(self.graph(), 'LocalLRUMemo', 4096)
        self.assertIs(type(c), main.replication.EvidenceCache)

    def test_reverse_charge_and_empty_dependency_invalidation(self):
        c = main.EvidenceCache(self.graph(), 'EagerIndexLRU', 4096)
        self.check(c)
        self.assertIn(12, c.reverse)  # Relation1, source3: initially empty.
        before = c.stats['invalidated']
        self.assertTrue(main.apply_edge(c, 1, 3, 7, True))
        self.assertGreater(c.stats['invalidated'], before)
        self.assertNotIn((0, (0, 1)), c.entries)
        self.assertIn(7, self.check(c)[1])
        self.assertEqual(c.stats['validated_buckets'], 0)

    def test_reverse_parent_eviction_does_not_drop_descendant_dependencies(self):
        c = main.EvidenceCache(self.graph(), 'EagerIndexLRU', 4096)
        self.check(c)
        c.drop((0, (0,)), 'evicted')
        self.invariant(c)
        self.assertIn((0, (0, 1)), c.reverse[0])
        main.apply_edge(c, 0, 0, 1, False)
        self.assertFalse(c.entries)
        self.assertNotIn(4, self.check(c)[1])

    def test_unrelated_vector_noop_and_aba_changes(self):
        c = main.EvidenceCache(self.graph(), 'EagerIndexLRU', 4096)
        self.check(c)
        e = c.entries[(0, (0, 1))]
        main.apply_edge(c, 2, 8, 7, True)
        self.assertIs(c.entries[(0, (0, 1))], e)
        self.assertFalse(main.apply_edge(c, 2, 8, 7, True))
        c.g.vector(4, [-5, -5, -5])
        self.check(c)
        self.assertIs(c.entries[(0, (0, 1))], e)
        main.apply_edge(c, 1, 3, 7, True)
        main.apply_edge(c, 1, 3, 7, False)
        self.assertNotIn((0, (0, 1)), c.entries)
        self.check(c)

    def test_missed_mutation_notification_fails_closed(self):
        c = main.EvidenceCache(self.graph(), 'EagerIndexLRU', 4096)
        self.check(c)
        c.g.edge(1, 3, 7, True)
        self.assertIn(7, self.check(c)[1])
        self.assertEqual(c.stats['missed_notification_flushes'], 1)
        c.g.edge(1, 3, 7, False)
        main.apply_edge(c, 2, 8, 7, True)  # Notification gap discovered here.
        self.assertFalse(c.entries)
        self.check(c)

    def test_eviction_replacement_and_zero_capacity_cleanup(self):
        for capacity in (0, 80, 200, 350, 512, 1024):
            c = main.EvidenceCache(self.graph(), 'EagerIndexLRU', capacity)
            for anchor in range(9):
                self.check(c, anchor, (0, 1, 2))
            for key in list(c.entries):
                c.drop(key, 'evicted')
            self.assertEqual(c.used, 0)
            self.assertFalse(c.reverse)
            self.invariant(c)
        c = main.EvidenceCache(self.graph(), 'EagerIndexLRU', 4096)
        c.put((0, (0,)), frozenset({1}), {0: 0}, 1)
        c.put((0, (0,)), frozenset(), {8: 0}, 0)
        self.assertEqual(set(c.reverse), {8})
        self.invariant(c)
        with self.assertRaises(ValueError):
            c.put((0, (0,)), frozenset(), {-1: 0}, 0)

    def test_diagnostic_semantics_match_primary_all_controls(self):
        rng = random.Random(117)
        for method in main.replication.METHODS+['EagerIndexLRU']:
            a = main.EvidenceCache(self.graph(), method, 1300, hybrid_tau=2)
            b = main.EvidenceCache(self.graph(), method, 1300, hybrid_tau=2,
                                 profile=True, diagnostics=True)
            for step in range(30):
                rel, src, dst = rng.randrange(3), rng.randrange(9), rng.randrange(9)
                insert = dst not in a.g.adj[rel][src]
                if step%3 == 0:
                    main.apply_edge(a, rel, src, dst, insert)
                    main.apply_edge(b, rel, src, dst, insert)
                anchor = rng.randrange(3)
                path = tuple(rng.randrange(3) for _ in range(rng.randrange(1, 5)))
                self.assertEqual(self.check(a, anchor, path), self.check(b, anchor, path))
                self.assertEqual(a.entries, b.entries)
                self.assertEqual(a.stats, b.stats)
                self.assertEqual(a.used, b.used)
                phases = b.last_profile_ns
                self.assertEqual(sum(v for k,v in phases.items() if k != 'total_ns'), phases['total_ns'])
                self.assertGreaterEqual(phases['residual_ns'], 0)

    def test_conditional_validation_and_fastpath_are_separate(self):
        c = main.EvidenceCache(self.graph(), 'LocalLRU', 4096, diagnostics=True)
        self.check(c)
        self.check(c)
        self.assertEqual(c.last_diagnostics['epoch_fast_path_hits'], 1)
        self.assertEqual(c.last_diagnostics['validation_calls'], 0)
        main.apply_edge(c, 2, 8, 7, True)  # Unrelated graph change forces validation.
        self.check(c)
        d = c.last_diagnostics
        self.assertEqual(d['queries_after_graph_update'], 1)
        self.assertEqual(d['validation_calls'], 1)
        self.assertEqual(d['validation_tokens'], 4)
        self.assertEqual(d['final_hits'], 1)
        self.assertEqual(d['epoch_fast_path_hits'], 0)
        self.assertGreater(d['validation_ns'], 0)
        main.apply_edge(c, 1, 3, 7, True)
        self.check(c)
        self.assertEqual(c.last_diagnostics['validation_failures'], 1)
        self.assertEqual(c.last_diagnostics['prefix_hits'], 1)
        self.assertEqual(c.last_diagnostics['query_misses'], 0)

    def test_before_update_overlap_does_not_modify_state(self):
        c = main.EvidenceCache(self.graph(), 'LocalLRU', 4096)
        self.check(c)
        saved = (dict(c.stats), list(c.entries.items()), c.used)
        d = main.overlap_audit(c, [0, 12, 26, 12])
        self.assertEqual(d['changed_bucket_count'], 3)
        self.assertEqual(d['overlap_bucket_count'], 2)
        self.assertEqual(d['resident_domain_buckets'], 4)
        self.assertEqual(d['affected_resident_entries'], 2)
        self.assertEqual(saved, (dict(c.stats), list(c.entries.items()), c.used))
        h = main.EvidenceCache(self.graph(), 'LocalHybrid', 4096, hybrid_tau=0)
        self.check(h)
        d = main.overlap_audit(h, [8, 17, 26])
        self.assertEqual(d['overlap_bucket_count'], 2)
        self.assertEqual(d['resident_domain_buckets'], 18)
        self.assertEqual(d['coverage_kind'], 'conservative_token_domain')

    def test_footprint_and_snapshot_have_declared_scope(self):
        for method in ('LocalLRU', 'EagerIndexLRU', 'AdaptiveGate'):
            c = main.EvidenceCache(self.graph(), method, 4096)
            self.check(c)
            d = main.cache_snapshot(c, footprint=True)
            self.assertEqual(d['used_bytes'], c.used)
            self.assertGreater(d['resident_python_bytes'], 0)
            self.assertEqual(d['used_bytes'], d['entry_charge_bytes']+d['index_bytes']+d['gate_bytes'])
            self.assertEqual(d['resident_entries'], len(c.entries))

    def test_independent_ranking_ties_empty_and_large_k(self):
        vectors = np.array([[1, 0], [-1, 0], [0, 0], [0, 1]], dtype=np.float64)
        for ids in (set(), {0, 1, 2, 3}, {0, 1, 3}):
            for k in (0, 1, 2, 10):
                self.assertEqual(main.rank(vectors, ids, np.zeros(2), k),
                                 main.independent_rank(vectors, ids, np.zeros(2), k))
        self.assertEqual(main.independent_rank(vectors, {0, 1, 3}, [0, 0], 10), (0, 1, 3))
        with self.assertRaises(ValueError):
            main.independent_rank(vectors, {0}, [0, 0], -1)

    def test_gate_probe_schedule_and_no_free_shadow(self):
        c = main.EvidenceCache(self.graph(), 'AdaptiveGate', 4096,
                             gate_warmup=4, gate_period=8, gate_block=2)
        counts, probes = [], []
        for i in range(23):
            self.check(c)
            if c.last_gate['probe']:
                probes.append(i+1)
            counts.append(c.stats['gate_probes'])
        self.assertEqual(probes, [1, 2, 3, 4, 13, 14, 21, 22])
        self.assertEqual(c.stats['gate_blocks'], 3)
        self.assertEqual(c.stats['gate_cache_queries']+c.stats['gate_memo_queries'], 23)
        self.assertEqual(c.used, 256+c.cache.used)
        c = main.EvidenceCache(self.graph(), 'AdaptiveGate', 4096)
        self.check(c)
        self.assertEqual(c.last_memo_peak_bytes,
                         max(c.cache.last_memo_peak_bytes, c.memo.last_memo_peak_bytes))

    def test_gate_causal_selection_and_dormant_validation(self):
        c = main.EvidenceCache(self.graph(), 'AdaptiveGate', 4096,
                             gate_warmup=2, gate_period=4, gate_block=2, gate_seed=0)
        # Even probe: cache=10, memo=100; odd reverses execution order.
        with patch.object(main, 'perf_counter_ns', side_effect=[0,10,20,120,200,300,310,320]):
            self.check(c)
            self.assertEqual(c.selected, 'EagerMemo')
            self.check(c)
        self.assertEqual(c.selected, 'LocalLRUMemo')
        self.assertEqual(c.last_gate['served_by'], 'EagerMemo')  # Decision affects next request.
        self.assertEqual(c.cache_estimate, 10)
        self.assertEqual(c.memo_estimate, 100)
        c.selected = 'EagerMemo'  # Exercise dormant-cache correctness, not policy speed.
        e = c.entries[(0, (0, 1))]
        main.apply_edge(c, 1, 3, 7, True)
        self.check(c)
        self.assertIs(c.entries[(0, (0, 1))], e)
        c.selected = 'LocalLRUMemo'
        self.assertIn(7, self.check(c)[1])
        self.assertIsNot(c.entries[(0, (0, 1))], e)

    def test_randomized_bursts_and_vector_updates_all_new_paths(self):
        rng = random.Random(9127)
        for method in ('EagerIndexLRU', 'AdaptiveGate'):
            for capacity in (512, 4096):
                c = main.EvidenceCache(self.graph(), method, capacity)
                for i in range(300):
                    for _ in range(rng.randrange(5)):
                        rel, src, dst = rng.randrange(3), rng.randrange(9), rng.randrange(9)
                        main.apply_edge(c, rel, src, dst, bool(rng.randrange(2)))
                        self.invariant(c)
                    if i%3 == 0:
                        c.g.vector(rng.randrange(9), [rng.randrange(-10, 10) for _ in range(3)])
                    anchor = rng.randrange(9)
                    path = tuple(rng.randrange(3) for _ in range(rng.randrange(6)))
                    self.check(c, anchor, path, rng.randrange(12))


if __name__ == '__main__':
    unittest.main()

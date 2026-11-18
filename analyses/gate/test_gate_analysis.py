"""Tests of definitions that could materially change the reported conclusions."""
import math
import unittest

from analyze_gate_analysis import (confusion, correlation, decomposition, longest_true,
                             process_ratio, ranks, routes)


class GateDefinitions(unittest.TestCase):
    def test_different_denominators_are_not_subtracted(self):
        d=decomposition(130,110,100,10)
        self.assertAlmostEqual(d['excess'],.30)
        self.assertAlmostEqual(d['paid'],.10)
        self.assertAlmostEqual(d['remainder'],.20)
        self.assertAlmostEqual(d['paid_over_gate'],10/130)
        self.assertNotAlmostEqual(d['remainder'],d['excess']-d['paid_over_gate'])

    def test_best_static_is_per_matched_run_not_pooled(self):
        a=decomposition(120,100,200,10)
        b=decomposition(120,200,100,10)
        macro=(a['excess']+b['excess'])/2
        pooled=decomposition(240,300,300,20)['excess']
        self.assertAlmostEqual(macro,.2)
        self.assertAlmostEqual(pooled,-.2)

    def test_probe_executes_unreturned_candidate(self):
        ledger=[dict(served_by='EagerMemo',probe='False'),
                dict(served_by='EagerMemo',probe='True'),
                dict(served_by='LocalLRUMemo',probe='False'),
                dict(served_by='LocalLRUMemo',probe='True')]
        r=routes(ledger,list(range(4)))
        self.assertEqual(r['cache_served_fraction'],.5)
        self.assertEqual(r['cache_candidate_executed_fraction'],.75)
        self.assertEqual(r['dormant_requests'],1)
        self.assertEqual(r['cache_served_nonprobe_fraction'],.5)

    def test_margin_can_only_remove_sign_positive_classifications(self):
        estimates=[(.90,1),(.98,1),(1.01,1),(.95,1)]
        sign=[c<m for c,m in estimates]
        margin=[c<=.95*m for c,m in estimates]
        self.assertTrue(all(not b or a for a,b in zip(sign,margin)))
        self.assertEqual(sign,[True,True,False,True])
        self.assertEqual(margin,[True,False,False,True])
        truth=[True,True,False,False]
        self.assertEqual(confusion(sign,truth)['FN'],0)
        self.assertEqual(confusion(margin,truth)['FN'],1)

    def test_mean_process_ratios_not_ratio_of_global_means(self):
        rr=[dict(process_repeat=i,a=a,b=b) for i,(a,b) in enumerate([(2,1),(20,20),(300,100)])]
        r=process_ratio(rr,'a','b')
        self.assertEqual(r['mean'],2)
        self.assertNotAlmostEqual(r['mean'],322/121)

    def test_rank_ties_and_dimensionless_correlations(self):
        self.assertEqual(ranks([3,1,1,2]),[3,.5,.5,2])
        r=correlation([1,2,4,8],[2,8,32,128])
        self.assertAlmostEqual(r['pearson_log_ratio'],1)
        self.assertAlmostEqual(r['spearman_ratio'],1)
        self.assertIsNone(correlation([1,1,1],[1,2,3])['pearson_log_ratio'])

    def test_dormancy_streak_resets_on_paid_probe(self):
        self.assertEqual(longest_true([True,True,False,True,True,True]),3)


if __name__=='__main__':
    unittest.main(verbosity=2)

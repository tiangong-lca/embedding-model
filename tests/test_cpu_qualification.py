"""Numerical/identity guards for offline CPU qualification (no model download)."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('cpu_qualification', Path(__file__).parents[1] / 'scripts/tools/qualify_cpu_embedding.py')
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


class QualityMetricTests(unittest.TestCase):
    def test_exact_ranking_relevance_and_rank_denominator(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('NumPy is required; run in qualification runtime')
        q = np.array([[1., 0.], [0., 1.]])
        c = np.array([[1., 0.], [0., 1.], [0.6, 0.8]])
        rel = {'q1': {'d1': 1}, 'q2': {'d3': 1}}
        result = qualification.retrieval_metrics(q, c, ['q1', 'q2'], ['d1', 'd2', 'd3'], rel)
        self.assertEqual(result['mean']['Recall@1'], 0.5)
        self.assertEqual(result['mean']['Recall@5'], 1)
        self.assertEqual(result['mean']['MRR@5'], 0.75)
        self.assertAlmostEqual(result['per_query']['q2']['NDCG@5'], 1 / __import__('math').log2(3))

    def test_subset_and_multiple_judged_positives(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('NumPy is required; run in qualification runtime')
        q = np.array([[1., 0.]])
        c = np.array([[1., 0.], [0., 1.]])
        result = qualification.retrieval_metrics(q, c, ['q'], ['a', 'b'], {'q': {'a': 1, 'b': 1}}, {'q'})
        self.assertEqual(result['mean']['Recall@1'], 0.5)
        self.assertEqual(result['mean']['NDCG@5'], 1)
        with self.assertRaisesRegex(ValueError, 'No evaluable'):
            qualification.retrieval_metrics(q, c, ['q'], ['a', 'b'], {'q': {'a': 1}}, {'absent'})

    def test_percentiles_retain_sample_count(self):
        result = qualification.latency_summary([0.1, 0.2, 0.3])
        self.assertEqual(result['completed'], 3)
        self.assertEqual(result['p50_ms'], 200)
        self.assertAlmostEqual(result['p95_ms'], 290)
        self.assertIn('not a production', result['p99_estimation'])


if __name__ == '__main__':
    unittest.main()

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

class ProvenanceGuardTests(unittest.TestCase):
    def test_content_and_semantic_drift_cannot_reuse_same_ids(self):
        import copy
        manifest = {'files_sha256': {'model.safetensors': 'same'}, 'dtype': 'fp32',
                    'max_length': 1024, 'pooling': 'last', 'normalization': 'L2',
                    'prompts': 'none', 'padding_side': 'right', 'runtime': {'torch': '2.6.0'}}
        meta = {'manifest': manifest, 'dataset': {'file_sha256': {'corpus.jsonl': 'original'}}}
        data = {'file_sha256': {'corpus.jsonl': 'original'}}
        ids = (['q'], ['d'])
        self.assertEqual(qualification.validate_comparison_identity(meta, meta, data, ids, ids), {})
        changed = copy.deepcopy(meta)
        changed['manifest']['max_length'] = 512
        with self.assertRaisesRegex(ValueError, 'max_length'):
            qualification.validate_comparison_identity(changed, meta, data, ids, ids)
        changed = copy.deepcopy(meta)
        changed['dataset']['file_sha256']['corpus.jsonl'] = 'different text, same id'
        with self.assertRaisesRegex(ValueError, 'dataset content'):
            qualification.validate_comparison_identity(changed, meta, data, ids, ids)
        with self.assertRaisesRegex(ValueError, 'ID order'):
            qualification.validate_comparison_identity(meta, meta, data, ids, (['q'], ['different']))


if __name__ == '__main__':
    unittest.main()

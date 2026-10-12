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

class OutputIntegrityTests(unittest.TestCase):
    def test_tampered_vectors_or_ids_cannot_reuse_completed_manifest(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            for name in qualification.OUTPUT_FILES:
                (Path(folder) / name).write_bytes(b'original')
            manifest = {'output_sha256': {name: qualification.sha256(Path(folder) / name) for name in qualification.OUTPUT_FILES}}
            qualification.verify_output_hashes(folder, manifest)
            (Path(folder) / 'queries_embeddings.npy').write_bytes(b'partial replacement')
            with self.assertRaisesRegex(ValueError, 'content hash differs'):
                qualification.verify_output_hashes(folder, manifest)

    def test_existing_partial_output_is_preserved_and_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'partial'
            qualification.fresh_output(out)
            sentinel = out / 'queries_embeddings.npy'
            sentinel.write_bytes(b'partial run evidence')
            with self.assertRaises(FileExistsError):
                qualification.fresh_output(out)
            self.assertEqual(sentinel.read_bytes(), b'partial run evidence')

class BenchmarkControlTests(unittest.TestCase):
    def test_serialized_queue_and_per_call_checkpoints(self):
        import tempfile
        import time
        from types import SimpleNamespace
        from unittest.mock import patch
        try:
            import numpy as np
        except ImportError:
            self.skipTest('NumPy is required; run in qualification runtime')
        class Encoder:
            meta = {}
            def __init__(self, args):
                self.np = np
                self.max_length = 1024
            def sized_text(self, length):
                return 'fixture'
            def encode(self, texts):
                time.sleep(0.02)
                return np.ones((1, 1024), dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'run'
            args = SimpleNamespace(output=str(out), execution_mode='serialized',
                                   lengths=[32], concurrency=[2], requests=4, warmup=1, cell_seconds=2)
            with patch.object(qualification, 'ResidentEncoder', Encoder):
                result = qualification.benchmark(args)
            raw = result['cells'][0]['raw']
            self.assertEqual(len(raw), 4)
            self.assertGreater(max(row['queue_wait_seconds'] for row in raw), 0.01)
            self.assertTrue(all(row['seconds'] >= row['queue_wait_seconds'] for row in raw))
            self.assertEqual(len((out / 'progress-32-c2.jsonl').read_text().splitlines()), 4)

    def test_native_forward_watchdog_exits_owned_process_with_stage_receipt(self):
        import json
        import subprocess
        import sys
        import tempfile
        script = Path(__file__).parents[1] / 'scripts/tools/qualify_cpu_embedding.py'
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'timeout-run'
            code = f'''import importlib.util, time
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location('qualification', {str(script)!r})
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
class Encoder:
    meta = {{}}
    calls = 0
    def __init__(self, args): pass
    def sized_text(self, length): return 'fixture'
    def encode(self, texts):
        self.calls += 1
        if self.calls > 1: time.sleep(10)
q.ResidentEncoder = Encoder
q.benchmark(SimpleNamespace(output={str(out)!r},execution_mode='serialized',lengths=[32],concurrency=[1],requests=1,warmup=1,cell_seconds=0.05))
'''
            result = subprocess.run([sys.executable, '-c', code], timeout=3, capture_output=True)
            self.assertEqual(result.returncode, 124, result.stderr.decode())
            receipt = json.loads((out / 'cell-32-c1.json').read_text())
            self.assertTrue(receipt['interrupted'])
            self.assertEqual(receipt['phase'], 'warming')
            self.assertEqual(receipt['completed_checkpoints'], 0)

    def test_delayed_watchdog_cannot_claim_complete_partial_cell(self):
        import json
        import subprocess
        import sys
        import tempfile
        script = Path(__file__).parents[1] / 'scripts/tools/qualify_cpu_embedding.py'
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'delayed-watchdog'
            code = f'''import importlib.util, time
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location('qualification', {str(script)!r})
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
class DelayedTimer:
    def __init__(self, *args): pass
    def start(self): pass
    def cancel(self): pass
class Encoder:
    meta = {{}}
    def __init__(self, args): pass
    def sized_text(self, length): return 'fixture'
    def encode(self, texts): time.sleep(0.02)
q.threading.Timer = DelayedTimer
q.ResidentEncoder = Encoder
q.benchmark(SimpleNamespace(output={str(out)!r},execution_mode='serialized',lengths=[32],concurrency=[1],requests=2,warmup=1,cell_seconds=0.01))
'''
            result = subprocess.run([sys.executable, '-c', code], timeout=3, capture_output=True)
            self.assertEqual(result.returncode, 124, result.stderr.decode())
            receipt = json.loads((out / 'cell-32-c1.json').read_text())
            self.assertTrue(receipt['interrupted'])
            self.assertEqual(receipt['phase'], 'timing')
            self.assertEqual(receipt['completed_checkpoints'], 0)
            self.assertFalse((out / 'benchmark.json').exists())


if __name__ == '__main__':
    unittest.main()

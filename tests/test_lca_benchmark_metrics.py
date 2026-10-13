import json
import math
from pathlib import Path
import tempfile
import unittest

from src.lca_benchmark.metrics import evaluate, rank_scores


class BenchmarkMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write("queries.jsonl", [{"query_id": "q", "source_group": "g"}])
        self.ids = [f"d{i:02}" for i in range(12)]
        self.write("corpus.jsonl", [{"document_id": d} for d in self.ids])
        self.refs = [{"query_id": "q", "document_id": d, "relevance": 1 if i < 5 else 0} for i, d in enumerate(self.ids[:6])]
        self.write("qrels.jsonl", self.refs)

    def write(self, filename, records):
        (self.root / filename).write_text("".join(json.dumps(r) + "\n" for r in records))

    def ranks(self, ids=None):
        return [{"query_id": "q", "document_ids": self.ids[:10] if ids is None else ids}]

    def test_perfect_and_unjudged_distinct_from_zero(self):
        result = evaluate(self.root, self.ranks())
        self.assertTrue(all(v == 1 for v in result["metrics"].values()))
        self.assertEqual(result["unjudged_returns"], {"5": 0, "10": 4})

    def test_fractional_fixed_five_not_generic_idcg(self):
        self.write("qrels.jsonl", [{"query_id": "q", "document_id": d, "relevance": .5} for d in self.ids[:10]])
        result = evaluate(self.root, self.ranks())
        self.assertAlmostEqual(result["metrics"]["ndcg_at_5"], .5)
        self.assertAlmostEqual(result["metrics"]["ndcg_at_10"], .5 * sum(1/math.log2(i+1) for i in range(1, 11)) / sum(1/math.log2(i+1) for i in range(1, 6)))

    def test_reject_invalid_rankings(self):
        cases = [[], self.ranks() * 2, [{"query_id": "wrong", "document_ids": self.ids[:10]}],
                 self.ranks(self.ids[:9]), self.ranks(["missing"] + self.ids[1:10]), self.ranks([self.ids[1]] + self.ids[1:10])]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                evaluate(self.root, case)

    def test_reject_invalid_references_and_duplicate_query(self):
        for extra in [self.refs[0], {"query_id": "q", "document_id": "missing", "relevance": 0},
                      {"query_id": "q", "document_id": "d11", "relevance": float("nan")}]:
            self.write("qrels.jsonl", self.refs + [extra])
            with self.assertRaises(ValueError):
                evaluate(self.root, self.ranks())
        self.write("qrels.jsonl", self.refs[:-2])
        with self.assertRaises(ValueError):
            evaluate(self.root, self.ranks())
        self.write("queries.jsonl", [{"query_id": "q"}] * 2)
        with self.assertRaises(ValueError):
            evaluate(self.root, self.ranks())

    def test_ties_are_id_sorted_and_nonfinite_rejected(self):
        self.assertEqual(rank_scores(self.ids[::-1], [1.] * 12), self.ids[:10])
        with self.assertRaises(ValueError):
            rank_scores(self.ids, [float("nan")] * 12)


if __name__ == "__main__":
    unittest.main()

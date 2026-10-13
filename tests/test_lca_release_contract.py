import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

from src.lca_benchmark.releases import artifact_path, download, read_manifest, verify_files, validate_dataset

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = b"release\n"
        self.manifest = {"schema_version": 1, "kind": "model", "repo_id": "Org/model", "revision": "a"*40,
                         "files": {"sub/file.txt": {"bytes": len(self.payload), "sha256": hashlib.sha256(self.payload).hexdigest()}}}

    def test_atomic_download_verified_cache_and_pinned_url(self):
        seen = []
        def opener(url, timeout):
            seen.append(url)
            return io.BytesIO(self.payload)
        self.assertEqual(download(self.manifest, self.root, opener), 1)
        self.assertEqual(download(self.manifest, self.root, opener), 0)
        self.assertEqual(seen, ["https://huggingface.co/Org/model/resolve/" + "a"*40 + "/sub/file.txt"])
        verify_files(self.manifest, self.root)

    def test_wrong_download_not_published(self):
        with self.assertRaises(ValueError):
            download(self.manifest, self.root, lambda *a, **kw: io.BytesIO(b"bad"))
        self.assertFalse((self.root / "sub/file.txt").exists())
        self.assertEqual(list((self.root / "sub").iterdir()), [])

    def test_oversize_and_transport_failure_remove_partial_files(self):
        for payload in (self.payload * 2, None):
            def opener(*args, **kwargs):
                if payload is None:
                    raise TimeoutError("transport unavailable")
                return io.BytesIO(payload)
            with self.assertRaises((ValueError, TimeoutError)):
                download(self.manifest, self.root, opener)
            self.assertEqual(list((self.root / "sub").iterdir()), [])

    def test_dataset_table_paths_must_be_checksummed(self):
        path = self.root / "manifest.json"
        candidate = dict(self.manifest, kind="training", row_counts={"../outside.jsonl": 1})
        path.write_text(json.dumps(candidate))
        with self.assertRaises(ValueError):
            read_manifest(path)

    def test_cli_offline_failure_is_json_and_nonzero(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/tools/download_lca_release.py"),
                                 "--manifest", str(ROOT / "config/releases/lca-nemotron-embed-1b.json"),
                                 "--output-dir", str(self.root), "--verify-only"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("Missing artifact", json.loads(result.stderr)["error"])

    def test_retrieval_rejects_existing_output_before_model_loading(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/tools/retrieve_lca_benchmark.py"),
                                 "--model-dir", str(self.root / "absent-model"), "--data-dir", str(self.root),
                                 "--output-dir", str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Output directory already exists", json.loads(result.stderr)["error"])

    def test_wrong_benchmark_protocol_rejected(self):
        value = json.loads((ROOT / "config/datasets/lca-retrieval-benchmark-v1.json").read_text())
        value["protocol"] = "generic-ndcg"
        path = self.root / "manifest.json"
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "Unsupported benchmark protocol"):
            read_manifest(path)

    def test_training_joins_grades_and_retention(self):
        tables = {
            "queries.jsonl": [{"query_id": "q", "text": "query", "anchor_document_id": "d"}],
            "corpus.jsonl": [{"document_id": "d", "text": "document"}],
            "relations.jsonl": [{"query_id": "q", "document_id": "d", "grade": 4, "label": "direct", "role": "primary"}],
            "sampling.jsonl": [{"draw_id": "draw", "query_id": "q", "position": 0}],
            "retention.jsonl": [{"query_id": "q", "document_id": "d", "teacher_cosine_score": 0.8}],
        }
        manifest = {"kind": "training", "row_counts": {n: len(v) for n, v in tables.items()}, "retention_query_count": 1}
        def save():
            for name, values in tables.items():
                (self.root / name).write_text("".join(json.dumps(r) + "\n" for r in values))
        save()
        self.assertEqual(validate_dataset(manifest, self.root), manifest["row_counts"])
        for name, field, bad in [("queries.jsonl", "anchor_document_id", "missing"),
                                 ("relations.jsonl", "grade", 1),
                                 ("relations.jsonl", "better_document_ids", ["missing"]),
                                 ("sampling.jsonl", "query_id", "missing"),
                                 ("sampling.jsonl", "position", 1),
                                 ("retention.jsonl", "document_id", "missing"),
                                 ("retention.jsonl", "teacher_cosine_score", float("nan"))]:
            original = dict(tables[name][0])
            tables[name][0][field] = bad
            save()
            with self.subTest(name=name, field=field), self.assertRaises(ValueError):
                validate_dataset(manifest, self.root)
            tables[name][0] = original

    def test_existing_corruption_not_overwritten(self):
        path = self.root / "sub/file.txt"
        path.parent.mkdir()
        path.write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            download(self.manifest, self.root, lambda *a, **kw: self.fail("must not download"))
        self.assertEqual(path.read_bytes(), b"corrupt")

    def test_symlinks_and_traversal_rejected(self):
        (self.root / "link").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            artifact_path(self.root, "link/a")
        path = self.root / "manifest.json"
        for filename in ("../escape", "/absolute", "a/../b", "a\\b", "a//b"):
            candidate = dict(self.manifest, files={filename: next(iter(self.manifest["files"].values()))})
            path.write_text(json.dumps(candidate))
            with self.assertRaises(ValueError):
                read_manifest(path)
        path.write_text(json.dumps(dict(self.manifest, revision="main")))
        with self.assertRaises(ValueError):
            read_manifest(path)

    def test_published_manifests_and_separate_protocols(self):
        paths = [ROOT / "config/releases/lca-nemotron-embed-1b.json",
                 ROOT / "config/datasets/lca-embedding-training-v1.json",
                 ROOT / "config/datasets/lca-retrieval-benchmark-v1.json"]
        values = [read_manifest(p) for p in paths]
        self.assertEqual([v["kind"] for v in values], ["model", "training", "benchmark"])
        self.assertNotEqual(values[1]["files"]["corpus.jsonl"]["sha256"], values[2]["files"]["corpus.jsonl"]["sha256"])
        self.assertEqual(values[2]["row_counts"]["queries.jsonl"], 100)
        self.assertEqual(values[0]["inference"]["dimension"], 2048)


if __name__ == "__main__":
    unittest.main()

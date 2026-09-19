"""Executable dependency and ML compatibility boundaries."""

from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class DependencyContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pyproject = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
        cls.lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text())
        cls.specifiers = {
            dependency.split("[", 1)[0].split("=", 1)[0].split("<", 1)[0].split(">", 1)[0]: dependency
            for dependency in cls.pyproject["project"]["dependencies"]
        }
        cls.locked_versions = {package["name"]: package["version"] for package in cls.lock["package"]}

    def test_reviewed_direct_dependency_contract(self) -> None:
        expected = {
            "accelerate": "accelerate>=1.14.0",
            "datasets": "datasets>=5.0.1",
            "deepspeed": "deepspeed>=0.19.6",
            "faiss-gpu-cu12": "faiss-gpu-cu12>=1.14.1.post1",
            "flagembedding": "flagembedding>=1.4.2",
            "gguf": "gguf>=0.19.0",
            "loguru": "loguru>=0.7.3",
            "matplotlib": "matplotlib>=3.11.1",
            "modelscope": "modelscope>=1.39.1",
            "openai": "openai>=3.6.0",
            "peft": "peft==0.13.0",
            "python-dotenv": "python-dotenv>=1.2.3",
            "pytrec-eval": "pytrec-eval>=0.5",
            "requests": "requests>=2.34.2",
            "sentence-transformers": "sentence-transformers>=5.7.0,<6.0.0",
            "supabase": "supabase>=2.31.0",
            "tidas-sdk": "tidas-sdk==0.2.14",
            "torch": "torch==2.9.1",
            "transformers": "transformers==4.51.3",
        }
        self.assertEqual(self.specifiers, expected)

    def test_lock_resolves_latest_compatible_direct_versions(self) -> None:
        expected = {
            "accelerate": "1.14.0",
            "datasets": "5.0.1",
            "deepspeed": "0.19.6",
            "faiss-gpu-cu12": "1.14.1.post1",
            "flagembedding": "1.4.2",
            "gguf": "0.19.0",
            "loguru": "0.7.3",
            "matplotlib": "3.11.1",
            "modelscope": "1.39.1",
            "openai": "3.6.0",
            "peft": "0.13.0",
            "python-dotenv": "1.2.3",
            "pytrec-eval": "0.5",
            "requests": "2.34.2",
            "sentence-transformers": "5.7.0",
            "supabase": "2.31.0",
            "tidas-sdk": "0.2.14",
            "torch": "2.9.1",
            "transformers": "4.51.3",
        }
        self.assertEqual(
            {name: self.locked_versions.get(name) for name in expected},
            expected,
        )

    def test_cuda12_training_boundary_rejects_cuda13_drift(self) -> None:
        self.assertFalse(any("cu13" in name for name in self.locked_versions))
        self.assertEqual(self.locked_versions["faiss-gpu-cu12"], "1.14.1.post1")
        self.assertEqual(self.locked_versions["torch"], "2.9.1")


if __name__ == "__main__":
    unittest.main()

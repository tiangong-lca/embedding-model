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
            "tidas-sdk": "tidas-sdk==0.3.0",
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
            "matplotlib": "3.11.2",
            "modelscope": "1.39.1",
            "openai": "3.35.1",
            "peft": "0.13.0",
            "python-dotenv": "1.2.4",
            "pytrec-eval": "0.5",
            "requests": "2.34.2",
            "sentence-transformers": "5.7.0",
            "supabase": "2.33.0",
            "tidas-sdk": "0.3.0",
            "torch": "2.9.1",
            "transformers": "4.51.3",
        }
        self.assertEqual(
            {name: self.locked_versions.get(name) for name in expected},
            expected,
        )

    def test_only_openai_has_an_explicit_official_index(self) -> None:
        uv = self.pyproject["tool"]["uv"]
        self.assertEqual(uv["index-url"], "https://mirrors.aliyun.com/pypi/simple")
        self.assertEqual(uv["sources"], {"openai": {"index": "openai-pypi"}})
        self.assertEqual(uv["index"], [{"name": "openai-pypi", "url": "https://pypi.org/simple", "explicit": True}])
        registries = {package["name"]: package["source"]["registry"] for package in self.lock["package"] if "registry" in package["source"]}
        self.assertEqual(registries.pop("openai"), "https://pypi.org/simple")
        self.assertEqual(set(registries.values()), {"https://mirrors.aliyun.com/pypi/simple"})
        self.assertNotIn("index-strategy", uv)

    def test_accepted_security_versions_remain_locked(self) -> None:
        self.assertEqual(
            {name: self.locked_versions[name] for name in ("pyjwt", "multidict", "urllib3")},
            {"pyjwt": "2.15.1", "multidict": "6.9.1", "urllib3": "2.8.0"},
        )

    def test_supabase_family_stays_aligned(self) -> None:
        for name in ("supabase", "postgrest", "storage3", "supabase-auth", "supabase-functions", "realtime"):
            self.assertEqual(self.locked_versions[name], "2.33.0")

    def test_cuda12_training_boundary_rejects_cuda13_drift(self) -> None:
        self.assertFalse(any("cu13" in name for name in self.locked_versions))
        self.assertEqual(self.locked_versions["faiss-gpu-cu12"], "1.14.1.post1")
        self.assertEqual(self.locked_versions["torch"], "2.9.1")


if __name__ == "__main__":
    unittest.main()

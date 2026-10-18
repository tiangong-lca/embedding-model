"""The isolated consumer proof must fail closed on installed dependency drift."""
from __future__ import annotations

import importlib.util
import json
import hashlib
from io import BytesIO
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/tools/run_portable_tests.py"
spec = importlib.util.spec_from_file_location("portable_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def distribution(name, version):
    return types.SimpleNamespace(metadata={"Name": name}, version=version)


class PortableEnvironmentTest(unittest.TestCase):
    def test_actual_installed_consumer_dependencies_are_all_locked(self):
        receipt = runner.environment_receipt()
        self.assertIn("numpy", receipt["installed"])
        self.assertIn("pydantic", receipt["installed"])
        self.assertIn("httpx", receipt["installed"])
        self.assertIn("httpx2", receipt["installed"])
        self.assertEqual(len(receipt["lock_sha256"]), 64)

    def test_audit_rejects_unlocked_transitives_missing_roots_and_training_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "uv.lock").write_text('[[package]]\nname="openai"\nversion="3.35.1"\n[[package]]\nname="torch"\nversion="2.9.1"\n')
            (root / "pyproject.toml").write_text('[project]\nname="fixture"\n')
            for distributions, message in [
                ([distribution("openai", "3.35.1"), distribution("unrecorded-transport", "1")], "drift"),
                ([], "missing"),
                ([distribution("openai", "3.35.1"), distribution("torch", "2.9.1")], "GPU/training"),
            ]:
                with self.subTest(message=message), \
                     patch.object(runner, "PORTABLE_PACKAGES", ("openai",)), \
                     patch.object(runner.metadata, "distributions", return_value=distributions), \
                     self.assertRaisesRegex(RuntimeError, message):
                    runner.environment_receipt(root)

    def test_explicit_source_uses_only_the_hash_bound_locked_universal_wheel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pyproject.toml").write_text('[tool.uv.sources]\nopenai={index="official"}\n[[tool.uv.index]]\nname="official"\nurl="https://pypi.org/simple"\nexplicit=true\n')
            url = "https://files.pythonhosted.org/openai-3.35.1-py3-none-any.whl"
            digest = "a" * 64
            lock = f'[[package]]\nname="openai"\nversion="3.35.1"\nsource={{registry="https://pypi.org/simple"}}\nwheels=[{{url="{url}",hash="sha256:{digest}",size=3}}]\n'
            (root / "uv.lock").write_text(lock)
            with patch.object(runner, "PORTABLE_PACKAGES", ("openai",)):
                artifacts = {"openai": {"path": str(root / "verified.whl")}}
                self.assertEqual(runner.portable_requirements(root / "constraints", root, artifacts).splitlines(), [f'-c "{(root / "constraints").as_posix()}"', f"openai @ {(root / 'verified.whl').as_uri()}"])
                with self.assertRaisesRegex(RuntimeError, "verified first"):
                    runner.portable_requirements(root / "constraints", root)
                (root / "uv.lock").write_text(lock.replace("-py3-none-any.whl", "-cp312-win_amd64.whl"))
                with self.assertRaisesRegex(RuntimeError, "universal"):
                    runner.explicit_portable_wheels(root)

    def test_bad_artifact_hash_stops_before_the_isolated_installer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = {"url": "https://files.pythonhosted.org/openai-3.35.1-py3-none-any.whl", "sha256": hashlib.sha256(b"good").hexdigest(), "size": 4}
            commands = []
            with patch.object(runner, "ROOT", root), \
                 patch.object(runner.shutil, "which", return_value="/uv"), \
                 patch.object(runner.subprocess, "check_output", return_value="uv 0.13.1 (test)"), \
                 patch.object(runner.subprocess, "run", side_effect=lambda command, **kwargs: commands.append(command)), \
                 patch.object(runner.sys, "version_info", (3, 12)), \
                 patch.object(runner, "explicit_portable_wheels", return_value={"openai": expected}), \
                 patch.object(runner.urllib.request, "urlopen", return_value=BytesIO(b"evil")), \
                 patch.dict(runner.os.environ, {"TMPDIR": temporary}), \
                 self.assertRaisesRegex(RuntimeError, "SHA256"):
                runner.run_suite(root)
            self.assertEqual([command[1] for command in commands], ["lock", "export"])
            self.assertEqual(list((root / "wheels").iterdir()), [])

    def test_frozen_constraint_launcher_selects_consumers_without_syncing_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pyproject.toml").write_text('[tool.uv]\nindex-url="https://example.invalid/simple"\n')
            (root / "uv.lock").write_text("package=[]\n")
            commands = []
            with patch.object(runner, "ROOT", root), \
                 patch.object(runner.shutil, "which", return_value="/uv"), \
                 patch.object(runner.subprocess, "check_output", return_value="uv 0.13.1 (test)"), \
                 patch.object(runner.subprocess, "run", side_effect=lambda command, **kwargs: commands.append(command)), \
                 patch.object(runner.sys, "version_info", (3, 12)), \
                 patch.dict(runner.os.environ, {"TMPDIR": temporary}):
                runner.run_suite(root)
            self.assertEqual(commands[0][1:3], ["lock", "--check"])
            self.assertIn("--frozen", commands[1])
            self.assertIn("--no-emit-project", commands[1])
            for command in commands[2:]:
                self.assertIn("--isolated", command)
                self.assertIn("--no-project", command)
                self.assertIn("--with-requirements", command)
                self.assertNotIn("sync", command)
            requirements = (root / "portable-requirements.txt").read_text().splitlines()
            self.assertEqual(requirements[0], f'-c "{(root / "root-constraints.txt").as_posix()}"')
            self.assertEqual(requirements[1:], list(runner.PORTABLE_PACKAGES))
            self.assertNotIn("torch", requirements)
            self.assertEqual(json.loads((root / "commands.json").read_text())["commands"], commands)

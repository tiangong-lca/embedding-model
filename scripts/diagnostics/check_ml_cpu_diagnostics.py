#!/usr/bin/env python3
"""Standard-library control tests; run explicitly, outside portable discovery."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from run_ml_cpu_diagnostics import SANDBOX_POLICY, disk_usage, group_usage, supervise


class SupervisorTests(unittest.TestCase):
    def run_child(self, code, root, **limits):
        return supervise([sys.executable, "-I", "-c", code], cwd=root, env=os.environ.copy(),
                         log=root/"child.log", timeout=limits.pop("timeout", 5),
                         deadline=time.monotonic()+10, disk_roots=[root], **limits)

    def test_completion_has_actual_rss_and_no_remaining_group(self):
        with tempfile.TemporaryDirectory() as folder:
            fact = self.run_child("import time; time.sleep(.2)", Path(folder))
            self.assertEqual(fact["returncode"], 0)
            self.assertIsNone(fact["limit"])
            self.assertGreater(fact["peak_group_rss_bytes"], 0)
            self.assertEqual(fact["remaining_group_pids_before_cleanup"], [])

    def test_timeout_reaps_only_its_owned_group(self):
        with tempfile.TemporaryDirectory() as folder:
            fact = self.run_child("import time; time.sleep(30)", Path(folder), timeout=.15)
            self.assertEqual(fact["limit"], "stage-timeout")
            self.assertEqual(group_usage(fact["pgid"])[1], [])

    def test_resident_memory_limit_uses_real_pages(self):
        with tempfile.TemporaryDirectory() as folder:
            fact = self.run_child("import time; data=bytearray(80*1024**2); time.sleep(30)",
                                  Path(folder), max_rss=40*1024**2)
            self.assertEqual(fact["limit"], "actual-group-rss")
            self.assertGreater(fact["peak_group_rss_bytes"], 40*1024**2)
            self.assertEqual(group_usage(fact["pgid"])[1], [])

    def test_disk_limit_refuses_to_start_process(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"input").write_bytes(b"xx")
            fact = self.run_child("raise AssertionError('must never run')", root, max_disk=1)
            self.assertEqual(fact["limit"], "disk-before-start")
            self.assertIsNone(fact["returncode"])

    def test_monitor_error_reaps_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            # A control-only mock tests supervisor failure handling, never an ML package.
            with patch("run_ml_cpu_diagnostics.group_usage", side_effect=RuntimeError("monitor failed")):
                with self.assertRaisesRegex(RuntimeError, "monitor failed"):
                    self.run_child("import time; time.sleep(30)", root)

    def test_monitor_error_after_leader_exit_still_kills_owned_descendant(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            children = []
            actual_popen = subprocess.Popen
            def start(*args, **kwargs):
                child = actual_popen(*args, **kwargs)
                children.append(child)
                return child
            def fail_after_exit(pgid):
                until = time.monotonic()+3
                while not (root/"descendant.pid").exists() and time.monotonic() < until:
                    time.sleep(.01)
                self.assertTrue((root/"descendant.pid").exists())
                children[0].wait(timeout=3)
                raise RuntimeError("leader exited before monitor failed")
            code = ("import subprocess,sys,pathlib; child=subprocess.Popen([sys.executable,'-c',"
                    "'import time; time.sleep(30)']); "
                    f"pathlib.Path({str(root/'descendant.pid')!r}).write_text(str(child.pid))")
            with patch("run_ml_cpu_diagnostics.subprocess.Popen", side_effect=start), patch(
                    "run_ml_cpu_diagnostics.group_usage", side_effect=fail_after_exit):
                with self.assertRaisesRegex(RuntimeError, "leader exited"):
                    self.run_child(code, root)
            until = time.monotonic()+3
            while group_usage(children[0].pid)[1] and time.monotonic() < until:
                time.sleep(.05)
            self.assertEqual(group_usage(children[0].pid)[1], [])

    def test_offline_audit_guard_blocks_dns(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            script = Path(__file__).with_name("ml_cpu_fixture.py")
            result = subprocess.run([sys.executable, "-I", str(script), "--stage", "network-guard",
                "--source", str(root), "--lock", str(root/"unused-lock"), "--build-receipt",
                str(root/"unused-build"), "--output", str(root/"output")],
                capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((root/"output/receipt.json").read_text())
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual(receipt["denied_operations"][0]["event"], "socket.getaddrinfo")
            self.assertFalse(receipt["accepts_original51"])

    @unittest.skipUnless(sys.platform == "darwin", "Native macOS sandbox control")
    def test_native_policy_denies_compiler_execution(self):
        result = subprocess.run(["/usr/bin/sandbox-exec", "-p", SANDBOX_POLICY,
                                 "/usr/bin/clang", "--version"],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 71)
        self.assertIn("Operation not permitted", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)

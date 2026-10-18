#!/usr/bin/env python3
"""Supervise one explicitly provisioned native macOS CPU row, without installing.

Requires a new evidence directory, exact frozen input files, a fresh owned
Python 3.12 venv and an artifact/build receipt. All stages remain separate from
ordinary portable tests. CPU success never accepts a CUDA training update.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

from ml_cpu_fixture import STAGES

SANDBOX_POLICY = r'''(version 1)(allow default)(deny network*)
(deny process-exec (regex #".*/(clang[^/]*|gcc[^/]*|g[+][+]|cc|c[+][+]|ninja|make|cmake|ld|as|rustc|cargo|nvcc|hipcc|ptxas)(-[0-9.]+)?$"))'''


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_usage(pgid):
    output = subprocess.check_output(["/bin/ps", "-axo", "pid=,pgid=,rss="], text=True)
    rows = [list(map(int, line.split())) for line in output.splitlines() if line.strip()]
    owned = [row for row in rows if row[1] == pgid]
    return sum(row[2] * 1024 for row in owned), [row[0] for row in owned]


def disk_usage(paths):
    # Count apparent bytes conservatively (including separate hard links).
    # Installers atomically remove temporary files while the monitor walks.
    total = 0
    for root in paths:
        for directory, _, files in os.walk(root, onerror=lambda error: (_ for _ in ()).throw(error)):
            for name in files:
                try:
                    total += (Path(directory)/name).lstat().st_size
                except FileNotFoundError:
                    continue
    return total


def terminate_group(process):
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            continue
    process.wait()


def supervise(command, *, cwd, env, log, timeout, deadline, disk_roots,
              max_disk=12*1024**3, max_rss=8*1024**3):
    """Measure actual group RSS; terminate only this new owned process group."""
    started = time.monotonic()
    peak_rss = 0
    peak_disk = disk_usage(disk_roots)
    if peak_disk > max_disk:
        return {"command": command, "returncode": None, "limit": "disk-before-start",
                "peak_disk_bytes": peak_disk, "peak_group_rss_bytes": 0}
    limit = None
    with Path(log).open("xb") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        next_disk = started
        try:
            while process.poll() is None:
                now = time.monotonic()
                rss, _ = group_usage(process.pid)
                peak_rss = max(peak_rss, rss)
                if now >= next_disk:
                    peak_disk = max(peak_disk, disk_usage(disk_roots))
                    next_disk = now + 2
                if rss > max_rss:
                    limit = "actual-group-rss"
                elif peak_disk > max_disk:
                    limit = "owned-disk"
                elif now >= deadline:
                    limit = "total-deadline"
                elif now-started >= timeout:
                    limit = "stage-timeout"
                if limit:
                    terminate_group(process)
                    break
                time.sleep(0.1)
            returncode = process.wait()
            _, remaining = group_usage(process.pid)
            if remaining:
                # A detached/background task is not valid fixture behavior.
                limit = limit or "remaining-owned-group"
                terminate_group(process)
        except BaseException:
            # Monitoring failure must not leave an unsupervised child running.
            # The leader may already have exited while its children remain.
            terminate_group(process)
            raise
    return {"command": command, "pid": process.pid, "pgid": process.pid,
            "returncode": returncode, "limit": limit, "seconds": time.monotonic()-started,
            "peak_group_rss_bytes": peak_rss, "peak_disk_bytes": peak_disk,
            "remaining_group_pids_before_cleanup": remaining}


def runtime_environment(resources, python):
    # Credentials, provider settings and ambient PYTHONPATH are not inherited.
    # HOME remains real; library-specific caches and temp files are all owned.
    env = {key: os.environ[key] for key in ("HOME", "USER", "LOGNAME", "LANG") if key in os.environ}
    cache = resources / "runtime-cache"
    cache.mkdir(exist_ok=True)
    temp = resources / "tmp"
    temp.mkdir(exist_ok=True)
    env.update(PATH=f"{Path(python).parent}:/usr/bin:/bin", TMPDIR=str(temp),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               CUDA_VISIBLE_DEVICES="", DS_ACCELERATOR="cpu", DS_BUILD_OPS="0",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               HF_HOME=str(cache/"huggingface"), MODELSCOPE_CACHE=str(cache/"modelscope"),
               MODELSCOPE_HOME=str(cache/"modelscope-home"),
               XDG_CACHE_HOME=str(cache), TORCH_HOME=str(cache/"torch"),
               TORCH_EXTENSIONS_DIR=str(cache/"no-native-extensions"),
               NUMBA_CACHE_DIR=str(cache/"numba"), TRITON_CACHE_DIR=str(cache/"triton"),
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2",
               HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", WANDB_DISABLED="true")
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--row", required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--build-receipt", type=Path, required=True)
    p.add_argument("--resources", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--total-seconds", type=float, default=2700)
    args = p.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        p.error("This explicitly bounded lane currently requires native macOS arm64")
    if args.total_seconds <= 0 or args.total_seconds > 2700:
        p.error("--total-seconds must be within (0, 2700]")
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[2]
    fixture = Path(__file__).with_name("ml_cpu_fixture.py")
    deadline = time.monotonic()+args.total_seconds
    env = runtime_environment(args.resources, args.python)
    sandbox = "/usr/bin/sandbox-exec"
    if not Path(sandbox).is_file():
        p.error("Native deny-network sandbox is unavailable; do not run unconfined")
    result = {"row": args.row, "started_at": datetime.now(timezone.utc).isoformat(),
              "source": str(source), "manifest_sha256": sha256(args.manifest),
              "lock_sha256": sha256(args.lock), "fixture_sha256": sha256(fixture),
              "supervisor_sha256": sha256(__file__), "sandbox_policy": SANDBOX_POLICY,
              "sandbox_sha256": sha256(sandbox), "stages": [], "accepts_original51": False}
    # Test native sandbox before loading any ML package. Audit hook is absent here.
    probe = [sandbox, "-p", SANDBOX_POLICY, str(args.python), "-I", "-c",
             "import socket; s=socket.socket();\ntry: s.bind(('127.0.0.1',0))\n"
             "except PermissionError: print('native network denial observed')\n"
             "else: raise SystemExit('NETWORK SANDBOX FAILED')"]
    native = supervise(probe, cwd=source, env=env, log=args.output/"native-network.log",
                       timeout=10, deadline=deadline, disk_roots=[args.resources, args.output, source])
    result["native_network_probe"] = native
    if native["returncode"] != 0 or native["limit"]:
        (args.output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
        raise SystemExit("Native network sandbox verification failed")
    for stage in STAGES:
        command = [sandbox, "-p", SANDBOX_POLICY, str(args.python), "-I", str(fixture),
                   "--stage", stage, "--source", str(source), "--lock", str(args.lock),
                   "--build-receipt", str(args.build_receipt), "--output", str(args.output/stage)]
        fact = supervise(command, cwd=source, env=env, log=args.output/f"{stage}.log",
                         timeout=300 if stage.removeprefix("ordered-") == "trainer" else 120, deadline=deadline,
                         disk_roots=[args.resources, args.output, source])
        receipt_path = args.output/stage/"receipt.json"
        fact.update(stage=stage, receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None)
        if fact["receipt"]:
            fact["peak_group_rss_bytes"] = max(fact["peak_group_rss_bytes"],
                       fact["receipt"]["actual_self_peak_rss_bytes"])
            if fact["peak_group_rss_bytes"] > 8*1024**3:
                fact["limit"] = fact["limit"] or "child-reported-peak-rss"
        result["stages"].append(fact)
        result["all_stages_passed"] = all(s["returncode"] == 0 and not s["limit"]
                 and s["receipt"] and s["receipt"]["status"] == "passed" for s in result["stages"])
        result["complete"] = len(result["stages"]) == len(STAGES)
        (args.output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({"row": args.row, "stage": stage, "returncode": fact["returncode"],
                          "limit": fact["limit"], "rss": fact["peak_group_rss_bytes"]}), flush=True)
        if fact["limit"]:
            break
    if not result["complete"] or not result["all_stages_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

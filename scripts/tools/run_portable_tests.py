#!/usr/bin/env python3
"""Run the portable consumer suite in a root-lock-constrained uv environment.

The exported requirements are disposable constraints, not another dependency
source. Only the listed consumer libraries and their locked dependencies install;
the GPU project is never synced or installed.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PORTABLE_PACKAGES = (
    "tidas-sdk", "matplotlib", "openai", "python-dotenv", "supabase", "tqdm", "loguru",
)


def canonical_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def locked_versions(root: Path = ROOT) -> dict[str, set[str]]:
    versions: dict[str, set[str]] = {}
    for package in tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))["package"]:
        versions.setdefault(canonical_name(package["name"]), set()).add(package["version"])
    return versions


def explicit_portable_wheels(root: Path = ROOT) -> dict[str, dict]:
    """Carry explicit index bindings through uv's requirements-format export.

    That format cannot express per-package index bindings. The reviewed OpenAI
    exception is pure Python, so its one universal wheel and SHA256 come
    directly from the root lock; no index-wide fallback is introduced.
    """
    uv = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8")).get("tool", {}).get("uv", {})
    indexes = {item["name"]: item for item in uv.get("index", [])}
    packages = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))["package"]
    result = {}
    for name in PORTABLE_PACKAGES:
        source = uv.get("sources", {}).get(name)
        if source is None:
            continue
        if not isinstance(source, dict) or set(source) != {"index"}:
            raise RuntimeError(f"Unsupported portable source binding for {name}")
        index = indexes[source["index"]]
        if index.get("explicit") is not True:
            raise RuntimeError(f"Portable source index must remain explicit: {name}")
        matches = [item for item in packages if item["name"] == name and item.get("source", {}).get("registry") == index["url"]]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one locked explicit package: {name}")
        wheels = [item for item in matches[0].get("wheels", []) if item["url"].endswith("-py3-none-any.whl")]
        if len(wheels) != 1 or not re.fullmatch(r"sha256:[0-9a-f]{64}", wheels[0]["hash"]):
            raise RuntimeError(f"Expected one hash-bound universal portable wheel: {name}")
        url = urllib.parse.urlsplit(wheels[0]["url"])
        if url.scheme != "https" or url.hostname != "files.pythonhosted.org" or url.username or url.password or url.query or url.fragment:
            raise RuntimeError(f"Expected a credential-free official wheel URL: {name}")
        if not isinstance(wheels[0].get("size"), int) or wheels[0]["size"] <= 0:
            raise RuntimeError(f"Expected a locked artifact size: {name}")
        result[name] = {"url": wheels[0]["url"], "sha256": wheels[0]["hash"].split(":", 1)[1], "size": wheels[0]["size"]}
    return result


def prepare_explicit_artifacts(evidence: Path, root: Path = ROOT) -> dict[str, dict]:
    """Verify exact locked wheel bytes before uv can execute the interpreter."""
    artifacts = {}
    directory = evidence / "wheels"
    directory.mkdir()
    for name, expected in explicit_portable_wheels(root).items():
        filename = Path(urllib.parse.urlsplit(expected["url"]).path).name
        path = directory / filename
        digest = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(expected["url"], timeout=30) as response, path.open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > expected["size"]:
                        raise RuntimeError(f"Locked artifact size mismatch: {name}")
                    digest.update(chunk)
                    output.write(chunk)
            if size != expected["size"] or digest.hexdigest() != expected["sha256"]:
                raise RuntimeError(f"Locked artifact SHA256/size mismatch: {name}")
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        artifacts[name] = {**expected, "path": str(path.resolve())}
    (evidence / "explicit-artifacts.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    return artifacts


def portable_requirements(constraints: Path, root: Path = ROOT, artifacts: dict | None = None) -> str:
    explicit = explicit_portable_wheels(root)
    requirements = [f'-c "{constraints.as_posix()}"']
    for name in PORTABLE_PACKAGES:
        if name in explicit:
            if artifacts is None or name not in artifacts:
                raise RuntimeError(f"Explicit portable artifact must be verified first: {name}")
            requirements.append(f"{name} @ {Path(artifacts[name]['path']).as_uri()}")
        else:
            requirements.append(name)
    return "\n".join(requirements) + "\n"


def environment_receipt(root: Path = ROOT) -> dict:
    """Reject an unrecorded dependency before any consumer test executes."""
    locked = locked_versions(root)
    installed = {
        canonical_name(distribution.metadata["Name"]): distribution.version
        for distribution in metadata.distributions()
    }
    missing = sorted(set(PORTABLE_PACKAGES) - installed.keys())
    drift = {
        name: version for name, version in installed.items()
        if version not in locked.get(name, set())
    }
    forbidden = sorted(installed.keys() & {"torch", "transformers", "faiss-gpu-cu12", "datasets", "sentence-transformers", "peft", "deepspeed"})
    if forbidden:
        raise RuntimeError(f"GPU/training packages entered portable proof: {forbidden}")
    if missing or drift:
        raise RuntimeError(f"Portable environment differs from uv.lock: missing={missing}, drift={drift}")
    direct_sources = {}
    explicit = explicit_portable_wheels(root)
    receipt_path = os.environ.get("EMBEDDING_PORTABLE_ARTIFACTS")
    artifacts = json.loads(Path(receipt_path).read_text(encoding="utf-8")) if receipt_path else {}
    for name, expected in explicit.items():
        artifact = artifacts.get(name, {})
        if any(artifact.get(key) != value for key, value in expected.items()) or "path" not in artifact:
            raise RuntimeError(f"Missing root-bound artifact receipt: {name}")
        path = Path(artifact["path"])
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["size"] or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise RuntimeError(f"Portable artifact changed after verification: {name}")
        raw = metadata.distribution(name).read_text("direct_url.json")
        direct = json.loads(raw) if raw else {}
        if direct.get("url") != path.resolve().as_uri():
            raise RuntimeError(f"Installed explicit source differs from verified artifact: {name}")
        direct_sources[name] = artifact
    return {
        "direct_sources": direct_sources,
        "python": sys.version,
        "executable": sys.executable,
        "lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((root / "pyproject.toml").read_bytes()).hexdigest(),
        "installed": dict(sorted(installed.items())),
    }


def run_suite(evidence: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required; use the repository-pinned uv 0.13.1")
    actual_uv = subprocess.check_output([uv, "--version"], text=True).strip()
    if not actual_uv.startswith("uv 0.13.1 "):
        raise RuntimeError(f"Expected uv 0.13.1, received {actual_uv}")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Portable consumer qualification requires Python 3.12")

    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", MPLBACKEND="Agg",
        MPLCONFIGDIR=str(evidence / "matplotlib"),
    )
    env.setdefault("UV_CACHE_DIR", str(evidence / "uv-cache"))
    env.setdefault("TMPDIR", str(evidence / "tmp"))
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    constraints = evidence / "root-constraints.txt"
    requirements = evidence / "portable-requirements.txt"
    commands: list[list[str]] = []

    def run(command: list[str]) -> None:
        commands.append(command)
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    run([uv, "lock", "--check", "--python", sys.executable])
    run([
        uv, "export", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes",
        "--format", "requirements.txt", "--output-file", str(constraints),
    ])
    artifacts = prepare_explicit_artifacts(evidence, ROOT)
    requirements.write_text(portable_requirements(constraints, ROOT, artifacts), encoding="utf-8")
    env["EMBEDDING_PORTABLE_ARTIFACTS"] = str(evidence / "explicit-artifacts.json")
    index = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["uv"]["index-url"]
    prefix = [
        uv, "run", "--isolated", "--no-project", "--no-config", "--no-env-file",
        "--no-python-downloads", "--python", sys.executable,
        "--default-index", index, "--with-requirements", str(requirements), "python",
    ]
    run(prefix + [str(Path(__file__).resolve()), "--verify-environment", str(evidence / "runtime.json")])
    run(prefix + ["-m", "unittest", "discover", "-s", "tests", "-v"])
    (evidence / "commands.json").write_text(json.dumps({"uv": actual_uv, "commands": commands}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, help="New directory retaining lock exports and runtime receipt")
    parser.add_argument("--verify-environment", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.verify_environment:
        receipt = environment_receipt()
        args.verify_environment.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, sort_keys=True))
        return
    if args.evidence_dir:
        args.evidence_dir.mkdir(parents=True, exist_ok=False)
        run_suite(args.evidence_dir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="embedding-portable-") as temporary:
            run_suite(Path(temporary))


if __name__ == "__main__":
    main()

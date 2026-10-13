"""Download immutable Hugging Face releases and validate their integrity."""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from urllib.parse import quote
from urllib.request import urlopen


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line, value in enumerate(stream, 1):
            if not value.strip():
                raise ValueError(f"{path}:{line}: empty JSONL row")
            row = json.loads(value)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line}: expected an object")
            yield row


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_manifest(path):
    value = read_json(path)
    require(isinstance(value, dict), "Manifest must be an object")
    require(value.get("schema_version") == 1, "Unsupported manifest schema")
    require(value.get("kind") in ("model", "training", "benchmark"), "Unknown release kind")
    require(re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", value.get("repo_id", "")), "Invalid repo ID")
    require(re.fullmatch(r"[0-9a-f]{40}", value.get("revision", "")), "Revision must be an immutable commit SHA")
    require(isinstance(value.get("files"), dict) and value["files"], "Missing files")
    for name, spec in value["files"].items():
        require(isinstance(spec, dict), "File specification must be an object")
        parts = PurePosixPath(name).parts
        require(name and not name.startswith("/") and "\\" not in name and
                all(p not in (".", "..") for p in name.split("/")) and
                str(PurePosixPath(name)) == name, "Unsafe artifact path")
        require(parts and re.fullmatch(r"[0-9a-f]{64}", spec.get("sha256", "")), "Invalid SHA256")
        require(type(spec.get("bytes")) is int and spec["bytes"] >= 0, "Invalid file size")
    if value["kind"] != "model":
        expected = {"queries.jsonl", "corpus.jsonl"}
        expected |= ({"relations.jsonl", "sampling.jsonl", "retention.jsonl"}
                     if value["kind"] == "training" else {"qrels.jsonl", "model-rankings.jsonl"})
        counts = value.get("row_counts")
        require(isinstance(counts, dict) and set(counts) == expected, "Invalid dataset tables")
        require(expected <= value["files"].keys(), "Dataset tables missing from checksums")
        require(all(type(n) is int and n > 0 for n in counts.values()), "Invalid row counts")
        field = "retention_query_count" if value["kind"] == "training" else "source_group_count"
        require(type(value.get(field)) is int and value[field] > 0, f"Invalid {field}")
        if value["kind"] == "benchmark":
            require("model-results.json" in value["files"], "Missing reference metrics checksum")
            from .metrics import PROTOCOL
            require(value.get("protocol") == PROTOCOL, "Unsupported benchmark protocol")
    return value


def artifact_path(root, name):
    root = Path(root).resolve()
    target = root / name
    # Refuse symlinked artifacts or parents rather than writing outside the cache.
    for part in (target, *target.parents):
        require(not part.is_symlink(), f"Symlink not allowed: {part}")
        if part == root:
            break
    require(target.resolve().is_relative_to(root.resolve()), "Artifact escapes output directory")
    return target


def verify_file(path, spec):
    require(path.is_file(), f"Missing artifact: {path}")
    require(path.stat().st_size == spec["bytes"], f"Size mismatch: {path}")
    require(digest(path) == spec["sha256"], f"SHA256 mismatch: {path}")


def verify_files(manifest, root):
    for name, spec in manifest["files"].items():
        verify_file(artifact_path(root, name), spec)


def download(manifest, root, opener=urlopen):
    prefix = "" if manifest["kind"] == "model" else "datasets/"
    base = f"https://huggingface.co/{prefix}{manifest['repo_id']}/resolve/{manifest['revision']}/"
    downloaded = 0
    for name, spec in manifest["files"].items():
        path = artifact_path(root, name)
        if path.exists():
            verify_file(path, spec)  # Corrupt local files require explicit user disposition.
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".lca-download-", dir=path.parent)
        temp = Path(temporary)
        try:
            with os.fdopen(fd, "wb") as output, opener(base + quote(name, safe="/"), timeout=60) as response:
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    require(total <= spec["bytes"], f"Oversized response: {name}")
                    output.write(block)
            verify_file(temp, spec)
            # Atomic no-clobber publication; concurrent invocations cannot overwrite results.
            try:
                os.link(temp, path)
            except FileExistsError:
                verify_file(path, spec)
            downloaded += 1
        finally:
            temp.unlink(missing_ok=True)
    verify_files(manifest, root)
    return downloaded


def unique(items, key):
    result = {}
    for row in items:
        value = row.get(key)
        require(isinstance(value, str) and bool(value.strip()) and value not in result, f"Invalid/duplicate {key}")
        result[value] = row
    return result


def validate_dataset(manifest, root):
    """Validate all released tables and their ID joins."""
    if manifest["kind"] == "model":
        return {}
    root = Path(root)
    tables = {name: list(rows(root / name)) for name in manifest["row_counts"]}
    counts = {name: len(table) for name, table in tables.items()}
    require(counts == manifest["row_counts"], "Dataset row counts differ from release")
    queries = unique(tables["queries.jsonl"], "query_id")
    docs = unique(tables["corpus.jsonl"], "document_id")
    for row in [*queries.values(), *docs.values()]:
        require(isinstance(row.get("text"), str) and row["text"].strip(), "Empty text")
        if "text_sha256" in row:
            require(hashlib.sha256(row["text"].encode()).hexdigest() == row["text_sha256"], "Text hash mismatch")
    if manifest["kind"] == "training":
        for row in queries.values():
            require(row["anchor_document_id"] in docs, "Unknown anchor document")
        labels = {0: "negative", 1: "hard_negative", 2: "acceptable", 3: "direct", 4: "direct"}
        seen = set()
        for row in tables["relations.jsonl"]:
            q, d, grade = row["query_id"], row["document_id"], row["grade"]
            require(q in queries and d in docs and (q, d) not in seen, "Invalid relation join/duplicate")
            require(type(grade) is int and labels.get(grade) == row["label"], "Grade/label mismatch")
            require(row["role"] in ("primary", "additional_positive"), "Invalid relation role")
            require(all(v in docs for v in row.get("better_document_ids", [])), "Invalid preference document")
            seen.add((q, d))
        draws = unique(tables["sampling.jsonl"], "draw_id")
        require(all(r["query_id"] in queries for r in draws.values()), "Unknown sampled query")
        require([r["position"] for r in draws.values()] == list(range(len(draws))), "Invalid sampling positions")
        retention = tables["retention.jsonl"]
        require(len({r["query_id"] for r in retention}) == manifest["retention_query_count"], "Retention query count mismatch")
        seen = set()
        for row in retention:
            pair = row["query_id"], row["document_id"]
            score = row["teacher_cosine_score"]
            require(pair not in seen and pair[0] in queries and pair[1] in docs,
                    "Invalid retention join/duplicate")
            require(type(score) in (int, float) and math.isfinite(score) and -1 <= score <= 1, "Invalid teacher score")
            seen.add(pair)
    else:
        from .metrics import validate_references, validate_rankings
        validate_references(queries, docs, tables["qrels.jsonl"])
        validate_rankings(queries, docs, tables["model-rankings.jsonl"])
        require(len({r["source_group"] for r in queries.values()}) == manifest["source_group_count"], "Source group count mismatch")
    return counts

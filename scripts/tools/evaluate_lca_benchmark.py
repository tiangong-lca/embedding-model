"""Score saved Top10 rankings against the pinned LCA Retrieval Benchmark (no ML dependencies)."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.lca_benchmark.metrics import METRICS, evaluate
from src.lca_benchmark.releases import digest, read_json, read_manifest, require, rows, validate_dataset, verify_files

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "config/datasets/lca-retrieval-benchmark-v1.json")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--verify-model-results", action="store_true")
    args = parser.parse_args()
    try:
        manifest = read_manifest(args.manifest)
        require(manifest["kind"] == "benchmark", "Expected benchmark manifest")
        verify_files(manifest, args.data_dir)
        validate_dataset(manifest, args.data_dir)
        result = evaluate(args.data_dir, list(rows(args.rankings)))
        result.update(benchmark_revision=manifest["revision"], rankings_sha256=digest(args.rankings))
        if args.verify_model_results:
            expected = read_json(args.data_dir / "model-results.json")["metrics"]
            require(set(expected) == set(METRICS), "Incomplete reference metrics")
            require(all(math.isfinite(expected[k]) and abs(result["metrics"][k] - expected[k]) <= 1e-12 for k in METRICS), "Published metrics mismatch")
            result["published_model_metrics_verified"] = True
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

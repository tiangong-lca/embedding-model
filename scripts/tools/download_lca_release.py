"""Download a pinned LCA release or verify an existing local copy; JSON result on stdout."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.lca_benchmark.releases import download, read_manifest, validate_dataset, verify_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Explicit cache directory; use data/releases/... to keep artifacts outside Git")
    parser.add_argument("--verify-only", action="store_true", help="Offline: require every pinned file to exist and match")
    args = parser.parse_args()
    try:
        manifest = read_manifest(args.manifest)
        count = 0 if args.verify_only else download(manifest, args.output_dir)
        verify_files(manifest, args.output_dir)
        counts = validate_dataset(manifest, args.output_dir)
        print(json.dumps({"verified": True, "repo_id": manifest["repo_id"], "revision": manifest["revision"],
                          "downloaded_files": count, "verified_files": len(manifest["files"]), "row_counts": counts}, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

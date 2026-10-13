"""Optional offline pure-embedding Top10 retrieval in the isolated Nemotron runtime."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.lca_benchmark.metrics import rank_scores
from src.lca_benchmark.releases import digest, read_manifest, require, rows, validate_dataset, verify_files

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not already exist")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    try:
        require(args.batch_size > 0, "batch-size must be positive")
        require(not args.output_dir.exists(), "Output directory already exists; choose a new path")
        model_release = read_manifest(ROOT / "config/releases/lca-nemotron-embed-1b.json")
        data_release = read_manifest(ROOT / "config/datasets/lca-retrieval-benchmark-v1.json")
        verify_files(model_release, args.model_dir)
        verify_files(data_release, args.data_dir)
        validate_dataset(data_release, args.data_dir)
        # Import only after offline integrity checks; score-only tools never import ML libraries.
        import torch
        import torch.nn.functional as F
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(args.model_dir), device=args.device, local_files_only=True,
                                    trust_remote_code=False, model_kwargs={"dtype": torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
                                                                        "attn_implementation": "sdpa"})
        require(model.prompts == {"query": "query: ", "document": "passage: "}, "Unexpected prompts")
        require(model.get_sentence_embedding_dimension() == 2048 and model[0].auto_model.config.is_causal is False,
                "Unexpected embedding model contract")
        require(model[1].pooling_mode_mean_tokens and model[1].include_prompt, "Unexpected pooling")
        docs = sorted(rows(args.data_dir / "corpus.jsonl"), key=lambda r: r["document_id"])
        queries = sorted(rows(args.data_dir / "queries.jsonl"), key=lambda r: r["query_id"])
        def encode(records, kind, limit):
            model.max_seq_length = limit
            fn = model.encode_query if kind == "query" else model.encode_document
            with torch.inference_mode():
                vectors = fn([r["text"] for r in records], batch_size=args.batch_size, convert_to_tensor=True,
                             normalize_embeddings=False, show_progress_bar=False).float()
                require(bool(torch.isfinite(vectors).all()), "Non-finite embedding")
                return F.normalize(vectors, p=2, dim=1).cpu()
        dvec = encode(docs, "document", 2048)
        qvec = encode(queries, "query", 512)
        doc_ids = [d["document_id"] for d in docs]
        rankings = [{"query_id": q["query_id"], "document_ids": rank_scores(doc_ids, (qvec[i] @ dvec.T).tolist())}
                    for i, q in enumerate(queries)]
        args.output_dir.mkdir(parents=True, exist_ok=False)
        output = args.output_dir / "rankings.jsonl"
        output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rankings), encoding="utf-8")
        receipt = {"model_revision": model_release["revision"], "benchmark_revision": data_release["revision"],
                   "rankings_sha256": digest(output), "query_tokens": 512, "document_tokens": 2048,
                   "device": args.device, "batch_size": args.batch_size, "similarity": "CPU float32 inner product",
                   "runtime": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "sentence-transformers")},
                   "queries": len(queries), "documents": len(docs)}
        (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

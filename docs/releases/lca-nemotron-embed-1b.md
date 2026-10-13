# LCA-Nemotron-Embed-1B

2048-dimensional embeddings for life-cycle inventory process retrieval, with
attention-mask mean pooling and L2 normalization. The full checkpoint is available
from [Tiangong-LCA on Hugging Face](https://huggingface.co/Tiangong-LCA/LCA-Nemotron-Embed-1B).

## Download and verify

Run from the repository root with Python 3.12. These commands need no ML packages:

```bash
python scripts/tools/download_lca_release.py --manifest config/releases/lca-nemotron-embed-1b.json --output-dir data/releases/model
python scripts/tools/download_lca_release.py --manifest config/datasets/lca-embedding-training-v1.json --output-dir data/releases/training
python scripts/tools/download_lca_release.py --manifest config/datasets/lca-retrieval-benchmark-v1.json --output-dir data/releases/benchmark
```

The model download is approximately 4.58 GB; training and benchmark downloads are
approximately 27 MB and 38 MB. Manifests pin immutable HF commits and every payload's
SHA-256 and byte count, including the upstream checksum manifest itself. Each file
is verified before atomic publication. Valid existing files are reused; corrupt
existing files cause an error and are not overwritten. Use a new output directory
to retry a damaged cache. Interrupted temporary downloads are removed.

Add `--verify-only` to check the local cache without network access. Dataset checks
also cover row counts, unique IDs, document/query joins, text hashes, relation grades,
sampling draws and reference-weight mass. Success is JSON on stdout; validation
errors are JSON on stderr with exit code 2. All heavy files stay under ignored `data/`.

## Dataset structure

| Resource | Contents | Manifest |
|---|---|---|
| [LCA Embedding Training](https://huggingface.co/datasets/Tiangong-LCA/LCA-Embedding-Training) | Final-stage snapshot: 1,256 queries, 12,605 graded relations, 6,228 documents, 2,048 sampling draws and 11,848 retention scores | `config/datasets/lca-embedding-training-v1.json` |
| [LCA Retrieval Benchmark](https://huggingface.co/datasets/Tiangong-LCA/LCA-Retrieval-Benchmark) | 100 queries / 42 source groups, 15,671 documents, 14,182 reference relations and published Top10 rankings | `config/datasets/lca-retrieval-benchmark-v1.json` |

JSONL tables join on `query_id` and `document_id`. The dataset cards define fields,
grades and source attribution. The training snapshot covers the final training stage,
not a from-base training recipe. Retention scores are cosine supervision, not relevance
labels. Training and benchmark corpora have independent text hashes.

## Reproduce published metrics

```bash
python scripts/tools/evaluate_lca_benchmark.py --data-dir data/releases/benchmark --rankings data/releases/benchmark/model-rankings.jsonl --verify-model-results
```

| nDCG@5 | nDCG@10 | Coverage@5 | Coverage@10 |
|---:|---:|---:|---:|
| 0.3727708130 | 0.4403736443 | 0.336890 | 0.458947 |

For your own results, provide one JSONL row per query:
`{"query_id":"...","document_ids":["...ten distinct document IDs in ranked order..."]}`.
Replace `--rankings` and omit `--verify-model-results`. Missing/duplicate/unknown IDs
are rejected; results include per-query metrics, unjudged counts and input hashes.

The frozen protocol is `lca-fixed-five-reference-v1`. With relevance weight `p` and
rank starting at 1, Coverage@K is `sum(p)/5`, and nDCG@K is
`sum(p/log2(rank+1))/sum(1/log2(i+1), i=1..5)` for K=5,10. It does not use
exponential gain or corpus-wide graded IDCG. Unjudged documents score zero but are
counted separately from judged-zero documents. These are weighted retrieval metrics,
not classification accuracy. This exploratory, model-assisted benchmark is not a
blind holdout; uncertainty analysis should resample source groups.

## Optional model inference

The existing project environment serves the Qwen pipeline. Nemotron uses a separate
Python 3.12 / Linux x86_64 environment with the published reproduction versions:
Torch 2.11.0 CUDA 12.9, Transformers 5.4.0 and Sentence Transformers 5.3.0.

```bash
uv sync --project environments/nemotron-inference --frozen
environments/nemotron-inference/.venv/bin/python data/releases/model/encode.py --model data/releases/model --device cuda:0 --input-type query "中国天然气联合循环发电清单"
environments/nemotron-inference/.venv/bin/python data/releases/model/encode.py --model data/releases/model --device cuda:0 --input-type document "Electricity production; combined cycle; natural gas; China; kWh"
```

Supply unprefixed text: saved prompts add `query: ` and `passage: `. Query/document
limits are 512/2048 tokens including prefixes and special tokens; truncation is enabled.
Document text should retain process name, geography, technology, product/unit,
system boundary and time. The [model card](https://huggingface.co/Tiangong-LCA/LCA-Nemotron-Embed-1B)
includes Python loading and HTTP API examples.

To regenerate full-corpus benchmark rankings locally (GPU inference, not scoring a
saved file):

```bash
environments/nemotron-inference/.venv/bin/python scripts/tools/retrieve_lca_benchmark.py --model-dir data/releases/model --data-dir data/releases/benchmark --output-dir data/releases/retrieval-run --device cuda:0 --batch-size 8
python scripts/tools/evaluate_lca_benchmark.py --data-dir data/releases/benchmark --rankings data/releases/retrieval-run/rankings.jsonl
```

The runner verifies both releases, loads only local model files without remote code,
encodes all documents, sorts cosine scores with document-ID tie breaking, and writes
rankings plus a runtime/revision receipt to a new directory. It uses no reranking.
Precision and hardware can affect close rankings. The GPU run is an explicit optional
validation step, separate from the standard-library tests and reference-score parity.

## Maintainer validation

```bash
python3.12 -m unittest discover -s tests -p 'test_lca_*.py' -v
uv lock --check --project environments/nemotron-inference --python 3.12
```

Release updates require new verified commit/hash manifests and metric parity checks;
do not move a pinned benchmark to mutable `main`. Keep model/dataset binaries on HF,
not in Git. Model license and attribution are included with the HF payload; repository
tooling follows the repository license.

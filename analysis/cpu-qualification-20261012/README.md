# Production-configured embedding artifact: warm 4 CPU qualification

Issue [embedding-model #42](https://github.com/tiangong-lca/embedding-model/issues/42), PR [#43](https://github.com/tiangong-lca/embedding-model/pull/43), workspace coordination [#1835](https://github.com/tiangong-lca/workspace/issues/1835). Test date: 2026-10-12. The production inference endpoint was not invoked; SageMaker configuration was unchanged.

The exact configured production artifact completed a warm FP32 CPU test with one forward at a time and measured queue wait. On this host, 32-token single-input P95 is 142.53 ms; 1024-token single-input P95 is 1860.80 ms. Concurrent requests increase queue latency rather than sustained long-input throughput. This is a feasible resident-worker configuration for the measured hardware, with a substantial long-input latency tradeoff.

## Artifact, runtime and limits

Fresh read-only SageMaker control-plane inspection at approximately 04:39 UTC still configured the selected S3 artifact, model `huggingface-pytorch-inference-2026-01-29-18-53-22-119`, endpoint InService with one `ml.g4dn.xlarge`. This binds the configured artifact; it does not attest tensors currently in GPU memory. Archive SHA256: `df73c8671f74e238bdc0a41c06d7bb33a03744e795becfb507f620a908571cf2`. Weight SHA256: `a13692bcf8f32ca5d931132bfabdf82846289fac89a3257f343d23c6908c60c1`. All 11 regular archive members were individually verified and extracted without paths outside the owned directory. [File identities](artifact-file-hashes.json).

Native Linux x86_64 host `DELLw3425-new-local-148`, Intel Xeon w5-3425. Docker quota 4 CPUs, affinity 0–3 on four distinct physical cores, memory limit 8 GiB, swap 0, no network or GPU, non-root UID1000, read-only root/model/data/source mounts, private result bind. These four physical-core threads are not a demonstrated equivalent to every cloud 4-vCPU shape.

Core libraries reproduce the deployed line: Torch 2.6.0+cpu and Transformers 4.51.3; Python 3.12.15. NumPy 2.3.5 matches the existing evaluation-host baseline for comparability. No latest-version claim or training-lock upgrade is made. Resolved base digest `sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1`; exact built image `sha256:302eecfba037c7f7ada8949232eaa28f96c62716240013b222f5733613c4a5ed`; [dependency freeze](runtime-freeze.txt).

Production semantics: no query prefix, right padding, truncation at 1024 tokens, last valid token pooling, L2 normalization, 1024-dimensional FP32 output. Five direct comparisons with the reviewed archive's actual `predict_fn` (32/128/512/1024-token singles and a mixed 32/1024-token batch) were bit-exact on CPU. This verifies the reimplementation against the same-artifact handler; it is not a live GPU-vector comparison.

## Warm serving measurements

One model is loaded once per run and remains resident across the matrix. All request measurements occur after warmup. Startup is excluded; the recorded model-load field covers AutoModel loading, rather than the entire process/container startup. The candidate requires an always-running worker and readiness after model loading/warmup. Scaling to zero or loading weights per request does not meet the requested no-cold-start condition.

Serialized mode permits one forward at a time through a lock. Timers start before lock acquisition, so queue wait is included. The lock is not specified as FIFO. Tokenization, forward, pooling, normalization and JSON serialization are measured; HTTP/network and production search/index latency are outside the measurement. Closed-loop clients send one input per request. Each of the 12 cells completed 30/30 requests: **360 requests, zero errors**. Cell wall limit 300 seconds; phase and per-request checkpoints are retained. P95/P99 are descriptions of these small samples, not established production tail SLOs.

| Tokens | Concurrent clients | P50 ms | P95 ms | P99 ms | Requests/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 140.36 | 142.53 | 142.84 | 7.107 |
| 32 | 3 | 421.02 | 422.59 | 423.88 | 7.119 |
| 32 | 6 | 847.13 | 852.85 | 853.55 | 7.074 |
| 128 | 1 | 262.49 | 267.41 | 273.66 | 3.818 |
| 128 | 3 | 784.36 | 788.28 | 792.55 | 3.827 |
| 128 | 6 | 1570.57 | 1579.78 | 1581.15 | 3.817 |
| 512 | 1 | 912.11 | 934.99 | 952.97 | 1.100 |
| 512 | 3 | 2732.82 | 2758.10 | 2765.26 | 1.099 |
| 512 | 6 | 5431.37 | 5489.11 | 5501.25 | 1.102 |
| 1024 | 1 | 1807.97 | 1860.80 | 1875.56 | 0.554 |
| 1024 | 3 | 5388.51 | 5456.78 | 5475.14 | 0.557 |
| 1024 | 6 | 10718.64 | 10819.09 | 10850.37 | 0.558 |

[Complete serialized samples and source manifest](serialized-benchmark.json). Frozen measured harness: `f56b587efbf15c4d78cac27cc23eb6fc0137bef5`, SHA256 `dc86a0cecd04d136c8db52041af3d1ef726834129d907d76f661b11ad2068cbd`.

The earlier direct parallel-forward configuration completed seven cells / 840 requests, then failed to finish the 512-token concurrency-3 cell for over 10 minutes. It was stopped at the main agent's bound; OOM was false. Its phase and in-cell completed count are unknown because the original harness only checkpointed completed cells, including a concurrent warmup before timed calls. No missing-cell quantiles are invented. 22 process threads and nearly 400% CPU were observed; oversubscription is plausible but unproven, with little cgroup throttling. [Completed parallel samples](parallel-benchmark-partial.json), [interruption receipt](parallel-interruption.json). Its source identity is reconstructed from the command/commit timeline (`fc3216d`), not a script hash captured at launch. These results describe that configuration, not all possible 4 CPU services.

## Retrieval quality

All 1,893 queries and 3,786 documents were regenerated from the exact production-configured artifact. Offline batch size 4, descending token-length buckets, original ID order restored. Query encoding took 75.54 s; corpus encoding took 2878.23 s. Exact cosine retrieval over the full corpus returns top 100. Model/data/runtime identities and hashes of all four NPY/ID outputs are bound in [the full quality result](quality.json); output hashes were verified after completion. Frozen quality harness: `dc4aea5a2cd4f2e90fcd37b344fa3f4cfc04044c`, SHA256 `20df892dbdcc39cd9ff1dd684d48a073bc802cfc192661b7befacc2bc4c35c28`.

| Evaluation | Queries | Recall@10 | NDCG@10 | MRR@10 | Recall@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical full dataset | 1893 | 0.904385 | 0.764047 | 0.718826 | 0.994717 |
| Deduplicated, query train-disjoint subset | 1795 | 0.915877 | 0.783042 | 0.740045 | 0.994429 |

89 unique evaluation query strings occur in training, and **all 3,786 corpus documents occur as training positives**. The 1,795 subset excludes duplicate query strings and strings seen in training; it is not independent unseen-document generalization. Labels identify one positive per query and do not exhaustively judge other plausible documents. This is historical retrieval/regression evidence, with no production multilingual or downstream hybrid-search qualification.

1,825 queries (96.4%) have <=32 tokens. Corpus has 19 documents of 33–128 tokens, 2,336 of 129–512, 1,311 of 513–1024 and 120 above 1024. Those 120 (3.17%) are truncated under the existing production rule. [Exact token inventory](token-inventory.json). The historical published fine-tuned Recall@10 of 0.9049 is context only: its old GPU cache lacks exact deployed binary provenance and is not used as a production golden or proof of live GPU parity.

## Memory, validation and resource disposition

Serialized maximum `ru_maxrss` is 5212.64 MiB. This is the process lifetime high water, including initialization, parity and preceding cells. Serialized cgroup peak is 3.336 GiB; quality cgroup peak 2.245 GiB. Cgroup accounting and process RSS differ; no isolated steady RSS measurement is claimed. No observed OOM events.

17/17 unit tests passed with exact TIDAS 0.2.14 and NumPy 2.3.5, source compilation and AI-doc lint passed, training lock check passed. Independent review resolved dataset/output hash and deadline races; no material scoped findings remained. Final code changes after measured freezes only summarize quality stdout; measured source identities above remain explicit.

The shared development-resource controller reviewed and removed exactly three task-created containers. All three removals are verified in [the completed effect receipt](cleanup-effect-receipt.json), bound to [the exact plan](cleanup-plan.json). [Post-release inventory](cleanup-inventory-after.json) has no task containers; GPUStack, its data volume, all other existing volumes/networks and image IDs are unchanged. No task-created named volumes/networks remain.

Retention owner is [embedding #42](https://github.com/tiangong-lca/embedding-model/issues/42), with resource adoption/labels `embedding-model#42`. The archive, frozen sources, raw vectors/results, runtime image and build cache are retained to reproduce PR #43 and required root integration. Owner review is due 2026-10-19: after durable evidence and integration are verified, remove unnecessary private copies through an explicit resource review; if a follow-up still needs them, record the reason and next review date in the Issue. This is a bounded retention disposition, with no automatic deletion. Original study data belongs to its existing owner and is preserved. Private retained directory: `/home/david/.cache/embedding-model-42-20261012`; local evidence mirror `/tmp/embedding-cpu-20261012/results`.

## Decision boundary

Low-concurrency short queries are a plausible CPU candidate on the measured host. Bursts of 512/1024-token requests produce substantial queue latency, and the naive concurrent-forward configuration was not qualified. Meeting the no-cold-start requirement needs resident models, readiness only after prewarming, a prewarmed spare during updates/restarts and controlled traffic switching. A single resident instance alone does not cover failure or restart gaps. Keep the existing SageMaker service while separately qualifying the target cloud 4-vCPU hardware, HTTP load, failover and rollout behavior. This experiment does not authorize a production migration.

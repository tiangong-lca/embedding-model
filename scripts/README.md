# Scripts Layout

## pipeline/

端到端链路（数据 -> 训练 -> 缓存 -> 评测）：

- 01_generate_markdown.py
- 02_generate_queries.py / 02_generate_single.py
- 03_prepare_ft_data.py
- 04_finetune_st.py / 04_finetune_st.sh
- 05_cache_embeddings.py
- 06_evaluate_cached.py
- 07_evaluate_st.py

`01_generate_markdown.py` resolves its default `data/` directories from the repository root and delegates process/flow rendering to TIDAS SDK 0.2.14's canonical Markdown surface. Use `python scripts/pipeline/01_generate_markdown.py --help` for the exact current arguments.

## reports/

报告生成：

- generate_report_html.py
- generate_cached_eval_report.py
- generate_eval_results_report.py

## tools/

辅助工具：

- `download_lca_release.py`：按不可变版本下载模型/数据，校验大小、SHA-256、行数和关联；`--verify-only` 离线核验。
- `evaluate_lca_benchmark.py`：仅用标准库对 Top10 JSONL 排名评分，输出指标、未判定数量和输入身份。
- `retrieve_lca_benchmark.py`：在独立 Nemotron 环境中生成全语料纯 embedding 排名；显式运行才占用 GPU。
- 用法及数据口径见 [发布指南](../docs/releases/lca-nemotron-embed-1b.md)。

- apply_hard_negatives.py, mine_hard_negatives.py
- export_embedding_atlas_dataset.py（导出 Embedding Atlas 可视化数据集）
- export_embedding_atlas_markdown_dir.py（将 Supabase 导出的 markdown 目录打包为 Embedding Atlas 数据集）
- cache_markdown_embeddings.py（为 Supabase markdown 预计算并缓存 embedding 向量）
- evaluate_similarity_gaps.py, compare_pair_similarity.py
- check_bf16_env.py, download_model.py, export_supabase_json_ordered.py
- upload_hf_model.py, upload_ollama_model.sh, convert_to_gguf.sh

## legacy/

旧入口（评估/训练），逐步淘汰：

- evaluate.py
- finetune.sh

根目录保留 scripts_overview.md 供概览；后续可在 docs/PIPELINE.md 补充端到端命令示例。

## Offline CPU qualification

`tools/qualify_cpu_embedding.py` measures the exact locally supplied artifact, using one resident model per run. `benchmark` warms each token length and measures closed-loop single-input concurrency; `--execution-mode serialized` permits one forward at a time and includes lock queue wait in latency. Warmup follows the selected policy. Every completed request is checkpointed and `--cell-seconds` bounds each concurrency cell, including its worker warmup, by exiting only the benchmark process on expiry; `quality` regenerates embeddings and retrieval metrics; `compare` requires binary-bound regenerated FP32 reference vectors and validates hashes of every vector/ID file. Each mode requires a fresh output directory; preserve interrupted runs instead of overwriting them. Freeze the harness source in an immutable per-revision mount before starting a run. It refuses visible GPUs and uses offline local-only loading.

Build `tools/cpu-qualification.Dockerfile` in an owned test directory, then run with explicit Docker CPU quota, distinct physical-core affinity, RAM and swap limits, no network/GPU, read-only model/data/script mounts, and an owned writable result mount. Example container arguments:

```text
--cpus=4 --cpuset-cpus=0-3 --memory=8g --memory-swap=8g --network=none
--read-only --tmpfs /tmp:rw,noexec,nosuid,size=512m --cap-drop=ALL
--security-opt=no-new-privileges --pids-limit=256
```

Resolve affinity from the actual host topology. Capture image ID, dependency freeze, artifact/data hashes, Docker inspect, cgroup readings, raw samples, and cleanup or retention receipts. Torch 2.6.0 and Transformers 4.51.3 reproduce the existing deployed core libraries; NumPy 2.3.5 matches the existing evaluation host baseline for comparability. These are reproduction pins, not a claim to use latest packages, and do not replace the training lock. Record the resolved base image digest and complete dependency freeze for each experiment.

```bash
python tools/qualify_cpu_embedding.py benchmark --model /model --output /results/benchmark --requests 200
python tools/qualify_cpu_embedding.py quality --model /model --data /data --output /results/fp32 --batch-size 4
python tools/qualify_cpu_embedding.py compare --data /data --candidate /results/fp32 --reference /results/reference --output /results/comparison
```

Run `--help` for all flags. The executable script path in the repository is `scripts/tools/qualify_cpu_embedding.py`; `/test/qualify_cpu_embedding.py` is the Docker entrypoint mount. Startup is recorded separately and excluded from warm latency. Percentiles are sample descriptions, not a guaranteed production tail SLO. The benchmark includes tokenization, model forward, pooling, normalization and JSON serialization, but no HTTP/network or production search latency. Regenerated CPU FP32 provides a numerical reference, not measured parity with a live GPU endpoint. Historical query/corpus/qrels metrics must be labeled with training-overlap and incomplete-judgment limitations; report the deduplicated train-disjoint query subset when training data is supplied.

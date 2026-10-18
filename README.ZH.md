# LCA 检索领域向量（Embedding）微调与评测

本项目聚焦一个问题：在生命周期评价（LCA）这类专业语料的检索任务中，领域化的向量微调能带来多大收益？

## 已发布模型与数据集

| 资源 | 用途 |
|---|---|
| [LCA-Nemotron-Embed-1B](https://huggingface.co/Tiangong-LCA/LCA-Nemotron-Embed-1B) | 2048 维 LCI 过程检索向量 |
| [LCA Embedding Training](https://huggingface.co/datasets/Tiangong-LCA/LCA-Embedding-Training) | 最终阶段分级监督、语料、采样与保持监督 |
| [LCA Retrieval Benchmark](https://huggingface.co/datasets/Tiangong-LCA/LCA-Retrieval-Benchmark) | 全语料检索任务、参考权重与可复算指标 |
| [LCA-Qwen3-0.6B](https://huggingface.co/BIaoo/lca-qwen3-embedding) | 现有 Qwen 系列模型 |

见 [Nemotron 下载、校验与评测指南](docs/releases/lca-nemotron-embed-1b.md)。
`config/releases/` 和 `config/datasets/` 固定发布版本与 SHA-256。
下载和已有排名评分仅依赖 Python 标准库；GPU 推理环境独立位于
`environments/nemotron-inference/`，不改变现有 Qwen 依赖。

LCA-Nemotron 在 LCA Retrieval Benchmark 上的 **nDCG@10 为 0.44037，Coverage@10 为 0.45895**。
下面的 Qwen 实验使用另一套数据及指标口径，其相对提升不能与上述指标直接比较。

## Qwen 实验与处理管线

当前 Python 3.12 / uv 根锁采用 TIDAS SDK 0.3.0、Matplotlib 3.11.2、OpenAI 3.35.1、python-dotenv 1.2.4 和 Supabase 2.33.0；Qwen/CUDA 12 训练依赖保持既有固定版本。

可移植验证：

```bash
uv lock --check --python 3.12
uv run --isolated --no-project --python 3.12 python scripts/tools/run_portable_tests.py
python3.12 -m compileall -q src scripts tests
```

验证器从根 `uv.lock` 导出临时冻结约束，只安装 TIDAS/辅助客户端及其锁定的传递依赖，记录实际安装版本，并执行完整 unittest。它不会同步 GPU 工程、调用真实供应商/数据库或下载模型/数据。TIDAS 升级的 127 项比较仅覆盖已记录的本地 Process/Flow 消费路径，不代表完整严格语料或 SDK 全部语义等价。python-dotenv 1.2.4 将 `KEY= # comment` 修正为空值。详细边界见 [依赖维护说明](ai/dependency-maintenance.md)。

## 背景与目标

通用向量模型在开放领域表现良好，但在 LCA 场景中往往会出现“语义相近但不满足领域约束”的错排；而 LCA 文档通常较长、字段多（地理/技术路径/时间等），对检索排序与覆盖提出更强的领域一致性要求。本项目通过统一评测对比，量化领域微调相对通用基线与云端模型的效果差异。

## 数据规模（本次实验）

- 数据源：TianGong LCA 数据；原始数据采用 Tidas 结构化格式（包含过程/流及元信息等字段），并转换为便于检索的文本形态用于构建评测集合。
- 训练集：17,037 条 query-doc
- 评测集：1,893 queries / 3,786 corpus / 1,893 qrels

## 对比设置（概览）

对比对象包括：

- `raw`：`Qwen3-Embedding-0.6B`（通用嵌入基线）
- `ft`：在 LCA 数据上微调后的 `Qwen3-Embedding-0.6B`
- 云端对照：`qwen3-embedding-8b` / `qwen3-embedding-4b`、`bge-m3`、`codestral-embed-2505`

## 指标与结果要点

- 指标：NDCG / MAP / Recall / Precision / MRR @ `{1,5,10,50,100}`（查询级平均）
- 结论：`ft` 相对 `raw` 在头部排序与长尾覆盖上均有显著提升

## 模型效果（摘要）

- 相对 `raw`，`ft` 显著提升：NDCG@10 +31.2%，Recall@10 +25.7%，MRR@10 +33.5%；Recall@100 +11.5%。

## 已发布模型与使用方式

### Hugging Face（SentenceTransformers）

- 模型地址：<https://huggingface.co/BIaoo/lca-qwen3-embedding>

```bash
pip install -U sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BIaoo/lca-qwen3-embedding")
emb = model.encode(["wood residue gasification heat recovery"], prompt_name="query", normalize_embeddings=True)
print(emb.shape)
```

### Ollama

- 模型地址：<https://ollama.com/BiaoLuo/lca-qwen3-embedding>

```bash
ollama pull BiaoLuo/lca-qwen3-embedding
```

通过本地 Ollama Server API 获取 embedding：

```bash
curl http://localhost:11434/api/embeddings \
  -d '{"model":"BiaoLuo/lca-qwen3-embedding","prompt":"wood residue gasification heat recovery"}'
```

## 可视化（Embedding Atlas）

Embedding Atlas（<https://apple.github.io/embedding-atlas）可以用来交互式查看> embedding 的结构、聚类、最近邻与元数据分布。结合本项目，推荐两种用法：

### 方案 A：直接从文本计算 embedding（最省事）

对评测集的 query+corpus 直接跑 Embedding Atlas，并指定你的微调模型作为 `--model`：

```bash
pip install -U embedding-atlas
embedding-atlas data/ft_data/test_queries.jsonl data/ft_data/corpus.jsonl \
  --text text \
  --model BIaoo/lca-qwen3-embedding \
  --trust-remote-code \
  --umap-metric cosine \
  --umap-random-state 42
```

提示：同时加载多个输入时，Embedding Atlas 会自动加一列 `FILE_NAME`，可用它筛选 query / corpus 两类点。

### 方案 B：复用本项目缓存的 embedding（对比 raw vs ft 更方便）

1) 先按评测管线缓存 embedding（见 `scripts/pipeline/05_cache_embeddings.py`）  
2) 导出为 Embedding Atlas 可读的数据集（包含 `vector_raw` / `vector_ft` 等列）：

```bash
.venv/bin/python scripts/tools/export_embedding_atlas_dataset.py \
  --data_dir data/ft_data \
  --cache_dir data/eval_cache \
  --model ft \
  --out data/output/embedding_atlas/lca_eval.parquet
```

然后分别选择要可视化的向量列启动 Embedding Atlas：

```bash
embedding-atlas data/output/embedding_atlas/lca_eval.parquet --vector vector_ft --text text --umap-metric cosine --umap-random-state 42
embedding-atlas data/output/embedding_atlas/lca_eval.parquet --vector vector_raw --text text --umap-metric cosine --umap-random-state 42
```

### 方案 C：对 Supabase 导出的 markdown 先预计算向量（大规模数据更省时）

适用于 `data/supabase_exports_markdown`（flows/processs）。先离线缓存 embedding，再导出带 `vector_*` 列的 parquet 给 Embedding Atlas 使用：

```bash
.venv/bin/python scripts/tools/cache_markdown_embeddings.py \
  --input_dir data/supabase_exports_markdown \
  --subset both \
  --text_mode plain \
  --model ft=local:data/output/lca-qwen3-st-finetuned \
  --output_dir data/embed_cache/supabase_markdown \
  --device cpu \
  --batch_size 64

.venv/bin/python scripts/tools/export_embedding_atlas_markdown_dir.py \
  --input_dir data/supabase_exports_markdown \
  --subset both \
  --text_mode plain \
  --vector_cache_dir data/embed_cache/supabase_markdown \
  --vector_model ft \
  --out data/output/embedding_atlas/supabase_markdown_ft.parquet

embedding-atlas data/output/embedding_atlas/supabase_markdown_ft.parquet --vector vector_ft --text text --umap-metric cosine --umap-random-state 42
```

## 结论

在本次 LCA 检索评测中，领域化向量微调相对通用基线在排序质量与覆盖度上都带来了明确收益，表明“通用向量 + 领域对齐”是专业检索场景的高性价比路径。

## 许可

- [MIT LICENSE](LICENSE)

## 链接与引用（占位）

- arXiv：TBA
- 引用格式：TBA
- Hugging Face：<https://huggingface.co/BIaoo/lca-qwen3-embedding>
- Ollama：<https://ollama.com/BiaoLuo/lca-qwen3-embedding>

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

---
title: domain-embedding AI Working Guide
docType: contract
scope: repo
status: active
authoritative: true
owner: domain-embedding
language: en
whenToUse:
  - when a task may change LCA embedding data preparation, fine-tuning, cached evaluation, reporting, or model packaging assets
  - when deciding whether work belongs in this repository, in tiangong-lca-mcp, or in a product/runtime repo
  - when routing from the workspace root into embedding-model
whenToUpdate:
  - when pipeline stages, runtime prerequisites, or ownership boundaries change
  - when canonical script entrypoints or dataset assumptions change
  - when the repo-local AI bootstrap docs under ai/ change
checkPaths:
  - AGENTS.md
  - README.md
  - README.ZH.md
  - TODO.md
  - ai/**/*.yaml
  - .githooks/**
  - pyproject.toml
  - uv.lock
  - src/**
  - scripts/**
  - tests/**
  - config/**
  - environments/**
  - docs/releases/**
  - data/**
  - analysis/**
  - report.md
  - report.ZH.md
lastReviewedAt: 2026-10-18
lastReviewedCommit: 0e0e612d626335c6c077b0778d7ab3a5f5c596de
lastReviewedNote: "Embedding #64 / original PR #50: reviewed immutable setup-node 7.1.0 for the existing manual-only read-only AI-doc gate with Node 24; triggers, inputs, shell enforcement and portable Python proof remain unchanged. Accepted security lock, TIDAS 0.2.14 and frozen Qwen/CUDA 12 boundaries are preserved; hosted Action qualification is separate."
related:
  - ai/repo.yaml
  - ai/doc-impact.yaml
  - README.md
  - scripts/README.md
---

# AGENTS.md — domain-embedding AI Working Guide

`lca-domain-embedding` owns the checked-in LCA retrieval embedding experiment pipeline: data preparation, fine-tuning, cached evaluation, reporting, and model packaging helpers. Start here when the task may change how this repository builds or evaluates embedding assets.

## AI Load Order

Load docs in this order:

1. `AGENTS.md`
2. `ai/repo.yaml`
3. `ai/doc-impact.yaml`
4. `README.md` for experiment scope and published-model context
5. `scripts/README.md` when the task touches pipeline entrypoints
6. the relevant pipeline or tool script under `scripts/**`

Do not infer production runtime ownership from this repo's experiment code.

## Repo Ownership

This repo owns:

- `src/**` for reusable embedding, evaluation, reporting, and preprocessing modules
- `scripts/pipeline/**` for the end-to-end experiment pipeline
- `scripts/tools/**` for export, atlas, model packaging, and evaluation helpers
- `config/**`, `data/**`, `analysis/**`, `report*.md`, and `TODO.md` for experiment configuration and durable findings
- `pyproject.toml` for Python runtime and dependency baselines

This repo does not own:

- product search runtime behavior
- MCP transport behavior
- workspace integration state after merge

Route those tasks to:

- `tiangong-lca-mcp` for MCP-exposed search tools or validation endpoints
- the owning product/runtime repo for production query behavior
- `lca-workspace` for root integration after merge

## Runtime Facts

- Repo-local AI-doc maintenance is enforced by the `.githooks/pre-push` hook via `scripts/ai-doc-lint-gate.sh`; `.github/workflows/ai-doc-lint.yml` is manual-only fallback.
- `.github/workflows/portable-python.yml` adds scoped, secretless Python 3.12 PR checks using the existing isolated contracts and syntax baseline. It never syncs the GPU project. `ai/dependency-maintenance.md` owns root-only updater coverage, bounded traffic and the separate ML qualification requirement.
- Python baseline: `>=3.12`
- Package/dependency manager: `uv`
- dependency/runtime proof: `uv lock --check --python 3.12` and `PYTHONPATH=. uv run --isolated --no-project --python 3.12 --with tidas-sdk==0.2.14 python -m unittest discover -s tests -v`
- source syntax proof: `python3.12 -m compileall -q src scripts tests`
- TIDAS baseline: exact Python SDK `0.2.14`; `src.pre_process` and the legacy public wrapper delegate to the SDK's canonical `TidasProcess.to_markdown()` instead of duplicating the 0.1 object model
- latest-compatible ML boundary: Sentence Transformers `5.7.x` remains below 6 because 6.0 requires Transformers 5; the released Qwen fine-tuning stack retains exact PEFT `0.13.0`, Transformers `4.51.3`, and Torch `2.9.1`
- CUDA boundary: `faiss-gpu-cu12` and Torch `2.9.1` retain CUDA 12 artifacts; the dependency contract rejects CUDA 13 lock drift
- the full GPU project cannot sync on macOS because `faiss-gpu-cu12` publishes Linux x86_64 wheels only; macOS runs the isolated TIDAS/dependency contracts, while full training/import validation belongs on the reviewed Linux x86_64 CUDA 12 environment

- CPU qualification: `scripts/tools/qualify_cpu_embedding.py` is an offline, resident-model experiment. Its isolated Docker runtime reproduces deployed Torch 2.6.0 CPU and Transformers 4.51.3 without syncing or upgrading the GPU training environment. Bind immutable artifact/data hashes, record cgroup limits and warm sample counts, and use regenerated same-artifact FP32 vectors as a numerical reference; legacy caches lacking binary identity are not production goldens. Historical train-related data does not prove unseen-document generalization.

- Public Nemotron releases: `docs/releases/lca-nemotron-embed-1b.md` owns download, integrity and benchmark usage. `config/releases/` and `config/datasets/` pin Hub commits and every payload's size/SHA256. Download and score tools use the standard library; `environments/nemotron-inference/` independently locks the optional Linux CUDA 12.9 inference runtime. This does not replace the Qwen baseline. The published benchmark uses fixed-five-reference probability coverage/nDCG, not the older Qwen evaluation protocol. Store downloaded weights/data in ignored caches, not Git.

## Hard Boundaries

- Do not treat experiment code here as the source of truth for production search contracts
- Do not change pipeline assumptions without updating the repo contract in the same change
- Do not replace the reviewed Qwen/Transformers 4 + CUDA 12 stack with Sentence Transformers 6, Transformers 5, or CUDA 13 without a separately tracked GPU qualification
- Do not treat a merged repo PR here as workspace-delivery complete if the root repo still needs a submodule bump

## Workspace Integration

A merged PR in `lca-domain-embedding` is repo-complete, not delivery-complete.

If the change must ship through the workspace:

1. merge the child PR into `lca-domain-embedding`
2. update the `lca-workspace` submodule pointer deliberately
3. complete any later workspace-level validation that depends on the updated embedding assets

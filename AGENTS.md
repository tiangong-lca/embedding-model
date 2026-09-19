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
  - data/**
  - analysis/**
  - report.md
  - report.ZH.md
lastReviewedAt: 2026-09-15
lastReviewedCommit: 2030e3b0abe82520f97e63b3bf086729a018dcc7
lastReviewedNote: "Reviewed for Embedding #8 and #14: Python TIDAS SDK is exact 0.2.14, compatible direct dependencies are current, Qwen/CUDA12 pins are executable contracts, canonical Markdown and strict-invalid behavior have isolated Python 3.12 proof, and manual CI invokes the unchanged unit-then-lint shell gate once."
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
- Python baseline: `>=3.12`
- Package/dependency manager: `uv`
- dependency/runtime proof: `uv lock --check --python 3.12` and `PYTHONPATH=. uv run --isolated --no-project --python 3.12 --with tidas-sdk==0.2.14 python -m unittest discover -s tests -v`
- source syntax proof: `python3.12 -m compileall -q src scripts tests`
- TIDAS baseline: exact Python SDK `0.2.14`; `src.pre_process` and the legacy public wrapper delegate to the SDK's canonical `TidasProcess.to_markdown()` instead of duplicating the 0.1 object model
- latest-compatible ML boundary: Sentence Transformers `5.7.x` remains below 6 because 6.0 requires Transformers 5; the released Qwen fine-tuning stack retains exact PEFT `0.13.0`, Transformers `4.51.3`, and Torch `2.9.1`
- CUDA boundary: `faiss-gpu-cu12` and Torch `2.9.1` retain CUDA 12 artifacts; the dependency contract rejects CUDA 13 lock drift
- the full GPU project cannot sync on macOS because `faiss-gpu-cu12` publishes Linux x86_64 wheels only; macOS runs the isolated TIDAS/dependency contracts, while full training/import validation belongs on the reviewed Linux x86_64 CUDA 12 environment

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

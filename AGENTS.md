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
  - src/**
  - scripts/**
  - config/**
  - data/**
  - analysis/**
  - report.md
  - report.ZH.md
lastReviewedAt: 2026-09-15
lastReviewedCommit: 2030e3b0abe82520f97e63b3bf086729a018dcc7
lastReviewedNote: "Reviewed for Embedding #14: manual CI now invokes the unchanged unit-then-lint shell gate once. Nine unit tests, real workflow/gate traces, duplicate and set-e mutation RED, primary independent diff review and actionlint pass. Existing model/data/runtime ownership, manual trigger and gate failure behavior remain; exact CI and root delivery pending."
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
- This repo currently has no single checked-in green-bar test wrapper like `pytest` or `make check`; validation is change-scoped and should use the narrowest safe pipeline or tool command that proves the touched stage
- If future canonical validation commands are added, document them here and in `ai/repo.yaml` in the same change

## Hard Boundaries

- Do not treat experiment code here as the source of truth for production search contracts
- Do not change pipeline assumptions without updating the repo contract in the same change
- Do not treat a merged repo PR here as workspace-delivery complete if the root repo still needs a submodule bump

## Workspace Integration

A merged PR in `lca-domain-embedding` is repo-complete, not delivery-complete.

If the change must ship through the workspace:

1. merge the child PR into `lca-domain-embedding`
2. update the `lca-workspace` submodule pointer deliberately
3. complete any later workspace-level validation that depends on the updated embedding assets

---
title: Embedding Dependency Maintenance
docType: guide
scope: repo
status: active
authoritative: false
owner: domain-embedding
language: en
whenToUse:
  - when reviewing root dependency or Actions proposals
  - when deciding what portable PR CI establishes
whenToUpdate:
  - when discovery coverage, traffic budgets or portable checks change
checkPaths:
  - .github/dependabot.yml
  - .github/workflows/portable-python.yml
  - ai/dependency-maintenance.md
lastReviewedAt: 2026-10-16
lastReviewedCommit: 8364123e0e16533b2b1511c581f434c8d239ed33
lastReviewedNote: "Embedding #54: two bounded discovery exclusions follow actual resolver failures and official metadata; root runtime/lock/tests and independent GPU qualification remain unchanged."
---

# Dependency maintenance

The root `uv` graph and GitHub Actions references are monitored weekly on
Tuesday in Asia/Shanghai, with a seven-day version cooldown and open version-PR
limits of two and one respectively. Related patch/minor training dependencies
form one compatibility group, supporting libraries a separate group, and Actions
a third group. Unexcluded major proposals remain individually visible; the two
bounded incompatible lines below are explicit exceptions, not a blanket
major-version ignore. Security settings and security PR limits are separate.

Discovery does not grant merge, model publication, training or deployment
authority. Automatic merge stays off. Before associating a bot proposal with
delivery, verify both native merge state and any independent merge service's
effective live policy, including queued work. When that policy is unknown,
leave proposals unadopted. Use the existing workspace controller for accepted
grouped delivery, review, merge, completion and exact workspace integration.

## Portable PR proof

`.github/workflows/portable-python.yml` runs only when the root manifest/lock,
Python sources, tests, experiment configuration or repository discovery/CI inputs
change. Unrelated documentation-only PRs gain no job. No new globally required
check or local development step is introduced.

The workflow selects Python 3.12 and pins uv 0.13.1, verifies the unchanged root
lock without syncing the project, then runs the existing isolated TIDAS SDK
0.2.14 unittest discovery, source compilation and repository automation unit
tests. The isolated environment
installs the SDK and its dependencies only. It imports no trained model and
downloads no model/data artifacts. Existing NumPy-dependent CPU-qualification
cases explicitly skip in this lightweight environment; their numerical proof
belongs to the separate qualification runtime.

The existing dependency contracts continue to assert every reviewed direct
manifest constraint and exact locked version, including PEFT 0.13.0,
Transformers 4.51.3, Torch 2.9.1, Sentence Transformers below 6 and CUDA 12.
A bot proposal that violates them should fail. Do not relax those assertions
as routine updater accommodation. Accepted ML changes require a separately
tracked compatibility decision and Linux x86_64 CUDA qualification of actual
model imports, fine-tuning and relevant FAISS/inference behavior. Portable CI
does not establish that qualification, even when it is green.

## Coverage and hosted follow-through

### Known incompatible version lines

[Hosted updater 38846295093](https://github.com/tiangong-lca/embedding-model/actions/runs/38846295093)
failed on two concrete proposals. [Embedding #54](https://github.com/tiangong-lca/embedding-model/issues/54)
owns these discovery-only exclusions:

| Package | Excluded version range | Observed incompatibility |
| --- | --- | --- |
| `datasets` | `>=5.1.0,<5.2.0` | 5.1.0 requires Hugging Face Hub `>=1.31.0,<3.0`; retained Transformers 4.51.3 requires `>=0.30.0,<1.0`. |
| `sentence-transformers` | `>=6.0.0,<7.0.0` | Both 6.0.0 and the failed 6.1.0 proposal require Transformers `>=5.0.0,<6.0.0`, contradicting the retained exact 4.51.3 pin. |

These are bounded affected version lines. The compatible locked Datasets 5.0.1
and Sentence Transformers 5.7.0 remain selectable. Other package versions and
unrelated major proposals remain visible. Future Datasets 5.2/6 or Sentence
Transformers 7 are not presumed compatible; inspect their actual metadata and
CI when proposed rather than silently extending these exclusions.

[Root-stack qualification #53](https://github.com/tiangong-lca/embedding-model/issues/53)
owns the explicit revisit: remove an exclusion when the corresponding reviewed
manifest/lock/contract migration and authorized Linux CUDA/model qualification
are merged, then prove the fresh hosted updater and candidate validation.
A blocked-version security alert still needs triage; an exclusion does not
retire that alert or authorize an unqualified migration. Existing unadopted bot
proposals retain their own review and validation requirements.

The [official ignore option](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference#versions-ignore)
uses package-manager version syntax. Dependabot's [uv requirement adapter](https://github.com/dependabot/dependabot-core/blob/main/uv/lib/dependabot/uv/requirement.rb)
delegates to the Python requirement parser, whose
[range tests](https://github.com/dependabot/dependabot-core/blob/main/uv/spec/dependabot/uv/requirement_spec.rb)
cover comma-separated conjunctions. The versioned PyPI metadata establishes the
dependency boundaries: [Datasets 5.1.0](https://pypi.org/pypi/datasets/5.1.0/json),
[Sentence Transformers 6.0.0](https://pypi.org/pypi/sentence-transformers/6.0.0/json),
[6.1.0](https://pypi.org/pypi/sentence-transformers/6.1.0/json) and
[Transformers 4.51.3](https://pypi.org/pypi/transformers/4.51.3/json).

### Independent environments

Only the root training graph is monitored here. The independently pinned
`environments/nemotron-inference/` graph uses Torch 2.11.0 with CUDA 12.9,
Transformers 5.4.0 and Sentence Transformers 5.3.0. It is not covered by this
root updater or root lock check. Separate work must establish that environment's
own lock and inference validation before enabling its updater; this coverage gap
is tracked in [Embedding #48](https://github.com/tiangong-lca/embedding-model/issues/48).
CPU benchmark
Docker reproduction pins, model/dataset manifests, immutable artifact hashes and
cloud environments are also outside this updater; retain their existing owners.

At the normal weekly queue review, inspect the hosted updater job and exact
candidate's portable CI. Record failed scans, lock drift and accepted work in the
owning Issue. A missing proposal is not proof of a successful scan. Security
triage follows urgency. No extra reminder or status feed is required.

Configuration and local tests do not prove that GitHub's hosted updater can
regenerate this root graph. The first real candidate still needs the frozen
lock check, unchanged contracts and any applicable independent GPU qualification.
See the [workspace policy](https://github.com/tiangong-lca/workspace/blob/main/_docs/workspace-dependency-update-policy.md).

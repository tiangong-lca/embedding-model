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
lastReviewedAt: 2026-10-19
lastReviewedCommit: 25c37fa94a05cfe6863fcaf9c4331e327b2fd645
lastReviewedNote: "Embedding #66: reviewed the explicit native CPU diagnostic lane, frozen artifact-bound full tuples, default import-order failures and separately labeled DeepSpeed-first synthetic diagnostics. Portable discovery, exact root ML/security pins, CUDA12 and independent #53 qualification remain unchanged; CPU proof does not accept original #51."
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

The workflow selects Python 3.12 and pins uv 0.13.1, checks the root lock
without syncing the project, then invokes `scripts/tools/run_portable_tests.py`.
The runner uses `uv export --frozen` to derive temporary root constraints and
selects TIDAS, Matplotlib, OpenAI, python-dotenv, Supabase, tqdm and Loguru in an
isolated `uv run --no-project` environment. Every installed transitive, including
NumPy, Pydantic, httpx and httpx2, must match the root lock; missing, unrecorded
or training packages fail before the complete unittest suite runs. The export
is disposable proof input, not a second committed dependency authority. Its
requirements format cannot express per-package indexes, so standard-library
code downloads only the selected OpenAI universal wheel from its root-locked,
credential-free official HTTPS URL into a fresh owned path. Its SHA256 and size
are checked before the isolated installer runs; that verified local wheel is
passed to uv. Runtime proof rechecks those bytes and the installed name/version
and local direct-source URL, binding them to the original source/lock receipt.
uv 0.13.1 did not enforce a deliberately wrong URL hash fragment in the reviewed
negative probe and left PEP610 archive_info empty; neither is used as integrity
proof. A maintained bad-hash case must stop before the isolated installer.
Other packages retain the root default index and frozen version constraints.
The runtime/lock inventory and commands are retained in the requested new
evidence directory. Source compilation and repository automation tests follow.

Tests execute real consumer serialization, mocked HTTP transports, headless
plotting, owned dummy dotenv inputs and local Process/Flow conversion fixtures.
They perform no provider/auth/database request, model import or model/data
download. NumPy-only CPU helper cases now execute in this subset; this does not
qualify any model, resident CPU runtime, training pipeline or GPU environment.

### Reviewed supporting-library adoption

Embedding #65 composes original #52 with accepted main, deliberately adopting
TIDAS 0.3.0 for the canonical Markdown consumers, Matplotlib 3.11.2, OpenAI
3.35.1, python-dotenv 1.2.4 and the aligned Supabase 2.33.0 family. The original
bot proposed OpenAI 3.26.1 and Supabase 2.32.0; current official stable metadata
and offline consumer proof support the reviewed manual selection. The existing
seven-day discovery cooldown remains configured. Accepted PyJWT 2.15.1,
multidict 6.9.1 and urllib3 2.8.0 are preserved rather than overwritten by the
older proposal's stale lock.

The default registry remains Aliyun. Only OpenAI has an `explicit = true` public
PyPI index binding. In the reviewed uv 0.13.1 environment, the identical minimal
`openai==3.35.1` requirement failed candidate discovery against Aliyun and
resolved through PyPI, including an isolated Python 3.12 reproduction. Aliyun
advertised and served the same valid wheel/hash; a missing artifact or Python
incompatibility is not established. The root cause below that resolver
interface remains unproven. This narrow source exception preserves every
unrelated registry origin/hash and does not change index priority/trust policy,
release-age settings, or discovery cooldown.

The reviewed TIDAS comparison contains 127 matching local entity/language,
pipeline/sanitization, strict-incomplete and batch/resume/preprocessing results.
Handwritten SDK Markdown/factory code is unchanged between the two releases,
but generated models and schemas changed. These partially authored fixtures
are not a real strict-valid corpus or all-SDK acceptance/diagnostic equivalence.
The complete maintained suite retains strict-invalid checks and fixes reviewed
version facts without dropping assertions. Existing Process output remains
`processs/`; its spelling is unrelated to this upgrade.

python-dotenv 1.2.4 intentionally fixes `KEY= # comment` to mean an empty string;
1.2.3 returned literal `# comment`. Ordinary dummy-key retrieval and environment
precedence are tested; arbitrary dotenv byte-equivalence is not claimed. Actual
client requests/results and decoded heatmap pixels matched the prior baseline
in the bounded offline investigation. Live providers, database permissions/RLS,
real corpus migration and platform-wide rendering remain separate proof.

The existing dependency contracts continue to assert every reviewed direct
manifest constraint and exact locked version, including PEFT 0.13.0,
Transformers 4.51.3, Torch 2.9.1, Sentence Transformers below 6 and CUDA 12.
A bot proposal that violates them should fail. Do not relax those assertions
as routine updater accommodation. Accepted ML changes require a separately
tracked compatibility decision and Linux x86_64 CUDA qualification of actual
model imports, fine-tuning and relevant FAISS/inference behavior. Portable CI
does not establish that qualification, even when it is green.

## Opt-in CPU API evidence

[The explicit diagnostic lane](../scripts/diagnostics/README.md) compares complete, source-bound root-stack proposals on native macOS arm64 with actual Torch/Transformers/PEFT/Sentence Transformers/Accelerate/DeepSpeed/ModelScope packages. Owned environments and original-lock artifact hashes remain separate from the ML-free portable runner. Its tiny random model and authored data can expose Python/API failures without pretrained downloads. Native network and compiler execution restrictions, actual RSS/time/disk limits and per-stage failures are retained. Default import-order failures are primary results; supplemental DeepSpeed-first diagnostics cannot replace them or turn a partial tuple into a pass.

This source fixture delivery does not adopt any dependency update, relax CUDA12 assertions, accept original #51 or retire the independent #53 scope. A pure-Python DeepSpeed wheel or absence of installed native extensions alone does not prove that its historical build never executed a compiler capability probe. An explicitly compiler-denied build must be distinguished from earlier unconfined builds; denied probes are not real native-op compatibility proof. Actual Linux CUDA/model qualification remains required.

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

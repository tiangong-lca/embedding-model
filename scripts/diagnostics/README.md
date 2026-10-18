# Explicit CPU training API diagnostics

This opt-in lane diagnoses the actual Python training libraries with a tiny,
randomly initialized Qwen3 model, an authored tokenizer and four synthetic
training records. It does not load a pretrained model or research dataset.
It is outside `tests/` and is never invoked by portable unittest discovery.

The portable runner deliberately excludes these ML libraries. The separate
`tools/qualify_cpu_embedding.py` experiment reproduces deployed Torch 2.6.0
inference against an existing artifact. Neither establishes compatibility of
the root training stack or replaces this diagnostic.

## Inputs and provisioning

Start a tracked, authorized compatibility scope before acquiring dependencies.
Run only on native macOS arm64 with Python 3.12 and a separately verified
uv 0.13.1. Provision fresh owned venvs; the runner does **not** install packages.
Preserve the real `HOME`, and put uv, pip, build, model-library caches and
temporary files in the task's owned resources directory.

For each row, save the exact source commit, manifest and lock. Export the frozen
root lock without installing the GPU project:

```bash
uv export --project /owned/row-inputs --frozen --no-dev --no-emit-project \
  --no-hashes --output-file /owned/evidence/root-constraints.txt
```

Resolve the selected Torch, Transformers, PEFT, Sentence Transformers,
Accelerate, DeepSpeed, ModelScope and Datasets closure against those constraints.
Full project sync is unavailable on macOS because FAISS CUDA12 has no Mac wheel.
Check every selected version against the original lock. Download only official
PyPI artifacts, verify each wheel/sdist's bytes, size and SHA256 against both
official metadata and the row lock, and install the complete closure from those
local artifacts using `uv pip install --no-index --require-hashes`. Do not use
`--no-deps` or install an unconstrained replacement environment.

DeepSpeed is an sdist. Use `DS_BUILD_OPS=0`, `DS_ACCELERATOR=cpu`, separately bound
pure-Python build prerequisites and `--no-build-isolation`. A source build of its
Python wheel is distinct from building native operations; native/JIT/CUDA
operation compilation is excluded. Preserve all actual build/install failures.
`wheel` is the only permitted extra runtime-visible build prerequisite; record
its exact official artifact separately. Setuptools and every other distribution
must remain in the row lock. Bind and inspect build outputs and retain the
source/artifact inputs before cleaning the venvs.

Supply a JSON artifact receipt with this shape (one artifact per installed
distribution, including the DeepSpeed source and the extra build prerequisite):

```json
{
  "runtime_build_prerequisites": {"wheel": "<verified exact version>"},
  "artifacts": [
    {
      "name": "torch",
      "version": "<row-locked version>",
      "path": "/owned/artifacts/<sha256>/<original-wheel-filename>",
      "url": "https://files.pythonhosted.org/<official-artifact-path>",
      "sha256": "<verified hex digest>"
    }
  ]
}
```

The environment stage verifies installed versions, original lock artifact
hashes, current local artifact bytes and installed `direct_url.json` paths.
That file's empty `archive_info` is not an integrity guarantee; the separately
checked bytes and original root lock provide integrity. The receipt must contain
the entire installed closure. A failed installation remains a failed row, even
when independent stages can execute.

## Execute and retain evidence

```bash
python scripts/diagnostics/run_ml_cpu_diagnostics.py \
  --row candidate --python /owned/venv/bin/python \
  --manifest /owned/row-inputs/pyproject.toml --lock /owned/row-inputs/uv.lock \
  --build-receipt /owned/evidence/runtime-artifacts.json \
  --resources /owned/resources --output /owned/evidence/new-candidate-run
```

The output directory must be new. Run rows sequentially, using the same fixture
source, synthetic inputs and constraints policy. The default bounds are 12 GiB
owned disk, 8 GiB actual process-group RSS, two compute threads, 120 seconds per
stage, 300 seconds per trainer stage and 45 minutes for the invocation. Pass
`--total-seconds` with the **remaining** overall task budget; separate row
invocations do not reset a previously started task deadline. Budget acquisition
and installation separately (Issue66 used at most 600 seconds per row).

The supervisor checks the native deny-network sandbox before imports, measures
actual process-group RSS, records the child's measured peak RSS, and terminates
only its owned process group when a bound is crossed. Monitoring failure also
reaps the owned child. It does not use virtual-address space as a RAM proxy.
Runtime environments omit ambient credentials and provider settings; GPU/MPS
execution is excluded by explicit CPU devices and tensor assertions. Every stage
runs separately; failures and tracebacks persist in its receipt and log.

Native networking stays denied, including loopback. The Python audit guard also
rejects external connections, DNS, sends and unknown subprocesses. Exact
`uname -p` and `file -b <the owned Python binary>` queries are recorded and
permitted. The known urllib3 IPv6 `::1:0` bind probe and DeepSpeed's attempted
`clang ... /owned/tmp/.../test.c` compatibility probe are separately recorded
and **denied before execution**. Libraries may catch these denied capability
probes; this does not establish IPv6, native operation or compiler support.
No downloader, compiler, extension builder or business/provider request is
permitted by these exceptions.

Ordinary `qwen`, `lora` and `trainer` stages preserve their natural import order.
`ordered-*` stages explicitly pre-import the actual DeepSpeed package and are
supplemental diagnostics of import order. They use the same complete tuple;
they do not patch a library or repository helper, fix the product entrypoint,
replace ordinary failures, or turn a partial row into a pass. A row is complete
only when every stage was attempted. Its aggregate passes only when every stage
passes; `accepts_original51` is always false.

The stages cover real CPU model forward/normalization/padding, ordinary LoRA
gradients/optimizer/base freeze/adapter save and reload, repository training
helpers and a two-step SentenceTransformerTrainer, Accelerator CPU backward,
DeepSpeed import/configuration and the actual ModelScope downloader's signature.
The latter binds the consumer's keywords without calling its download helper
(which can install packages or access a service). Tiny model save/reload uses
the same explicit eager attention implementation on both sides, avoiding an
unrelated implicit eager-to-SDPA backend change.

Run the standard-library supervisor controls explicitly:

```bash
python scripts/diagnostics/check_ml_cpu_diagnostics.py
```

These controls test resource limits, child cleanup and network denial. They do
not mock ML libraries and do not count as ML runtime proof.

## Original51 boundary

Issue66 compares baseline main `25c37fa` with original51 `a4ed0d0`:

| Row | Torch | PEFT | Accelerate | DeepSpeed | ModelScope |
| --- | --- | --- | --- | --- | --- |
| B | 2.9.1 | 0.13.0 | 1.14.0 | 0.19.6 | 1.39.1 |
| C | 2.14.1 | 0.21.2 | 1.15.0 | 0.19.7 | 1.40.2 |

Both use Transformers 4.51.3, Sentence Transformers 5.7.0 and Datasets 5.0.1.
The original proposal's Linux lock introduces CUDA13 beside FAISS CUDA12.
Successful CPU diagnostics cannot relax the reviewed CUDA12 contract, accept
original51, change its dependency assertions or retire the independent Issue53.
Actual Linux x86_64 GPU/model/fine-tuning/FAISS and numerical acceptance still
require an explicitly authorized qualification with immutable model/data inputs,
resource ownership and scientific thresholds. Source delivery of this fixture
requires its own reviewed PR and exact workspace integration.

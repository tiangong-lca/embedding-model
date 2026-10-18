#!/usr/bin/env python3
"""Explicit, offline CPU API diagnostics. Never part of portable unittest discovery.

Uses tiny random Qwen3 weights and authored tokenizer/data, not a pretrained model.
Each stage runs in a separately supervised interpreter so an import failure does
not hide independent diagnostics. No result qualifies CUDA, learned model quality,
the released artifact, distributed training or production service behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import tomllib
import traceback
from urllib.parse import unquote, urlsplit

PACKAGES = {
    "torch": "torch", "transformers": "transformers", "peft": "peft",
    "sentence-transformers": "sentence_transformers", "accelerate": "accelerate",
    "deepspeed": "deepspeed", "modelscope": "modelscope", "datasets": "datasets",
}
STAGES = ("environment", "network-guard", *(f"import-{n}" for n in PACKAGES),
          "qwen", "lora", "trainer", "accelerate", "deepspeed", "modelscope",
          "ordered-qwen", "ordered-lora", "ordered-trainer")
NETWORK_EVENTS = {"socket.connect", "socket.connect_ex", "socket.getaddrinfo",
                  "socket.sendto", "socket.bind", "socket.gethostbyname",
                  "socket.gethostbyaddr"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(name):
    return name.lower().replace("_", "-").replace(".", "-")


def guard_network(attempts, expected_denials, queries):
    def audit(event, args):
        if event in NETWORK_EVENTS:
            # urllib3 probes IPv6 support at import by binding ::1:0. The
            # native sandbox still denies it; record this known, caught probe
            # separately from an unexpected connection/DNS/service request.
            if event == "socket.bind" and len(args) == 2 and args[1] == ("::1", 0):
                expected_denials.append({"event": event, "reason": "urllib3 IPv6 loopback capability probe"})
                raise PermissionError("Offline sandbox denies loopback bind probe")
            attempts.append({"event": event, "arguments": repr(args)[:500]})
            raise PermissionError(f"Offline CPU fixture denied {event}")
        if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.exec"}:
            executable, argv = args[:2] if event == "subprocess.Popen" else ("", [])
            if isinstance(argv, (list, tuple)):
                name = Path(executable).name
                # Python platform queries are read-only fixed system commands.
                if (name == "uname" and list(argv[1:]) == ["-p"]) or (
                    name == "file" and len(argv) == 3 and argv[1] == "-b"
                    and Path(argv[2]).resolve() == Path(sys.executable).resolve()):
                    queries.append({"event": event, "executable": str(executable), "argv": list(argv)})
                    return
                # DeepSpeed import catches a failed native compatibility probe.
                # Deny compilation before process start, even for this probe.
                sources = [Path(a) for a in argv if isinstance(a, str) and a.endswith("/test.c")]
                if name in {"clang", "cc", "gcc"} and sources and all(
                    p.is_relative_to(Path(os.environ["TMPDIR"])) for p in sources):
                    expected_denials.append({"event": event, "reason": "DeepSpeed native capability compile probe denied",
                                            "executable": str(executable), "argv": list(argv)})
                    raise PermissionError("Native compilation remains forbidden")
            # No downloading helper, unknown subprocess or extension build may run.
            attempts.append({"event": event, "executable": str(executable), "argv": repr(argv)[:500]})
            raise PermissionError(f"CPU fixture denied process execution: {event}")
    sys.addaudithook(audit)


def environment(args):
    locked = {}
    for package in tomllib.loads(Path(args.lock).read_text())["package"]:
        locked.setdefault(canonical(package["name"]), set()).add(package["version"])
    build = json.loads(Path(args.build_receipt).read_text())
    exceptions = build.get("runtime_build_prerequisites", {})
    if set(exceptions) - {"wheel"}:
        raise ValueError("Only the separately bound pure-Python wheel build prerequisite is permitted")
    installed = {canonical(d.metadata["Name"]): d.version for d in metadata.distributions()}
    drift = {n: v for n, v in installed.items() if v not in locked.get(n, set())
             and exceptions.get(n) != v}
    missing = set(PACKAGES) - installed.keys()
    if missing or drift:
        raise ValueError(f"Incomplete/drifted row: missing={sorted(missing)}, drift={drift}")
    artifacts = {canonical(a["name"]): a for a in build["artifacts"]}
    verified = {}
    lock = tomllib.loads(Path(args.lock).read_text())
    for name, version in installed.items():
        artifact = artifacts[name]
        path = Path(artifact["path"])
        official = urlsplit(artifact["url"])
        if (official.scheme != "https" or official.hostname != "files.pythonhosted.org"
                or official.username or official.password or official.query or official.fragment):
            raise ValueError(f"Expected a credential-free official PyPI artifact URL: {name}")
        if name in exceptions and not path.name.endswith("-py3-none-any.whl"):
            raise ValueError("Build-only exception must be a pure-Python universal wheel")
        if artifact["version"] != version or not path.is_file() or digest(path) != artifact["sha256"]:
            raise ValueError(f"Installed artifact content/identity drift: {name}")
        if name not in exceptions:
            records = [p for p in lock["package"] if canonical(p["name"]) == name and p["version"] == version]
            hashes = {a["hash"] for p in records for a in p.get("wheels", [])}
            hashes.update(p["sdist"]["hash"] for p in records if "sdist" in p)
            if "sha256:"+artifact["sha256"] not in hashes:
                raise ValueError(f"Artifact is not bound by this root lock: {name}")
        raw = metadata.distribution(name).read_text("direct_url.json")
        direct = json.loads(raw) if raw else {}
        parsed = urlsplit(direct.get("url", ""))
        if parsed.scheme != "file" or Path(unquote(parsed.path)).resolve() != path.resolve():
            raise ValueError(f"Installed source differs from verified local artifact: {name}")
        verified[name] = {"version": version, "sha256": artifact["sha256"], "official_url": artifact["url"],
                          "installed_direct_url": direct["url"]}
    assert sys.version_info[:2] == (3, 12)
    assert platform.system() == "Darwin" and platform.machine() == "arm64"
    return {"installed": installed, "build_only_exceptions": exceptions,
            "python": sys.version, "executable": sys.executable,
            "artifacts_verified": verified, "lock_sha256": digest(args.lock), "platform": platform.platform()}


def cpu_setup():
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(66)
    if torch.cuda.is_available():
        raise ValueError("Visible CUDA is forbidden")
    return torch


def assert_cpu(model):
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Model escaped CPU")


def tiny_model(directory):
    torch = cpu_setup()
    from transformers import Qwen3Config, Qwen3Model, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=2, head_dim=16,
                        max_position_embeddings=64, pad_token_id=0,
                        eos_token_id=2, bos_token_id=1, attention_dropout=0.0,
                        sliding_window=None)
    config._attn_implementation = "eager"
    model = Qwen3Model(config).to("cpu").float()
    words = ["[PAD]", "[UNK]", "[EOS]", "electricity", "solar", "water", "carbon",
             "heat", "renewable", "process", "steel", "recycle", "generation"]
    raw = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="[PAD]",
                                        unk_token="[UNK]", eos_token="[EOS]",
                                        model_max_length=32, padding_side="right")
    directory.mkdir()
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    size = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    if size >= 5 * 1024**2:
        raise ValueError("Tiny fixture exceeded 5 MiB")
    assert_cpu(model)
    return torch, model, tokenizer, {"bytes": size, "config": config.to_dict(),
        "sha256": {str(p.relative_to(directory)): digest(p) for p in directory.rglob("*") if p.is_file()}}


def qwen(args):
    torch, model, tokenizer, identity = tiny_model(args.output / "tiny-model")
    from transformers import AutoModel
    texts = ["solar electricity", "water", "steel recycle heat"]
    def encode(texts):
        inputs = tokenizer(texts, padding=True, return_tensors="pt")
        assert all(t.device.type == "cpu" for t in inputs.values())
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
            index = inputs["attention_mask"].sum(1) - 1
            return torch.nn.functional.normalize(hidden[torch.arange(len(texts)), index], dim=1)
    model.eval()
    batched = encode(texts)
    singles = torch.cat([encode([text]) for text in texts])
    assert torch.isfinite(batched).all()
    assert torch.allclose(batched.norm(dim=1), torch.ones(3), atol=1e-6)
    assert torch.allclose(batched, singles, atol=2e-5, rtol=2e-5)
    reloaded = AutoModel.from_pretrained(args.output / "tiny-model", local_files_only=True,
                                        attn_implementation="eager").eval()
    assert_cpu(reloaded)
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    with torch.no_grad():
        difference = float((model(**inputs).last_hidden_state - reloaded(**inputs).last_hidden_state).abs().max())
    assert difference == 0
    return {"fixture": identity, "shape": list(batched.shape),
            "batch_single_max_difference": float((batched-singles).abs().max()),
            "save_reload_max_difference": difference}


def lora(args):
    torch, base, tokenizer, identity = tiny_model(args.output / "tiny-model")
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    from transformers import AutoModel
    adapter = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
                    target_modules=["q_proj", "v_proj"], task_type=TaskType.FEATURE_EXTRACTION))
    adapter.to("cpu").train()
    before = {n: p.detach().clone() for n, p in adapter.named_parameters()}
    trainable = [n for n, p in adapter.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    inputs = tokenizer(["solar electricity water", "steel recycle heat"], padding=True, return_tensors="pt")
    output = adapter(**inputs).last_hidden_state
    loss = (output * torch.linspace(-1, 1, output.numel()).reshape_as(output)).mean()
    loss.backward()
    gradients = {n: float(p.grad.abs().sum()) for n, p in adapter.named_parameters()
                 if p.requires_grad and p.grad is not None}
    assert gradients and all(torch.isfinite(p.grad).all() for p in adapter.parameters() if p.grad is not None)
    assert any(v > 0 for v in gradients.values())
    torch.optim.Adam([p for p in adapter.parameters() if p.requires_grad], lr=0.01).step()
    changed = [n for n, p in adapter.named_parameters() if not torch.equal(before[n], p.detach())]
    assert changed and all(n in trainable for n in changed)
    assert_cpu(adapter)
    adapter.eval()
    with torch.no_grad():
        expected = adapter(**inputs).last_hidden_state
        with adapter.disable_adapter():
            disabled = adapter(**inputs).last_hidden_state
        clean = AutoModel.from_pretrained(args.output / "tiny-model", local_files_only=True,
                                         attn_implementation="eager").eval()
        assert torch.equal(disabled, clean(**inputs).last_hidden_state)
    adapter.save_pretrained(args.output / "adapter", safe_serialization=True)
    loaded = PeftModel.from_pretrained(clean, args.output / "adapter", local_files_only=True).eval()
    with torch.no_grad():
        difference = float((loaded(**inputs).last_hidden_state - expected).abs().max())
    assert difference <= 1e-6
    return {"fixture": identity, "trainable": trainable, "grad_abs_sums": gradients,
            "changed": changed, "loss": float(loss.detach()), "reload_max_difference": difference}


def repository_pipeline(args):
    path = args.source / "scripts/pipeline/04_finetune_st.py"
    spec = importlib.util.spec_from_file_location("repository_finetune_st", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {"path": str(path), "sha256": digest(path)}


def trainer(args):
    torch, _, _, identity = tiny_model(args.output / "tiny-model")
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, models
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers
    from transformers import TrainerCallback
    pipeline, source = repository_pipeline(args)
    transform = models.Transformer(str(args.output / "tiny-model"), max_seq_length=16,
                                   model_args={"local_files_only": True, "attn_implementation": "eager"},
                                   tokenizer_args={"local_files_only": True})
    pool = models.Pooling(32, pooling_mode="lasttoken")
    model = SentenceTransformer(modules=[transform, pool], device="cpu")
    pipeline.configure_model_gradient_checkpointing(model, True)
    assert model[0].auto_model.is_gradient_checkpointing
    pipeline.configure_model_gradient_checkpointing(model, False)
    assert not model[0].auto_model.is_gradient_checkpointing
    pipeline.enforce_model_torch_dtype(model, torch.float32)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    rows = [{"query": q, "pos": [p], "neg": []} for q, p in [
        ("solar electricity", "renewable generation"), ("steel process", "recycle heat"),
        ("water generation", "carbon process"), ("renewable steel", "electricity water")]]
    data = args.output / "synthetic-training.jsonl"
    data.write_text("".join(json.dumps(row)+"\n" for row in rows))
    dataset = pipeline.load_training_data(str(data))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    training = SentenceTransformerTrainingArguments(output_dir=str(args.output / "checkpoints"),
        max_steps=2, per_device_train_batch_size=2, learning_rate=1e-3, use_cpu=True,
        bf16=False, fp16=False, gradient_checkpointing=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES, dataloader_drop_last=True,
        dataloader_num_workers=0, report_to=[], save_strategy="no", logging_steps=1,
        eval_strategy="no", seed=66, disable_tqdm=True)
    pipeline.force_single_gpu(training)
    assert training.device.type == "cpu"
    instance = SentenceTransformerTrainer(model=model, args=training,
                           train_dataset=dataset, loss=MultipleNegativesRankingLoss(model))
    gradient_steps = []
    class CaptureGradients(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert gradients and all(g.device.type == "cpu" and torch.isfinite(g).all() for g in gradients)
            total = sum(float(g.abs().sum()) for g in gradients)
            assert total > 0
            gradient_steps.append({"step_before_update": state.global_step, "gradient_abs_sum": total})
    instance.add_callback(CaptureGradients())
    result = instance.train()
    assert result.global_step == 2 and torch.isfinite(torch.tensor(result.training_loss))
    assert len(gradient_steps) == 2
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p.detach())]
    assert changed
    assert_cpu(model)
    encoded = model.encode(["solar electricity", "water"], convert_to_tensor=True,
                           normalize_embeddings=True, show_progress_bar=False, device="cpu")
    assert encoded.device.type == "cpu" and torch.isfinite(encoded).all()
    model.save(str(args.output / "trained"))
    loaded = SentenceTransformer(str(args.output / "trained"), device="cpu", local_files_only=True,
                                 model_kwargs={"attn_implementation": "eager"})
    attention = [model[0].auto_model.config._attn_implementation,
                 loaded[0].auto_model.config._attn_implementation]
    assert attention == ["eager", "eager"]
    actual = loaded.encode(["solar electricity", "water"], convert_to_tensor=True,
                          normalize_embeddings=True, show_progress_bar=False, device="cpu")
    assert torch.allclose(encoded, actual, atol=1e-6, rtol=1e-6)
    return {"fixture": identity, "pipeline": source, "steps": result.global_step,
            "training_loss": result.training_loss, "changed_parameters": changed,
            "finite_cpu_gradients": gradient_steps,
            "attention_implementations": attention,
            "checkpointing_helper_toggled": True, "history": instance.state.log_history,
            "reload_max_difference": float((encoded-actual).abs().max())}


def accelerate(args):
    torch = cpu_setup()
    from accelerate import Accelerator
    accelerator = Accelerator(cpu=True)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)
    before = model.weight.detach().clone()
    loss = model(torch.tensor([[1., 2., 3.]])).square().mean()
    accelerator.backward(loss)
    assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()
    optimizer.step()
    assert accelerator.device.type == "cpu" and not torch.equal(before, model.weight)
    assert_cpu(model)
    return {"device": str(accelerator.device), "distributed_type": str(accelerator.distributed_type),
            "loss": float(loss.detach()), "gradient_sum": float(model.weight.grad.abs().sum())}


def deepspeed(args):
    cpu_setup()
    import deepspeed
    from deepspeed.accelerator import get_accelerator
    from deepspeed.runtime.config import DeepSpeedConfig
    accelerator = get_accelerator()
    assert accelerator.device_name() == "cpu"
    config = DeepSpeedConfig({"train_batch_size": 2, "train_micro_batch_size_per_gpu": 2,
                             "gradient_accumulation_steps": 1, "zero_optimization": {"stage": 0},
                             "fp16": {"enabled": False}, "bf16": {"enabled": False}})
    return {"version": deepspeed.__version__, "accelerator": accelerator.device_name(),
            "communication_backend": accelerator.communication_backend_name(),
            "train_batch_size": config.train_batch_size,
            "limit": "Import/config only; no engine, native optimizer, collectives or kernels"}


def modelscope(args):
    from modelscope import snapshot_download
    signature = inspect.signature(snapshot_download)
    # Bind the repository consumer's exact keyword shape; do not call downloader.
    signature.bind(model_id="authored-fixture/not-a-real-model", local_dir=str(args.output / "unused"))
    path = args.source / "scripts/tools/download_model.py"
    return {"signature": str(signature), "module": snapshot_download.__module__,
            "consumer_sha256": digest(path), "limit": "Real import/signature binding; no remote download"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--build-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    attempts, expected_denials, queries = [], [], []
    guard_network(attempts, expected_denials, queries)
    started = time.monotonic()
    receipt = {"stage": args.stage, "fixture_sha256": digest(__file__),
               "kind": "random-synthetic-cpu-api-proof", "accepts_original51": False}
    code = 0
    try:
        if args.stage.startswith("import-"):
            name = args.stage.removeprefix("import-")
            module = importlib.import_module(PACKAGES[name])
            details = {"distribution": name, "version": metadata.version(name),
                       "module_file": module.__file__}
        elif args.stage == "network-guard":
            import socket
            try:
                socket.getaddrinfo("example.invalid", 443)
            except PermissionError:
                details = {"python_audit_denial_observed": True}
            else:
                raise AssertionError("Network guard did not deny DNS")
        elif args.stage.startswith("ordered-"):
            # Preserve the ordinary-order failures separately. This is a
            # diagnostic import order, not a production-helper/library patch.
            import deepspeed as actual_deepspeed
            details = globals()[args.stage.removeprefix("ordered-")](args)
            details["diagnostic_preimport"] = {"package": "deepspeed", "version": actual_deepspeed.__version__,
                                              "does_not_fix_repository_entrypoint": True}
        else:
            details = globals()[args.stage.replace("-", "_")](args)
        if attempts and args.stage != "network-guard":
            raise AssertionError("Runtime attempted forbidden network/process operation")
        receipt.update(status="passed", details=details)
    except BaseException as error:
        receipt.update(status="failed", error={"type": type(error).__name__, "message": str(error),
                       "traceback": traceback.format_exc()})
        traceback.print_exc()
        code = 1
    receipt.update(seconds=time.monotonic()-started,
                   actual_self_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   denied_operations=attempts,
                   denied_capability_probes=expected_denials, read_only_system_queries=queries)
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

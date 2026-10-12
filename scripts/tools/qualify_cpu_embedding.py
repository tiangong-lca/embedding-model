#!/usr/bin/env python3
"""Offline resident CPU qualification; all model/data paths must already be local.

No endpoint requests, package installation, artifact download, or production change.
The parent process/container owns CPU and memory isolation; cgroup facts are captured.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def read_jsonl(path):
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f'Empty dataset: {path}')
    return rows


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p / 100
    lo = math.floor(index)
    return ordered[lo] + (ordered[math.ceil(index)] - ordered[lo]) * (index - lo)


def latency_summary(seconds):
    return {'completed': len(seconds), 'p50_ms': percentile(seconds, 50) * 1000,
            'p95_ms': percentile(seconds, 95) * 1000,
            'p99_ms': percentile(seconds, 99) * 1000,
            'max_ms': max(seconds) * 1000,
            'p99_estimation': 'descriptive sample percentile; not a production tail SLO'}


def isolation_facts():
    facts = {'platform': platform.platform(), 'machine': platform.machine(),
             'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None}
    for name in ['cpu.max', 'cpuset.cpus.effective', 'memory.max', 'memory.swap.max',
                 'cpu.stat', 'memory.peak', 'memory.events']:
        path = Path('/sys/fs/cgroup') / name
        facts[name] = path.read_text().strip() if path.exists() else None
    facts['cpu_model'] = next((line.partition(':')[2].strip() for line in
                              Path('/proc/cpuinfo').read_text().splitlines()
                              if line.startswith('model name')), None) if Path('/proc/cpuinfo').exists() else None
    return facts


class ResidentEncoder:
    def __init__(self, args):
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.np, self.torch = np, torch
        if torch.cuda.is_available():
            raise RuntimeError('CPU qualification must not expose a GPU')
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        self.path = Path(args.model).resolve()
        self.max_length = args.max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        if self.tokenizer.padding_side != 'right':
            raise ValueError('Deployed sum(mask)-1 pooling requires right padding; do not silently change semantics')
        started = time.perf_counter()
        self.model = AutoModel.from_pretrained(self.path, local_files_only=True,
                        torch_dtype={'fp32': torch.float32, 'bf16': torch.bfloat16}[args.dtype]).eval()
        self.load_seconds = time.perf_counter() - started
        self.meta = {'model_directory': str(self.path), 'dtype': args.dtype,
                     'max_length': self.max_length, 'pooling': 'last valid token via sum(mask)-1',
                     'normalization': 'L2', 'prompts': 'none; matches deployed custom handler',
                     'padding_side': self.tokenizer.padding_side,
                     'attention_implementation': self.model.config._attn_implementation,
                     'parameter_dtype': str(next(self.model.parameters()).dtype),
                     'intraop_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                     'startup_seconds_excluded_from_latency': self.load_seconds,
                     'files_sha256': {str(p.relative_to(self.path)): sha256(p) for p in sorted(self.path.rglob('*'))
                                      if p.is_file()},
                     'runtime': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'numpy', 'tokenizers', 'safetensors']},
                     'isolation': isolation_facts()}
        if args.artifact_manifest:
            identity = json.loads(Path(args.artifact_manifest).read_text())
            expected = {item['path']: item['sha256'] for item in identity['members']}
            if expected != self.meta['files_sha256']:
                raise ValueError('Extracted model content does not match immutable artifact manifest')
            self.meta['artifact_archive_sha256'] = identity['archive_sha256']
        if args.verify_handler:
            self.verify_handler()

    def encode(self, texts):
        torch = self.torch
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=self.max_length, return_tensors='pt')
        with torch.no_grad():
            out = self.model(**inputs)
            idx = inputs['attention_mask'].sum(1).clamp(min=1) - 1
            emb = out.last_hidden_state[torch.arange(len(texts)), idx]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        # Production response calls .numpy().tolist(); bf16 candidate converts to float32.
        return emb.float().cpu().numpy()

    def verify_handler(self):
        # Execution is explicit and restricted to the previously reviewed,
        # SHA-bound artifact handler. It reuses the resident model/tokenizer.
        import importlib.util
        import numpy as np
        spec = importlib.util.spec_from_file_location('reviewed_production_handler', self.path / 'code/inference.py')
        handler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(handler)
        texts = ['wood residue gasification heat recovery',
                 'electricity generation Switzerland photovoltaic silicon lifecycle emissions ' * 20]
        checks = []
        for batch in [[texts[0]], [texts[1]], texts]:
            expected = np.asarray(handler.predict_fn({'inputs': batch}, (self.model, self.tokenizer)), dtype=np.float32)
            actual = self.encode(batch)
            if not np.array_equal(actual, expected):
                raise ValueError('Resident encoder differs from reviewed production handler')
            checks.append({'batch_size': len(batch), 'input_tokens': [self.length(t) for t in batch],
                           'max_abs_difference': float(np.max(np.abs(actual - expected)))})
        self.meta['reviewed_handler_parity'] = checks

    def length(self, text):
        return len(self.tokenizer(text, truncation=False)['input_ids'])

    def sized_text(self, count):
        # Decoder-language LCA payload; truncate with the exact tokenizer then verify.
        words = ' electricity generation renewable photovoltaic silicon process lifecycle emissions carbon water geography technology '
        ids = self.tokenizer(words * count, add_special_tokens=False)['input_ids']
        # Some production tokenizers append EOS even for single strings. Count
        # actual tokenizer output rather than assuming decode/encode is identity.
        for cut in range(max(1, count - 8), count + 9):
            text = self.tokenizer.decode(ids[:cut], skip_special_tokens=False)
            if self.length(text) == count:
                return text
        raise ValueError(f'Cannot construct exact {count}-token payload')



def benchmark(args):
    model = ResidentEncoder(args)
    result = {'kind': 'warm_cpu_benchmark', 'manifest': model.meta,
              'mode': 'closed-loop concurrent single-input handler calls; no HTTP/network latency',
              'cold_start_acceptable': False, 'cells': []}
    path = Path(args.output) / 'benchmark.json'
    for length in args.lengths:
        text = model.sized_text(length)
        for _ in range(args.warmup):
            model.encode([text])
        for concurrency in args.concurrency:
            # Warm each participating inference thread before timed requests.
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                list(executor.map(lambda _: model.encode([text]), range(concurrency)))
                records = []
                total_started = time.perf_counter()
                def worker(worker_id):
                    local = []
                    for index in range(worker_id, args.requests, concurrency):
                        t = time.perf_counter()
                        try:
                            vector = model.encode([text])
                            serialized = json.dumps({'embeddings': vector.tolist(), 'normalized': True})
                            valid = bool(model.np.isfinite(vector).all()) and vector.shape == (1, 1024)
                            local.append({'index': index, 'seconds': time.perf_counter() - t,
                                          'valid': valid, 'response_bytes': len(serialized), 'error': None})
                        except Exception as error:
                            local.append({'index': index, 'seconds': time.perf_counter() - t,
                                          'valid': False, 'error': f'{type(error).__name__}: {error}'})
                    return local
                for local in executor.map(worker, range(concurrency)):
                    records.extend(local)
            elapsed = time.perf_counter() - total_started
            good = [r['seconds'] for r in records if r['valid']]
            cell = {'tokens': length, 'encoded_tokens': min(length, model.max_length),
                    'concurrency': concurrency, 'warmup_calls_excluded': args.warmup + concurrency,
                    'requested': args.requests, 'errors': len(records) - len(good),
                    'wall_seconds': elapsed, 'requests_per_second': len(good) / elapsed,
                    'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                    'raw': sorted(records, key=lambda r: r['index']), 'isolation_end': isolation_facts()}
            if good:
                cell.update(latency_summary(good))
            result['cells'].append(cell)
            save_json(path, result)
            print(json.dumps({k: v for k, v in cell.items() if k not in ['raw', 'isolation_end']}), flush=True)
    return result


def dataset(args):
    base = Path(args.data)
    queries, corpus, labels = [read_jsonl(base / name) for name in
                               ['test_queries.jsonl', 'corpus.jsonl', 'test_qrels.jsonl']]
    qids, cids = [[str(x['id']) for x in rows] for rows in [queries, corpus]]
    if len(set(qids)) != len(qids) or len(set(cids)) != len(cids):
        raise ValueError('Duplicate query/corpus IDs')
    qrels = {}
    for row in labels:
        qid, cid = str(row['qid']), str(row['docid'])
        if qid not in set(qids) or cid not in set(cids):
            raise ValueError('Unresolved qrel')
        qrels.setdefault(qid, {})[cid] = int(row.get('relevance', 1))
    if set(qrels) != set(qids):
        raise ValueError('Every evaluated query must have relevance judgments')
    train_path = base / 'training.json'
    train_queries = {x['query'] for x in read_jsonl(train_path)} if train_path.exists() else None
    seen = set()
    clean = []
    for row in queries:
        if row['text'] not in seen and train_queries is not None and row['text'] not in train_queries:
            clean.append(str(row['id']))
        seen.add(row['text'])
    meta = {'file_sha256': {p.name: sha256(p) for p in base.iterdir() if p.name in
                           ['test_queries.jsonl', 'corpus.jsonl', 'test_qrels.jsonl', 'training.json']},
            'query_count': len(queries), 'corpus_count': len(corpus), 'qrel_rows': len(labels),
            'unique_query_texts': len(seen), 'unique_query_text_overlap_train':
            len(seen & train_queries) if train_queries is not None else None,
            'deduplicated_train_disjoint_query_ids': clean,
            'limitations': ['historical study regression; corpus documents are training-related',
                            'one-positive labels do not establish exhaustive negatives',
                            'no unseen-document or production multilingual generalization proof']}
    return queries, corpus, qrels, meta


def quality(args):
    model = ResidentEncoder(args)
    import numpy as np
    queries, corpus, qrels, data_meta = dataset(args)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    vectors = {}
    for name, rows in [('queries', queries), ('corpus', corpus)]:
        lengths = [model.length(x['text'] if isinstance(x['text'], str) else ' '.join(x['text'])) for x in rows]
        model.encode([rows[0]['text']])
        arr = np.empty((len(rows), 1024), dtype=np.float32)
        start = time.perf_counter()
        order = sorted(range(len(rows)), key=lambda index: lengths[index], reverse=True)
        for i in range(0, len(rows), args.batch_size):
            indices = order[i:i + args.batch_size]
            texts = [rows[index]['text'] if isinstance(rows[index]['text'], str) else ' '.join(rows[index]['text']) for index in indices]
            arr[indices] = model.encode(texts)
            if i % (args.batch_size * 25) == 0:
                print(json.dumps({'encoding': name, 'completed': min(i + len(texts), len(rows)),
                                  'total': len(rows), 'elapsed': time.perf_counter() - start}), flush=True)
        if not np.isfinite(arr).all():
            raise ValueError('Non-finite embeddings')
        np.save(out / f'{name}_embeddings.npy', arr)
        save_json(out / f'{name}_ids.json', [str(x['id']) for x in rows])
        vectors[name] = arr
        data_meta[name + '_encoding'] = {'seconds': time.perf_counter() - start,
                                         'batch_size': args.batch_size, 'length_bucketing': 'descending measured tokens; original ID order restored',
                                         'tokens_p50': percentile(lengths, 50),
                                         'tokens_p95': percentile(lengths, 95),
                                         'tokens_max': max(lengths),
                                         'truncated_count': sum(x > model.max_length for x in lengths)}
    meta = {'kind': 'cpu_quality_vectors', 'manifest': model.meta, 'dataset': data_meta,
            'metrics': retrieval_metrics(vectors['queries'], vectors['corpus'],
                                         [str(x['id']) for x in queries], [str(x['id']) for x in corpus], qrels),
            'isolation_end': isolation_facts()}
    save_json(out / 'quality.json', meta)
    print(json.dumps(meta['metrics']), flush=True)


def retrieval_metrics(query, corpus, qids, cids, qrels, subset=None):
    import numpy as np
    # Exact search, normalized cosine with stable document-index tie handling.
    query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    corpus = corpus / np.maximum(np.linalg.norm(corpus, axis=1, keepdims=True), 1e-12)
    rankings = np.argsort(-(query @ corpus.T), axis=1, kind='stable')[:, :min(100, len(cids))]
    per_query = {}
    for i, qid in enumerate(qids):
        if subset is not None and qid not in subset:
            continue
        rels = qrels[qid]
        relevant = sum(r > 0 for r in rels.values())
        if relevant == 0:
            raise ValueError(f'No positive qrel for {qid}')
        ranked_rel = [rels.get(cids[x], 0) for x in rankings[i]]
        ideal = sorted(rels.values(), reverse=True)
        scores = {}
        for k in [1, 5, 10, 50, 100]:
            rel = ranked_rel[:k]
            ideal_dcg = sum((2 ** r - 1) / math.log2(j + 2) for j, r in enumerate(ideal[:k]))
            dcg = sum((2 ** r - 1) / math.log2(j + 2) for j, r in enumerate(rel))
            first = next((j + 1 for j, r in enumerate(rel) if r > 0), None)
            scores[f'Recall@{k}'] = sum(r > 0 for r in rel) / relevant
            scores[f'NDCG@{k}'] = dcg / ideal_dcg
            scores[f'MRR@{k}'] = 1 / first if first else 0
        per_query[qid] = scores
    if not per_query:
        raise ValueError('No evaluable queries in subset')
    return {'query_count': len(per_query), 'mean': {key: statistics.mean(x[key] for x in per_query.values())
             for key in next(iter(per_query.values()))}, 'per_query': per_query}


def validate_comparison_identity(candidate_meta, reference_meta, data_meta, vector_ids, dataset_ids):
    c, r = candidate_meta['manifest'], reference_meta['manifest']
    if c['files_sha256'] != r['files_sha256']:
        raise ValueError('Candidate/reference model or tokenizer binary identity differs')
    if r['dtype'] != 'fp32':
        raise ValueError('Reference must be regenerated FP32')
    for key in ['max_length', 'pooling', 'normalization', 'prompts', 'padding_side']:
        if c[key] != r[key]:
            raise ValueError(f'Candidate/reference semantic setting differs: {key}')
    current = data_meta['file_sha256']
    if candidate_meta['dataset']['file_sha256'] != current or reference_meta['dataset']['file_sha256'] != current:
        raise ValueError('Candidate/reference/current dataset content hashes differ')
    if vector_ids != dataset_ids:
        raise ValueError('Cached vectors do not match current dataset ID order')
    return {key: {'candidate': c['runtime'].get(key), 'reference': r['runtime'].get(key)}
            for key in set(c['runtime']) | set(r['runtime']) if c['runtime'].get(key) != r['runtime'].get(key)}


def compare(args):
    import numpy as np
    queries, corpus, qrels, data_meta = dataset(args)
    def cache(path):
        p = Path(path)
        items = (np.load(p / 'queries_embeddings.npy'), np.load(p / 'corpus_embeddings.npy'),
                json.loads((p / 'queries_ids.json').read_text()), json.loads((p / 'corpus_ids.json').read_text()))
        for arr, ids in [(items[0], items[2]), (items[1], items[3])]:
            if arr.shape != (len(ids), 1024) or not np.isfinite(arr).all():
                raise ValueError('Invalid embedding shape or non-finite vectors')
        return items
    candidate, reference = cache(args.candidate), cache(args.reference)
    provenance = []
    for path in [args.candidate, args.reference]:
        meta_path = Path(path) / 'quality.json'
        if not meta_path.exists():
            raise ValueError('Unbound legacy cache: require regenerated quality.json model manifest')
        provenance.append(json.loads(meta_path.read_text()))
    if candidate[2:] != reference[2:]:
        raise ValueError('Candidate/reference query or corpus IDs are not identically ordered')
    qids, cids = candidate[2:]
    runtime_differences = validate_comparison_identity(provenance[0], provenance[1], data_meta,
                              candidate[2:], ([str(x['id']) for x in queries], [str(x['id']) for x in corpus]))
    if set(qids) != set(qrels):
        raise ValueError('Vector query IDs and qrels differ')
    result = {'kind': 'cpu_quality_comparison', 'candidate': args.candidate, 'reference': args.reference,
              'dataset': data_meta, 'vectors': {}, 'rankings': {}, 'metrics': {},
              'provenance': {'reference_kind': 'regenerated offline same-artifact CPU FP32; not live GPU parity',
                             'runtime_differences': runtime_differences,
                             'candidate_manifest': provenance[0]['manifest'],
                             'reference_manifest': provenance[1]['manifest']}}
    for label, cv, rv in [('queries', candidate[0], reference[0]), ('corpus', candidate[1], reference[1])]:
        if cv.shape != rv.shape or not np.isfinite(cv).all() or not np.isfinite(rv).all():
            raise ValueError('Invalid candidate/reference vectors')
        cosine = np.sum(cv * rv, axis=1) / np.maximum(np.linalg.norm(cv, axis=1) * np.linalg.norm(rv, axis=1), 1e-12)
        result['vectors'][label] = {'cosine_min': float(cosine.min()), 'cosine_p01': float(np.percentile(cosine, 1)),
            'cosine_median': float(np.median(cosine)), 'max_abs_difference': float(np.max(np.abs(cv - rv)))}
    modes = {'reference': (reference[0], reference[1]), 'candidate': (candidate[0], candidate[1]),
             'candidate_queries_reference_corpus': (candidate[0], reference[1])}
    ranks = {}
    for name, (q, c) in modes.items():
        # Inputs from qualification are normalized, but normalize explicitly for cache comparisons.
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
        ranks[name] = np.argsort(-(q @ c.T), axis=1, kind='stable')[:, :100]
        result['metrics'][name] = retrieval_metrics(q, c, qids, cids, qrels)
        clean = set(data_meta['deduplicated_train_disjoint_query_ids'])
        if clean:
            result['metrics'][name + '_dedup_train_disjoint'] = retrieval_metrics(q, c, qids, cids, qrels, clean)
    for name in ['candidate', 'candidate_queries_reference_corpus']:
        result['rankings'][name] = {'top1_agreement': float(np.mean(ranks[name][:, 0] == ranks['reference'][:, 0])),
            'top10_overlap': float(np.mean([len(set(a[:10]) & set(b[:10])) / 10
                 for a, b in zip(ranks[name], ranks['reference'])]))}
        rng = np.random.default_rng(42)
        deltas = {}
        for key in ['Recall@10', 'NDCG@10', 'MRR@10', 'Recall@100']:
            diff = np.array([result['metrics'][name]['per_query'][qid][key] -
                             result['metrics']['reference']['per_query'][qid][key] for qid in qids])
            means = [float(rng.choice(diff, size=len(diff), replace=True).mean()) for _ in range(2000)]
            deltas[key] = {'mean': float(diff.mean()), 'paired_bootstrap_95ci': [percentile(means, 2.5), percentile(means, 97.5)],
                            'queries_worse': int((diff < 0).sum()), 'queries_better': int((diff > 0).sum())}
        result['rankings'][name]['metric_delta'] = deltas
    save_json(Path(args.output) / 'comparison.json', result)
    print(json.dumps({'vectors': result['vectors'], 'rankings': result['rankings'],
                      'metrics': {k: v['mean'] for k, v in result['metrics'].items()}}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['benchmark', 'quality', 'compare'])
    p.add_argument('--verify-handler', action='store_true'); p.add_argument('--artifact-manifest'); p.add_argument('--model'); p.add_argument('--data'); p.add_argument('--candidate'); p.add_argument('--reference')
    p.add_argument('--output', required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--dtype', choices=['fp32', 'bf16'], default='fp32')
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lengths', type=int, nargs='+', default=[32, 128, 512, 1024])
    p.add_argument('--concurrency', type=int, nargs='+', default=[1, 3, 6])
    p.add_argument('--requests', type=int, default=200)
    p.add_argument('--warmup', type=int, default=3)
    a = p.parse_args()
    if a.threads < 1 or a.batch_size < 1 or a.requests < 1 or a.warmup < 1 or any(x < 1 for x in a.lengths + a.concurrency):
        p.error('Counts/lengths/concurrency must be positive')
    if a.verify_handler and (not a.artifact_manifest or a.dtype != 'fp32'):
        p.error('--verify-handler requires reviewed --artifact-manifest and FP32')
    if a.mode != 'compare' and not a.model:
        p.error('--model is required')
    if a.mode != 'benchmark' and not a.data:
        p.error('--data is required')
    if a.mode == 'compare' and (not a.reference or not a.candidate):
        p.error('compare requires --candidate and --reference')
    {'benchmark': benchmark, 'quality': quality, 'compare': compare}[a.mode](a)


if __name__ == '__main__':
    main()

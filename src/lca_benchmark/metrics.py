"""LCA fixed-five-reference protocol; not conventional graded-qrels nDCG."""
import math
from .releases import require, rows, unique

PROTOCOL = "lca-fixed-five-reference-v1"
METRICS = ("coverage_at_5", "coverage_at_10", "ndcg_at_5", "ndcg_at_10")


def validate_references(queries, documents, references):
    require(bool(queries) and len(documents) >= 10, "Empty queries or insufficient corpus")
    weights = {q: {} for q in queries}
    for row in references:
        q, d, p = row["query_id"], row["document_id"], row["relevance"]
        require(q in queries and d in documents, "Unknown reference ID")
        require(d not in weights[q] and type(p) in (float, int) and math.isfinite(p) and 0 <= p <= 1,
                "Duplicate or invalid reference")
        weights[q][d] = p
    require(all(abs(sum(v.values()) - 5) <= 1e-9 for v in weights.values()), "Reference mass must equal five")
    return weights


def validate_rankings(queries, documents, rankings):
    ranks = unique(rankings, "query_id")
    require(ranks.keys() == queries.keys(), "Exactly one ranking per query required")
    for row in ranks.values():
        ids = row.get("document_ids")
        require(isinstance(ids, list) and len(ids) == 10 and all(isinstance(i, str) for i in ids), "Ten document IDs required")
        require(len(set(ids)) == 10 and set(ids) <= documents.keys(), "Duplicate/unknown ranked document")
    return ranks


def rank_scores(document_ids, scores):
    require(len(document_ids) == len(scores) and len(set(document_ids)) == len(document_ids), "Invalid score dimensions/IDs")
    require(len(scores) >= 10 and all(math.isfinite(s) for s in scores), "Invalid retrieval scores")
    return [d for d, _ in sorted(zip(document_ids, scores), key=lambda pair: (-pair[1], pair[0]))[:10]]


def evaluate(root, rankings):
    queries = unique(rows(root / "queries.jsonl"), "query_id")
    documents = unique(rows(root / "corpus.jsonl"), "document_id")
    weights = validate_references(queries, documents, rows(root / "qrels.jsonl"))
    ranks = validate_rankings(queries, documents, rankings)
    per_query = []
    for qid in sorted(queries):
        ids, probs = ranks[qid]["document_ids"], weights[qid]
        entry = {"query_id": qid, "source_group": queries[qid].get("source_group")}
        for k in (5, 10):
            entry[f"coverage_at_{k}"] = sum(probs.get(d, 0) for d in ids[:k]) / 5
            entry[f"ndcg_at_{k}"] = sum(probs.get(d, 0) / math.log2(i + 1) for i, d in enumerate(ids[:k], 1)) / sum(1 / math.log2(i + 1) for i in range(1, 6))
            entry[f"unjudged_at_{k}"] = sum(d not in probs for d in ids[:k])
        per_query.append(entry)
    return {"protocol": PROTOCOL, "queries": len(queries),
            "metrics": {m: sum(r[m] for r in per_query) / len(queries) for m in METRICS},
            "unjudged_returns": {str(k): sum(r[f"unjudged_at_{k}"] for r in per_query) for k in (5, 10)},
            "per_query": per_query}

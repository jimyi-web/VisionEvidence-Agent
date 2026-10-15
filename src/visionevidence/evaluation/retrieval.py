"""Explicit qrels, macro Recall@K and MRR@K; relevance never affects ranking."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from visionevidence.retrieval.bm25 import BM25Index, tokenize, validate_k
from visionevidence.retrieval.corpus import read_jsonl


@dataclass(frozen=True)
class QueryJudgment:
    query_id: str
    query: str
    relevant_ids: frozenset[str]

    def __post_init__(self) -> None:
        for name in ("query_id", "query"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        if not tokenize(self.query):
            raise ValueError("query must contain at least one English letter or digit")
        if not isinstance(self.relevant_ids, frozenset):
            raise ValueError("relevant_ids must be a frozenset of non-empty strings")
        if any(not isinstance(item, str) or not item.strip() for item in self.relevant_ids):
            raise ValueError("relevant_ids must contain non-empty strings")
        object.__setattr__(self, "relevant_ids", frozenset(item.strip() for item in self.relevant_ids))


def load_qrels(path: str | Path) -> list[QueryJudgment]:
    queries: list[QueryJudgment] = []
    seen: set[str] = set()
    for line_number, record in read_jsonl(path):
        try:
            if record.keys() != {"query_id", "query", "relevant_ids"}:
                raise ValueError("expected exactly query_id, query, relevant_ids")
            relevant = record["relevant_ids"]
            if not isinstance(relevant, list) or any(
                not isinstance(item, str) or not item.strip() for item in relevant
            ):
                raise ValueError("relevant_ids must be a list of non-empty strings")
            query = QueryJudgment(record["query_id"], record["query"], frozenset(relevant))
            if query.query_id in seen:
                raise ValueError(f"duplicate query_id: {query.query_id}")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid qrels: {exc}") from exc
        seen.add(query.query_id)
        queries.append(query)
    if not queries:
        raise ValueError(f"{path}: qrels must not be empty")
    return queries


def metrics_at_k(ranked_ids: list[str], relevant_ids: frozenset[str], k: int) -> tuple[float, float]:
    """Return (recall, reciprocal rank). Empty qrels are excluded by the caller."""
    validate_k(k)
    if not relevant_ids:
        raise ValueError("metrics are undefined without relevant evidence annotations")
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("ranked evidence IDs must be unique")
    top = ranked_ids[:k]
    recall = len(set(top) & relevant_ids) / len(relevant_ids)
    reciprocal_rank = next((1 / rank for rank, item in enumerate(top, 1) if item in relevant_ids), 0.0)
    return recall, reciprocal_rank


def evaluate_retrieval(index: BM25Index, queries: list[QueryJudgment], k: int) -> dict[str, Any]:
    validate_k(k)
    if not queries:
        raise ValueError("qrels must not be empty")
    seen: set[str] = set()
    for query in queries:
        if query.query_id in seen:
            raise ValueError(f"duplicate query_id: {query.query_id}")
        seen.add(query.query_id)
        unknown = query.relevant_ids - index.document_ids
        if unknown:
            raise ValueError(f"{query.query_id}: qrels IDs absent from corpus: {sorted(unknown)}")
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for query in queries:
        if not query.relevant_ids:
            skipped.append(query.query_id)
            continue
        started = perf_counter()
        results = index.search(query.query, k)
        elapsed = perf_counter() - started
        ranked_ids = [item.document_id for item in results]
        recall, rr = metrics_at_k(ranked_ids, query.relevant_ids, k)
        rows.append({
            "query_id": query.query_id, "query": query.query,
            "relevant_ids": sorted(query.relevant_ids), "retrieved_ids": ranked_ids,
            "scores": [item.score for item in results],
            "recall_at_k": recall, "reciprocal_rank_at_k": rr,
            "retrieval_seconds": elapsed,
        })
    if not rows:
        raise ValueError("no evaluable queries: all queries have empty relevant_ids")
    return {
        "counts": {"documents": len(index.document_ids), "queries_total": len(queries),
                   "queries_evaluated": len(rows), "queries_skipped": len(skipped)},
        "metrics": {"recall_at_k": sum(row["recall_at_k"] for row in rows) / len(rows),
                    "mrr_at_k": sum(row["reciprocal_rank_at_k"] for row in rows) / len(rows)},
        "skipped_query_ids": skipped, "per_query": rows,
    }

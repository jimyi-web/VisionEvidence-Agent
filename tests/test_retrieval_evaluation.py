import json
from pathlib import Path

import pytest

from visionevidence.evaluation.retrieval import QueryJudgment, evaluate_retrieval, load_qrels, metrics_at_k
from visionevidence.retrieval import BM25Index
from visionevidence.schemas import Evidence


def test_hand_calculated_metrics() -> None:
    # Relevant {B,D}; Top-3 [A,B,C] => recall=1/2, reciprocal rank=1/2.
    assert metrics_at_k(["A", "B", "C", "D"], frozenset({"B", "D"}), 3) == (0.5, 0.5)
    assert metrics_at_k(["A", "B", "C", "D"], frozenset({"B", "D"}), 4) == (1.0, 0.5)
    assert metrics_at_k(["A", "B"], frozenset({"B", "D"}), 1) == (0.0, 0.0)
    assert metrics_at_k([], frozenset({"B"}), 3) == (0.0, 0.0)


def test_macro_average_skips_empty_qrels_and_does_not_rank_with_labels() -> None:
    index = BM25Index([Evidence(item, "synthetic", "rain") for item in ("A", "B", "C", "D")])
    queries = [QueryJudgment("Q1", "rain", frozenset({"B", "D"})),
               QueryJudgment("Q2", "rain", frozenset({"A"})),
               QueryJudgment("Q3", "rain", frozenset())]
    report = evaluate_retrieval(index, queries, 3)
    # Q1=(1/2,1/2), Q2=(1,1) => macro=(3/4,3/4), excluding Q3.
    assert report["metrics"] == {"recall_at_k": 0.75, "mrr_at_k": 0.75}
    assert report["counts"] == {"documents": 4, "queries_total": 3, "queries_evaluated": 2, "queries_skipped": 1}
    assert report["skipped_query_ids"] == ["Q3"]
    assert all(row["retrieved_ids"] == ["A", "B", "C"] for row in report["per_query"])
    assert evaluate_retrieval(index, [QueryJudgment("miss", "snow", frozenset({"A"}))], 3)["metrics"] == {"recall_at_k": 0.0, "mrr_at_k": 0.0}


@pytest.mark.parametrize("queries, message", [
    ([], "qrels must not be empty"),
    ([QueryJudgment("Q", "rain", frozenset())], "no evaluable queries"),
    ([QueryJudgment("Q", "rain", frozenset({"missing"}))], "absent from corpus"),
    ([QueryJudgment("Q", "rain", frozenset({"A"}))] * 2, "duplicate query_id"),
])
def test_invalid_evaluation(queries: list, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_retrieval(BM25Index([Evidence("A", "demo", "rain")]), queries, 3)


def test_metrics_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="undefined"):
        metrics_at_k([], frozenset(), 3)
    with pytest.raises(ValueError, match="unique"):
        metrics_at_k(["A", "A"], frozenset({"A"}), 3)
    with pytest.raises(ValueError, match="positive integer"):
        metrics_at_k(["A"], frozenset({"A"}), 0)


def test_load_qrels(tmp_path: Path) -> None:
    path = tmp_path / "qrels.jsonl"
    path.write_text(json.dumps({"query_id": "Q", "query": "rain", "relevant_ids": ["A", "A"]}), encoding="utf-8")
    assert load_qrels(path) == [QueryJudgment("Q", "rain", frozenset({"A"}))]


@pytest.mark.parametrize("content, message", [
    ("", "must not be empty"), ("oops", "invalid JSONL"), ("{}", "expected exactly"),
    ('{"query_id":"Q","query":"rain","relevant_ids":"A"}', "list"),
    ('{"query_id":"Q","query":"rain","relevant_ids":[1]}', "list"),
    ('{"query_id":"Q","query":"rain","relevant_ids":[""]}', "list"),
    ('{"query_id":"Q","query":" ","relevant_ids":[]}', "non-empty"),
    ('{"query_id":"Q","query":"!!!","relevant_ids":[]}', "English letter"),
    ('{"query_id":"Q","query":"rain","relevant_ids":[]}\n' * 2, "duplicate query_id"),
])
def test_invalid_qrels(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "qrels.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_qrels(path)

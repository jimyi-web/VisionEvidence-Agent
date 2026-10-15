import json
import math
from pathlib import Path

import pytest

from visionevidence.retrieval import BM25Index, evidence_to_dict, load_corpus, tokenize
from visionevidence.schemas import Evidence


def write_corpus(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "corpus.jsonl"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_utf8_and_schema_adapter(tmp_path: Path) -> None:
    row = {"evidence_id": "E1", "text": "Synthetic rain. 合成文本。", "source": "synthetic/demo", "metadata": {"语言": "en"}}
    documents = load_corpus(write_corpus(tmp_path, json.dumps(row, ensure_ascii=False)))
    assert documents == [Evidence("E1", row["source"], row["text"], metadata=row["metadata"])]
    assert evidence_to_dict(documents[0]) == {**row, "score": None}


@pytest.mark.parametrize("content, message", [
    ("{oops}", "invalid JSONL"), ("\n", "invalid JSONL"), ("[]", "JSON object"),
    ('{"evidence_id":"E1","text":"x"}', "missing fields"),
    ('{"evidence_id":"E1","text":"x","source":"demo","typo":1}', "unknown fields"),
    ('{"evidence_id":"E1","text":"  ","source":"demo"}', "text must not be empty"),
    ('{"evidence_id":" ","text":"x","source":"demo"}', "document_id must not be empty"),
    ('{"evidence_id":1,"text":"x","source":"demo"}', "document_id must be a string"),
    ('{"evidence_id":"E1","text":"x","source":""}', "source must not be empty"),
    ('{"evidence_id":"E1","text":"x","source":"demo","metadata":[]}', "metadata must be a dictionary"),
    ('{"evidence_id":"E1","text":"x","source":"demo","metadata":{"x":NaN}}', "non-finite"),
    ('{"evidence_id":"E1","text":"x","source":"demo","metadata":{"x":1e999}}', "invalid JSONL"),
    ('{"evidence_id":"E1","evidence_id":"E2","text":"x","source":"demo"}', "duplicate JSON key"),
])
def test_invalid_records_have_path_and_line(tmp_path: Path, content: str, message: str) -> None:
    path = write_corpus(tmp_path, content)
    with pytest.raises(ValueError, match=message) as exc:
        load_corpus(path)
    assert f"{path}:1:" in str(exc.value)


def test_corruption_after_valid_line_is_not_skipped(tmp_path: Path) -> None:
    path = write_corpus(tmp_path, '{"evidence_id":"E1","text":"x","source":"demo"}\ninvalid\n')
    with pytest.raises(ValueError, match=":2: invalid JSONL"):
        load_corpus(path)


def test_duplicate_id_after_normalization(tmp_path: Path) -> None:
    records = [{"evidence_id": item, "text": "x", "source": "demo"} for item in ("E1", " E1 ")]
    with pytest.raises(ValueError, match="duplicate evidence_id"):
        load_corpus(write_corpus(tmp_path, "\n".join(map(json.dumps, records))))


def test_empty_corpus(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        load_corpus(write_corpus(tmp_path, ""))
    with pytest.raises(ValueError, match="must not be empty"):
        BM25Index([])


def test_invalid_utf8_and_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    with pytest.raises(FileNotFoundError):
        load_corpus(path)
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="valid UTF-8"):
        load_corpus(path)


def test_tokenizer() -> None:
    assert tokenize("FLOOD, road-closures! 2026 don't 中文") == ["flood", "road", "closures", "2026", "don", "t"]


def test_bm25_score_against_hand_calculation() -> None:
    # N=3, df(apple)=2, average length=4/3; explicit formula checks TF and length effects.
    index = BM25Index([Evidence("B", "demo", "apple"), Evidence("A", "demo", "apple apple"), Evidence("C", "demo", "pear")])
    results = index.search("APPLE!", 10)
    assert [item.document_id for item in results] == ["A", "B"]
    idf = math.log(1 + 1.5 / 2.5)
    assert results[0].score == pytest.approx(idf * 5 / (2 + 1.5 * (0.25 + 0.75 * 2 / (4 / 3))))
    assert results[1].score == pytest.approx(idf * 2.5 / (1 + 1.5 * (0.25 + 0.75 / (4 / 3))))
    assert index.search("apple apple") == index.search("apple")
    assert len(index.search("apple", 1)) == 1
    assert index.search("unknown", 5) == []


def test_tie_breaks_and_input_order_independence() -> None:
    documents = [Evidence(item, "demo", "rain road") for item in ("E3", "E1", "E2")]
    first = BM25Index(documents).search("road rain", 2)
    second = BM25Index(reversed(documents)).search("rain road", 2)
    assert first == second
    assert [item.document_id for item in first] == ["E1", "E2"]


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_top_k(k: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        BM25Index([Evidence("E1", "demo", "rain")]).search("rain", k)


@pytest.mark.parametrize("query", ["", " \t", "!!!", "中文"])
def test_empty_or_unsupported_query(query: str) -> None:
    with pytest.raises(ValueError, match="English letter or digit"):
        BM25Index([Evidence("E1", "demo", "rain")]).search(query)


def test_corpus_without_supported_tokens() -> None:
    assert BM25Index([Evidence("E1", "demo", "中文。")]).search("rain") == []


def test_index_and_results_are_snapshots() -> None:
    document = Evidence("E1", "demo", "rain", metadata={"nested": [1]})
    index = BM25Index([document])
    document.text = "changed"
    result = index.search("rain")[0]
    result.metadata["nested"].append(2)
    result.text = "changed again"
    assert index.search("rain")[0].text == "rain"
    assert index.search("rain")[0].metadata == {"nested": [1]}


@pytest.mark.parametrize("options", [{"k1": 0}, {"k1": -1}, {"k1": float("nan")}, {"b": 1.1}, {"b": -0.1}, {"b": True}, {"b": "x"}])
def test_invalid_bm25_parameters(options: dict) -> None:
    with pytest.raises(ValueError):
        BM25Index([Evidence("E1", "demo", "rain")], **options)


def test_direct_index_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate evidence_id"):
        BM25Index([Evidence("E1", "demo", "rain"), Evidence("E1", "demo", "snow")])

"""Strict UTF-8 JSONL input and adapters for the existing Evidence contract."""

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

from visionevidence.schemas import Evidence


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def read_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Reject blank lines, malformed records and non-object JSON; never skip rows."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(
                        line, object_pairs_hook=_object, parse_constant=_invalid_constant
                    )
                    if not isinstance(record, dict):
                        raise ValueError("record must be a JSON object")
                    json.dumps(record, allow_nan=False)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
                yield line_number, record
    except UnicodeError as exc:
        raise ValueError(f"{path}: evidence/qrels file must be valid UTF-8: {exc}") from exc


def load_corpus(path: str | Path) -> list[Evidence]:
    """Load evidence_id/text/source/optional metadata; reject an empty corpus."""
    documents: list[Evidence] = []
    seen: set[str] = set()
    for line_number, record in read_jsonl(path):
        try:
            missing = {"evidence_id", "text", "source"} - record.keys()
            extra = record.keys() - {"evidence_id", "text", "source", "metadata"}
            if missing or extra:
                raise ValueError(f"missing fields: {sorted(missing)}; unknown fields: {sorted(extra)}")
            document = Evidence(
                document_id=record["evidence_id"], text=record["text"],
                source=record["source"], metadata=record.get("metadata", {}),
            )
            if document.document_id in seen:
                raise ValueError(f"duplicate evidence_id: {document.document_id}")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid evidence: {exc}") from exc
        seen.add(document.document_id)
        documents.append(document)
    if not documents:
        raise ValueError(f"{path}: evidence corpus must not be empty")
    return documents


def evidence_to_dict(document: Evidence) -> dict[str, Any]:
    """Use the external evidence_id name while preserving schema compatibility."""
    return {
        "evidence_id": document.document_id, "text": document.text,
        "source": document.source, "score": document.score, "metadata": document.metadata,
    }

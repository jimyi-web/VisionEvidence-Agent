"""Small, validated data contracts; no model or retrieval implementation."""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any


def _nonempty_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _validate_metadata(metadata: dict[str, Any]) -> None:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")
    if any(not isinstance(key, str) for key in metadata):
        raise TypeError("metadata keys must be strings")


@dataclass
class ClaimInput:
    """Text plus an optional existing local image file (contents not decoded)."""

    text: str
    image_path: str | Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = _nonempty_text(self.text, "text")
        _validate_metadata(self.metadata)
        if self.image_path is not None:
            if not isinstance(self.image_path, (str, Path)):
                raise TypeError("image_path must be a string or Path")
            if isinstance(self.image_path, str):
                _nonempty_text(self.image_path, "image_path")
            path = Path(self.image_path)
            if not path.is_file():
                raise ValueError(f"image_path must point to an existing file: {path}")
            self.image_path = path


@dataclass
class Evidence:
    """One evidence document; a retrieval score is not a truth probability."""

    document_id: str
    source: str
    text: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.document_id = _nonempty_text(self.document_id, "document_id")
        self.source = _nonempty_text(self.source, "source")
        self.text = _nonempty_text(self.text, "text")
        _validate_metadata(self.metadata)
        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
                raise TypeError("score must be a finite number or None")
            try:
                score = float(self.score)
            except OverflowError as exc:
                raise ValueError("score must be finite") from exc
            if not math.isfinite(score):
                raise ValueError("score must be finite")
            self.score = score


@dataclass
class VerificationResult:
    """Contract for future predictions; the T01 CLI never produces this object.

    Label vocabulary is dataset-specific. Evidence IDs refer to document_id.
    Callers must validate references against their evidence collection.
    """

    predicted_label: str
    evidence_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.predicted_label = _nonempty_text(self.predicted_label, "predicted_label")
        if not isinstance(self.evidence_ids, list):
            raise TypeError("evidence_ids must be a list of strings")
        self.evidence_ids = [
            _nonempty_text(evidence_id, "evidence_id") for evidence_id in self.evidence_ids
        ]
        _validate_metadata(self.metadata)

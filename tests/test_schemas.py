"""Synthetic fixtures validate contracts, not model accuracy."""

from pathlib import Path

import pytest

from visionevidence.schemas import ClaimInput, Evidence, VerificationResult


def test_text_input() -> None:
    claim = ClaimInput(text="  A test claim.  ")
    assert claim.text == "A test claim."
    assert claim.image_path is None


@pytest.mark.parametrize("as_string", [False, True])
def test_optional_image_path(tmp_path: Path, as_string: bool) -> None:
    image_path = tmp_path / "sample image.png"
    image_path.write_bytes(b"path-validation fixture; not a decoded image")
    claim = ClaimInput("A test claim.", str(image_path) if as_string else image_path)
    assert claim.image_path == image_path


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
def test_empty_text_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="text must not be empty"):
        ClaimInput(text)


def test_non_string_text_rejected() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        ClaimInput(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["missing", "directory", "empty"])
def test_invalid_image_path(tmp_path: Path, kind: str) -> None:
    path = {"missing": tmp_path / "missing.png", "directory": tmp_path, "empty": ""}[kind]
    with pytest.raises(ValueError, match="image_path"):
        ClaimInput("A test claim.", path)


def test_evidence_fields() -> None:
    evidence = Evidence("doc-1", "synthetic fixture", "Evidence text.", 1.5, {"split": "test"})
    assert evidence.document_id == "doc-1"
    assert evidence.source == "synthetic fixture"
    assert evidence.text == "Evidence text."
    assert evidence.score == 1.5
    assert evidence.metadata == {"split": "test"}
    assert Evidence("doc-2", "fixture", "Unscored text.").score is None


@pytest.mark.parametrize("field", ["document_id", "source", "text"])
def test_empty_evidence_fields_rejected(field: str) -> None:
    fields = {"document_id": "doc-1", "source": "fixture", "text": "Evidence text."}
    fields[field] = " "
    with pytest.raises(ValueError, match=field):
        Evidence(**fields)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf"), 10**400])
def test_nonfinite_evidence_scores_rejected(score: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        Evidence("doc-1", "fixture", "Evidence text.", score)


@pytest.mark.parametrize("score", [True, "0.5"])
def test_invalid_score_type_rejected(score: object) -> None:
    with pytest.raises(TypeError, match="score"):
        Evidence("doc-1", "fixture", "Evidence text.", score)  # type: ignore[arg-type]


def test_verification_result_contract() -> None:
    # A schema fixture, not an experimental or model prediction.
    result = VerificationResult("fixture_label", ["doc-1"], {"fixture": True})
    assert result.predicted_label == "fixture_label"
    assert result.evidence_ids == ["doc-1"]
    assert result.metadata == {"fixture": True}


def test_empty_prediction_rejected() -> None:
    with pytest.raises(ValueError, match="predicted_label"):
        VerificationResult(" ")


def test_empty_evidence_reference_rejected() -> None:
    with pytest.raises(ValueError, match="evidence_id"):
        VerificationResult("fixture_label", [""])


def test_evidence_references_must_be_list() -> None:
    with pytest.raises(TypeError, match="evidence_ids"):
        VerificationResult("fixture_label", "doc-1")  # type: ignore[arg-type]


@pytest.mark.parametrize("metadata", [[], {1: "value"}])
def test_invalid_metadata_rejected(metadata: object) -> None:
    with pytest.raises(TypeError, match="metadata"):
        ClaimInput("Claim.", metadata=metadata)  # type: ignore[arg-type]


def test_mutable_defaults_are_independent() -> None:
    first, second = ClaimInput("First."), ClaimInput("Second.")
    first.metadata["fixture"] = True
    assert second.metadata == {}
    first_evidence = Evidence("doc-1", "fixture", "Text.")
    second_evidence = Evidence("doc-2", "fixture", "Text.")
    first_evidence.metadata["fixture"] = True
    assert second_evidence.metadata == {}
    first_result, second_result = VerificationResult("fixture"), VerificationResult("fixture")
    first_result.evidence_ids.append("doc-1")
    first_result.metadata["fixture"] = True
    assert second_result.evidence_ids == []
    assert second_result.metadata == {}

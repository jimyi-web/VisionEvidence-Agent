"""Synthetic CPU-only contract/integration tests, not visual quality evaluation."""

import json
from pathlib import Path

import pytest

from visionevidence.observation import PROMPT, build_query, observe_and_retrieve, parse_observation
from visionevidence.retrieval import BM25Index
from visionevidence.schemas import Evidence
from visionevidence.vlm.schemas import VLMRequest, VLMResponse

IMAGE = Path(__file__).resolve().parents[1] / "examples/vlm_demo_fixture.png"
PAYLOAD = {"objects": ["flood road"], "visible_text": ["CLOSED"], "uncertainties": ["[MOCK] synthetic only"]}


class FixtureVLM:
    def __init__(self, description=None, *, real=False, metadata=None):
        self.description = json.dumps(PAYLOAD) if description is None else description
        self.real = real
        self.metadata = metadata or {}
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return VLMResponse(request.request_id, self.description, "fixture", "fixture" if self.real else "mock",
                           not self.real, 0, {"inference_performed": self.real, **self.metadata})


class CountingIndex(BM25Index):
    def __init__(self):
        super().__init__([Evidence("B", "synthetic/demo", "flood road closed", metadata={"synthetic": True}),
                          Evidence("A", "synthetic/demo", "flood road closed"),
                          Evidence("C", "synthetic/demo", "synthetic only")])
        self.calls = 0

    def search(self, *args, **kwargs):
        self.calls += 1
        return super().search(*args, **kwargs)


def request():
    return VLMRequest(IMAGE, PROMPT, request_id="synthetic-1")


def test_mock_integration_provenance_and_stable_order():
    model, index = FixtureVLM(), CountingIndex()
    result = observe_and_retrieve(request(), model, index)
    assert model.calls == index.calls == 1
    assert result["is_mock"] and not result["inference_performed"] and not result["verified"]
    assert result["query"] == "flood road closed"
    assert [x["document_id"] for x in result["evidence"]] == ["A", "B"]
    assert result["evidence"][1]["metadata"] == {"synthetic": True}
    assert result["query_sources"][1] == {"field": "visible_text", "index": 0, "terms": ["closed"]}
    assert len(result["trace"]["image_sha256"]) == 64
    assert result["vlm_response"]["description"] == json.dumps(PAYLOAD)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("text", [
    "{}", "[]", '```json\n{}\n```', '{"objects":[]',
    '{"objects":[],"objects":[],"visible_text":[],"uncertainties":[]}',
    json.dumps({**PAYLOAD, "truth": True}), json.dumps({**PAYLOAD, "objects": "road"}),
    json.dumps({**PAYLOAD, "objects": [" "]}), json.dumps({**PAYLOAD, "objects": [True]}),
    json.dumps({**PAYLOAD, "objects": ["a"] * 9}), json.dumps({**PAYLOAD, "objects": ["a" * 201]}),
    '{"objects":[NaN],"visible_text":[],"uncertainties":[]}', "x" * 8001,
])
def test_strict_observation(text):
    with pytest.raises(ValueError):
        parse_observation(text)


def test_chinese_only_is_explicit_failure_and_uncertainty_not_used():
    with pytest.raises(ValueError, match="no supported"):
        build_query(parse_observation(json.dumps({"objects": [], "visible_text": ["中文"],
                                                  "uncertainties": ["road"]})))


@pytest.mark.parametrize("metadata", [
    {"generated_tokens": 256, "max_new_tokens": 256}, {},
    {"generated_tokens": True, "max_new_tokens": 256}, {"inference_performed": False},
])
def test_bad_real_metadata_stops_before_search(metadata):
    index = CountingIndex()
    with pytest.raises(ValueError):
        observe_and_retrieve(request(), FixtureVLM(real=True, metadata=metadata), index)
    assert index.calls == 0


def test_invalid_json_stops_before_search():
    index = CountingIndex()
    with pytest.raises(ValueError):
        observe_and_retrieve(request(), FixtureVLM("[MOCK] truncated {"), index)
    assert index.calls == 0


def test_empty_hits_are_success():
    payload = {**PAYLOAD, "objects": ["unmatched"], "visible_text": []}
    result = observe_and_retrieve(request(), FixtureVLM(json.dumps(payload)), CountingIndex())
    assert result["status"] == "completed" and result["evidence"] == []


def test_bad_request_fails_before_generation():
    model = FixtureVLM()
    with pytest.raises(ValueError):
        observe_and_retrieve(VLMRequest(IMAGE, "wrong", request_id="id"), model, CountingIndex())
    assert model.calls == 0


def test_backend_failure_propagates_without_search():
    class Broken:
        def generate(self, request):
            raise RuntimeError("fixture backend failure")
    index = CountingIndex()
    with pytest.raises(RuntimeError, match="fixture backend failure"):
        observe_and_retrieve(request(), Broken(), index)
    assert index.calls == 0


def test_response_id_mismatch():
    class WrongID(FixtureVLM):
        def generate(self, request):
            return super().generate(VLMRequest(IMAGE, PROMPT, request_id="other"))
    index = CountingIndex()
    with pytest.raises(ValueError, match="request_id"):
        observe_and_retrieve(request(), WrongID(), index)
    assert index.calls == 0

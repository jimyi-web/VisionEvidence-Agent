"""Paired CPU proxy evaluation and durability, never real GPU quality tests."""

from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from visionevidence.batch import StructuredMock, run_batch
from visionevidence.caption_experiment import run_experiment
from visionevidence.datasets.flickr8k import adapt_flickr8k
from visionevidence.description import PROMPT as BASELINE_PROMPT, describe_and_retrieve
from visionevidence.observation import PROMPT
from visionevidence.retrieval.bm25 import BM25Index
from visionevidence.schemas import Evidence
from visionevidence.vlm.schemas import VLMRequest
from tests.test_flickr8k import make_raw


def bundle(tmp_path):
    raw = make_raw(tmp_path)
    dataset = tmp_path / "adapted"
    adapt_flickr8k(input_root=raw, output=dataset)
    return raw, dataset / "test"


class Spy(StructuredMock):
    def __init__(self):
        self.requests = []

    def generate(self, request):
        assert "HUMAN_REFERENCE_SECRET" not in request.prompt
        assert request.prompt in (PROMPT, BASELINE_PROMPT)
        self.requests.append(request)
        return super().generate(request)


def test_paired_controls_no_caption_leakage_and_cache(tmp_path):
    _, dataset = bundle(tmp_path)
    model, constructed = Spy(), []
    def factory():
        constructed.append(1)
        return model
    args = dict(dataset=dataset, output=tmp_path / "run", limit=1, backend_factory=factory)
    first = run_experiment(**args)
    assert len(model.requests) == 2 and len(constructed) == 1
    assert first["scale"] == {"candidate_caption_documents": 10, "evaluation_images": 1,
                               "retrieval_query_slots": 2, "real_vlm_image_calls": 0, "unique_real_vlm_images": 0}
    assert first["scores"]["baseline"]["denominator"] == first["scores"]["structured"]["denominator"] == 1
    args["limit"] = 2
    second = run_experiment(**args, resume=True)
    assert len(model.requests) == 4
    assert all(b["attempts"] == 2 for b in second["branches"].values())
    run_experiment(**{**args, "backend_factory": lambda: pytest.fail("cached models must not be constructed")}, resume=True)
    baseline = json.loads((args["output"] / "baseline/results-0001.jsonl").read_text())
    assert baseline["query"] == baseline["vlm_response"]["description"]


def test_structured_failure_zero_denominator_and_recovery(tmp_path):
    _, dataset = bundle(tmp_path)
    class Malformed(Spy):
        def generate(self, request):
            response = super().generate(request)
            return replace(response, description="[MOCK] natural language, not JSON") if request.prompt == PROMPT else response
    args = dict(dataset=dataset, output=tmp_path / "run", limit=2)
    first = run_experiment(**args, backend_factory=Malformed)
    assert first["status"] == "incomplete"
    assert first["scores"]["structured"]["denominator"] == 2
    assert first["scores"]["structured"]["metrics"]["10"]["hit_rate"] == 0
    second = run_experiment(**args, resume=True, max_attempts=2)
    assert second["status"] == "completed"
    assert second["branches"]["baseline"]["attempts"] == 2
    assert second["branches"]["structured"]["attempts"] == 4


def test_proxy_qrels_cannot_be_used_as_fact_relevance(tmp_path):
    _, dataset = bundle(tmp_path)
    with pytest.raises(ValueError, match="pair/classification"):
        run_batch(samples=dataset / "samples.jsonl", corpus=dataset / "corpus.jsonl",
                  qrels=dataset / "qrels.jsonl", output=tmp_path / "wrong-task", limit=1)


@pytest.mark.parametrize("tokens,limit", [(10, 10), (None, None)])
def test_baseline_real_truncation_or_missing_metadata_rejected(tmp_path, tokens, limit):
    raw, _ = bundle(tmp_path)
    request = VLMRequest(raw / "test/test_0.jpg", BASELINE_PROMPT, "s1")
    class FakeReal(StructuredMock):
        def generate(self, request):
            return replace(super().generate(request), backend="qwen3_vl", is_mock=False,
                           metadata={"inference_performed": True, "generated_tokens": tokens, "max_new_tokens": limit})
    index = BM25Index([Evidence("d1", "synthetic://test", "road sign")])
    with pytest.raises(ValueError, match="token"):
        describe_and_retrieve(request, FakeReal(), index)


def test_metrics_any_hit_and_five_target_recall_are_distinct(tmp_path):
    raw = make_raw(tmp_path, captions_prefix="road sign")
    adapted = tmp_path / "adapted"
    adapt_flickr8k(input_root=raw, output=adapted)
    result = run_experiment(dataset=adapted / "test", output=tmp_path / "run", limit=1)
    scores = result["scores"]["structured"]["metrics"]["1"]
    assert scores == {"hit_rate": 1.0, "caption_recall": 0.2, "mrr": 1.0}


def test_bundle_mutation_and_split_limit_rejected(tmp_path):
    _, dataset = bundle(tmp_path)
    with pytest.raises(ValueError, match="limit"):
        run_experiment(dataset=dataset, output=tmp_path / "too-large", limit=8000)
    with (dataset / "corpus.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="fingerprint"):
        run_experiment(dataset=dataset, output=tmp_path / "mutated", limit=1)


def test_missing_picture_repair_preserves_successful_other_branch_results(tmp_path):
    raw, dataset = bundle(tmp_path)
    target = raw / "test/test_1.jpg"
    image_bytes = target.read_bytes()
    target.unlink()
    args = dict(dataset=dataset, output=tmp_path / "run", limit=2)
    first = run_experiment(**args)
    assert all(b["sample_status"] == {"completed": 1, "failed": 1} for b in first["branches"].values())
    target.write_bytes(image_bytes)
    second = run_experiment(**args, resume=True, max_attempts=2)
    assert second["status"] == "completed"
    assert all(b["attempts"] == 3 for b in second["branches"].values())


def test_proxy_cli(tmp_path):
    _, dataset = bundle(tmp_path)
    result = subprocess.run([sys.executable, "-m", "visionevidence.caption_experiment", "--dataset", str(dataset),
                             "--output", str(tmp_path / "run"), "--mode", "mock", "--limit", "2"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["task"] == "image_to_caption_retrieval_proxy"


def test_caption_changes_never_change_generated_queries(tmp_path):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    datasets = []
    for directory, prefix in ((first_dir, "secret_reference_a"), (second_dir, "entirely_different_reference_b")):
        raw = make_raw(directory, captions_prefix=prefix)
        adapted = directory / "adapted"
        adapt_flickr8k(input_root=raw, output=adapted)
        run_experiment(dataset=adapted / "test", output=directory / "run", limit=1)
        datasets.append(directory / "run")
    for mode in ("baseline", "structured"):
        results = [json.loads((d / mode / "results-0001.jsonl").read_text()) for d in datasets]
        assert results[0]["query"] == results[1]["query"]
        assert results[0]["vlm_response"]["description"] == results[1]["vlm_response"]["description"]

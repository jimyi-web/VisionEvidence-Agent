"""Durability, provenance, leakage and scale checks using synthetic CPU fixtures."""

import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import replace

import pytest

from visionevidence.batch import StructuredMock, run_batch, run_lock
from visionevidence.vlm.factory import VLMConfig
from visionevidence.vlm.schemas import VLMResponse

ROOT = Path(__file__).resolve().parents[1]


def setup_run(tmp_path, count=3, missing=False):
    image = tmp_path / "image.png"
    if not missing:
        shutil.copyfile(ROOT / "examples/vlm_demo_fixture.png", image)
    samples = tmp_path / "samples.jsonl"
    with samples.open("w", encoding="utf-8") as stream:
        for i in range(count):
            stream.write(json.dumps({"sample_id": f"s{i}", "image_path": "image.png", "claim": "SECRET CAPTION",
                                     "dataset": "synthetic", "dataset_version": "1", "split": "smoke",
                                     "source": "synthetic://test", "pair_label": "untrusted label"}) + "\n")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"evidence_id": "d1", "text": "road sign", "source": "synthetic://doc"}) + "\n", encoding="utf-8")
    return {"samples": samples, "corpus": corpus, "output": tmp_path / "run"}


class CountingMock(StructuredMock):
    def __init__(self):
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        assert "SECRET" not in request.prompt
        assert "untrusted label" not in request.prompt
        return super().generate(request)


def test_staged_resume_uses_cache_and_keeps_old_snapshots(tmp_path):
    args = setup_run(tmp_path)
    model = CountingMock()
    first = run_batch(**args, limit=1, backend_factory=lambda: model)
    snapshot = (args["output"] / "results-0001.jsonl").read_bytes()
    assert first["scale"]["evaluation_queries"] == 1
    assert model.calls == 1
    result = run_batch(**args, limit=3, resume=True, batch_size=2, backend_factory=lambda: model)
    assert model.calls == 3
    assert result["scale"] == {"evidence_documents": 1, "evaluation_queries": 3, "successful_queries": 3,
                               "real_vlm_image_calls": 0, "real_vlm_responses": 0, "unique_real_vlm_images": 0}
    assert result["retrieval_metrics"] is None
    assert (args["output"] / "results-0001.jsonl").read_bytes() == snapshot
    run_batch(**args, resume=True, backend_factory=lambda: pytest.fail("should not construct model"))
    assert model.calls == 3


def test_missing_image_failure_then_repair_and_resume(tmp_path):
    args = setup_run(tmp_path, count=2, missing=True)
    first = run_batch(**args, max_attempts=1)
    assert first["status"] == "incomplete" and first["sample_status"] == {"failed": 2}
    assert first["error_counts"] == {"ValueError": 2}
    shutil.copyfile(ROOT / "examples/vlm_demo_fixture.png", tmp_path / "image.png")
    assert run_batch(**args, resume=True, max_attempts=1)["status"] == "incomplete"
    recovered = run_batch(**args, resume=True, max_attempts=2)
    assert recovered["status"] == "completed" and recovered["attempts"] == 4
    assert recovered["error_counts"] == {"ValueError": 2}


def test_transient_retry_and_raw_response_failure_journal(tmp_path):
    args = setup_run(tmp_path, count=1)
    class Flaky(CountingMock):
        def generate(self, request):
            response = super().generate(request)
            return replace(response, description="[MOCK] natural language is not JSON") if self.calls == 1 else response
    model = Flaky()
    report = run_batch(**args, backend_factory=lambda: model)
    assert report["status"] == "completed" and report["attempts"] == 2
    with sqlite3.connect(args["output"] / "state.sqlite3") as db:
        errors = db.execute("SELECT raw_response FROM attempts WHERE status='failed'").fetchall()
    assert json.loads(errors[0][0])["description"] == "[MOCK] natural language is not JSON"


def test_keyboard_interrupt_is_durable_and_recoverable(tmp_path):
    args = setup_run(tmp_path, count=3)
    class Interrupt(CountingMock):
        def generate(self, request):
            if self.calls == 1:
                raise KeyboardInterrupt()
            return super().generate(request)
    with pytest.raises(KeyboardInterrupt):
        run_batch(**args, backend_factory=Interrupt)
    result = run_batch(**args, resume=True)
    assert result["status"] == "completed"
    assert result["attempts"] == 4 and result["error_counts"] == {"Interrupted": 1}


@pytest.mark.parametrize("change", ["samples", "corpus", "config", "image"])
def test_resume_rejects_changed_inputs(tmp_path, change):
    args = setup_run(tmp_path, count=1)
    run_batch(**args)
    extra = {}
    if change in {"samples", "corpus"}:
        with args[change].open("a") as stream:
            stream.write("\n")
    elif change == "config":
        extra["config"] = VLMConfig(max_new_tokens=512)
    else:
        with (tmp_path / "image.png").open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError, match="changed|invalid"):
        run_batch(**args, resume=True, **extra)


def test_duplicate_ids_rejected_before_inference(tmp_path):
    args = setup_run(tmp_path, count=1)
    args["samples"].write_bytes(args["samples"].read_bytes() * 2)
    with pytest.raises(ValueError, match="duplicate"):
        run_batch(**args, backend_factory=lambda: pytest.fail("invalid dataset must not reach backend"))


def test_mode_provenance_mismatch_fails_closed(tmp_path):
    args = setup_run(tmp_path, count=1)
    class Wrong(StructuredMock):
        def generate(self, request):
            response = super().generate(request)
            return replace(response, backend="qwen3_vl", is_mock=False,
                           metadata={"inference_performed": True})
    result = run_batch(**args, max_attempts=1, backend_factory=Wrong)
    assert result["status"] == "incomplete" and result["scale"]["real_vlm_image_calls"] == 0
    assert result["error_counts"] == {"ValueError": 1}


def test_qrels_failure_zero_empty_excluded_pair_labels_unused(tmp_path):
    args = setup_run(tmp_path, count=3, missing=True)
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text("\n".join(json.dumps({"sample_id": f"s{i}", "relevant_ids": ids,
                                            "annotation_source": "synthetic fixture", "kind": "synthetic_relevance"})
                               for i, ids in enumerate((["d1"], [], ["d1"]))) + "\n", encoding="utf-8")
    result = run_batch(**args, qrels=qrels, max_attempts=1)
    assert result["retrieval_metrics"]["evaluated_queries"] == 2
    assert result["retrieval_metrics"]["recall_at_k"] == 0
    assert result["qrels_counts"] == {"evaluated": 2, "empty_excluded": 1, "unannotated": 0}
    shutil.copyfile(ROOT / "examples/vlm_demo_fixture.png", tmp_path / "image.png")
    fixed = run_batch(**args, qrels=qrels, max_attempts=2, resume=True)
    assert fixed["retrieval_metrics"]["recall_at_k"] == 1


@pytest.mark.parametrize("kind,ids", [("pair_label", ["d1"]), ("human_relevance", ["unknown"])])
def test_invalid_qrels_rejected(tmp_path, kind, ids):
    args = setup_run(tmp_path)
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(json.dumps({"sample_id": "s0", "relevant_ids": ids, "annotation_source": "fixture", "kind": kind}) + "\n")
    with pytest.raises(ValueError):
        run_batch(**args, qrels=qrels)


def test_ten_thousand_records_loaded_small_stage_only(tmp_path):
    args = setup_run(tmp_path, count=10000)
    model = CountingMock()
    result = run_batch(**args, limit=3, batch_size=2, backend_factory=lambda: model)
    with sqlite3.connect(args["output"] / "state.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM samples").fetchone()[0] == 10000
    assert model.calls == 3 and result["scale"]["evaluation_queries"] == 3


def test_no_overwrite_and_missing_resume_rejected(tmp_path):
    args = setup_run(tmp_path)
    with pytest.raises(ValueError, match="resume"):
        run_batch(**args, resume=True)
    run_batch(**args)
    with pytest.raises(FileExistsError):
        run_batch(**args)


def test_concurrent_worker_lock_rejected(tmp_path):
    lock = tmp_path / "lock"
    with run_lock(lock):
        with pytest.raises(ValueError, match="locked"):
            with run_lock(lock):
                pytest.fail("second worker acquired lock")
    with run_lock(lock):
        pass


def test_cli_mock_json_exit_codes_and_mode_guard(tmp_path):
    args = setup_run(tmp_path, count=2)
    cmd = [sys.executable, "-m", "visionevidence.batch", "--mode", "mock"]
    for key, value in args.items():
        cmd.extend([f"--{key}", str(value)])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["scale"]["evaluation_queries"] == 2
    result = subprocess.run(cmd + ["--resume", "--config", str(ROOT / "configs/t05_qwen3_vl.toml")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and "match" in result.stderr

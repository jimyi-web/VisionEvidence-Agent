"""Actual CPU CLI execution, synthetic inputs only."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_cli(tmp_path, text=None, *, extra=()):
    observation = ROOT / "examples/observation_demo.json"
    if text is not None:
        observation = tmp_path / "input.json"
        observation.write_text(text, encoding="utf-8")
    return subprocess.run([
        sys.executable, "-m", "visionevidence.cli", "observe-retrieve",
        "--observation", str(observation), "--image", str(ROOT / "examples/vlm_demo_fixture.png"),
        "--corpus", str(ROOT / "examples/evidence_demo.jsonl"),
        "--config", str(ROOT / "configs/retrieval_bm25.toml"), "--request-id", "test-1",
        "--dataset-name", "synthetic/demo", "--dataset-version", "1", "--split", "demo",
        "--output", str(tmp_path / "run"), *extra,
    ], capture_output=True, text=True, timeout=30)


def test_cli_provenance_and_replay(tmp_path):
    execution = run_cli(tmp_path)
    assert execution.returncode == 0, execution.stderr
    result = json.loads(execution.stdout)
    directory = tmp_path / "run"
    assert result == json.loads((directory / "result.json").read_text())
    assert result["schema_version"] == 1 and result["request_id"] == "test-1"
    assert result["is_mock"] and not result["inference_performed"] and not result["verified"]
    assert result["query"] == "flood road closures"
    assert [e["document_id"] for e in result["evidence"]] == ["E001", "E002", "E004"]
    assert "predicted_label" not in result
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["completed_samples"] == 1
    assert manifest["claim"] is None and manifest["metrics"] is None
    assert manifest["code"]["source_sha256"]
    assert manifest["environment"]["device"] == "cpu"
    for item in manifest["inputs"].values():
        assert hashlib.sha256((directory / item["snapshot"]).read_bytes()).hexdigest() == item["sha256"]
    assert manifest["result_sha256"] == hashlib.sha256((directory / "result.json").read_bytes()).hexdigest()
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    again = run_cli(tmp_path)
    assert again.returncode == 2
    assert json.loads(again.stderr)["status"] == "failed"
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    replay = run_cli(tmp_path, extra=("--output", str(tmp_path / "replay")))
    assert replay.returncode == 0, replay.stderr
    repeated = json.loads(replay.stdout)
    for key in ("observation", "query", "query_sources", "evidence"):
        assert repeated[key] == result[key]


@pytest.mark.parametrize("text", [
    '{', '{}', '[]',
    '{"objects":[],"objects":[],"visible_text":[],"uncertainties":["[MOCK]"]}',
    '{"objects":[],"visible_text":[],"uncertainties":["[MOCK]"]}',
    '{"objects":["road"],"visible_text":[],"uncertainties":[]}',
    '{"objects":["中文"],"visible_text":[],"uncertainties":["[MOCK]"]}',
])
def test_bad_observations_are_recorded(tmp_path, text):
    execution = run_cli(tmp_path, text)
    assert execution.returncode == 2 and execution.stdout == ""
    error = json.loads(execution.stderr)
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["status"] == error["status"] == "failed"
    assert manifest["error"] == error["error"]
    assert not (tmp_path / "run/result.json").exists()
    assert (tmp_path / "run/observation.json").read_text(encoding="utf-8") == text


def test_zero_hits_are_completed(tmp_path):
    execution = run_cli(tmp_path, json.dumps({"objects": ["zzzzzz"], "visible_text": [],
                                            "uncertainties": ["[MOCK]"]}))
    assert execution.returncode == 0, execution.stderr
    result = json.loads(execution.stdout)
    assert result["status"] == "completed" and result["evidence"] == []
    assert result["verified"] is False


@pytest.mark.parametrize("extra", [
    ("--top-k", "0"), ("--request-id", " "), ("--image", "missing.png"),
    ("--corpus", "missing.jsonl"), ("--config", "missing.toml"),
])
def test_invalid_run_inputs(tmp_path, extra):
    execution = run_cli(tmp_path, extra=extra)
    assert execution.returncode == 2, execution.stderr
    assert json.loads(execution.stderr)["status"] == "failed"
    assert json.loads((tmp_path / "run/manifest.json").read_text())["status"] == "failed"


def test_parser_failure_prevents_retrieval(tmp_path, monkeypatch):
    from visionevidence.cli import main
    def forbidden(*args, **kwargs):
        pytest.fail("invalid observation must not reach BM25.search")
    monkeypatch.setattr("visionevidence.retrieval.BM25Index.search", forbidden)
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert main(["observe-retrieve", "--observation", str(bad),
                 "--image", str(ROOT / "examples/vlm_demo_fixture.png"),
                 "--corpus", str(ROOT / "examples/evidence_demo.jsonl"),
                 "--request-id", "bad", "--output", str(tmp_path / "run"),
                 "--dataset-name", "synthetic", "--dataset-version", "1", "--split", "demo"]) == 2

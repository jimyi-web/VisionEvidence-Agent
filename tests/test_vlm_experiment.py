"""CPU-only checks. None of these measurements are real GPU results."""

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from visionevidence.vlm import experiment

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "examples/vlm_demo_fixture.png"
CONFIG = ROOT / "configs/vlm_mock.toml"


def test_mock_six_calls_artifacts_and_no_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "gpu_snapshot", lambda: pytest.fail("Mock accessed GPU"))
    output = tmp_path / "run"
    result = experiment.run_experiment(CONFIG, IMAGE, output)
    assert result["is_mock"] and result["purpose"] == "interface_test_only"
    assert result["status"] == "completed" and result["completed_runs"] == 6
    assert result["image_sha256"] == hashlib.sha256(IMAGE.read_bytes()).hexdigest()
    assert len(result["code_commit"]) == 40 and result["dependencies"]
    records = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
    assert [r["max_new_tokens"] for r in records] == [256, 512] * 3
    for row in records:
        assert row["phase"] == "mock" and row["response"]["is_mock"]
        assert not row["response"]["metadata"]["inference_performed"]
        assert row["generated_tokens"] is row["torch_peak_allocated_mib"] is None
        assert row["gpu_samples"] == []
    assert json.loads((output / "result.json").read_text())["status"] == "completed"
    before = (output / "result.json").read_bytes()
    with pytest.raises(FileExistsError):
        experiment.run_experiment(CONFIG, IMAGE, output)
    assert (output / "result.json").read_bytes() == before


def test_hash_manifest_detects_weight_change(tmp_path):
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"fake weights, not a model")
    first = experiment.model_manifest(tmp_path)
    shard.write_bytes(b"modified weights")
    assert first[0]["sha256"] != experiment.model_manifest(tmp_path)[0]["sha256"]
    assert first[0]["path"] == "model.safetensors"


def test_failure_saved_without_mock_fallback(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected backend failure")
    monkeypatch.setattr(experiment, "create_vlm", fail)
    with pytest.raises(RuntimeError, match="injected"):
        experiment.run_experiment(CONFIG, IMAGE, tmp_path / "failed")
    result = json.loads((tmp_path / "failed/result.json").read_text())
    assert result["status"] == "failed" and result["completed_runs"] == 0


@pytest.mark.parametrize("kwargs", [{"repeats": 2}, {"tokens": [256, 256]}, {"interval": 0}])
def test_invalid_protocol(tmp_path, kwargs):
    with pytest.raises(ValueError):
        experiment.run_experiment(CONFIG, IMAGE, tmp_path / "invalid", **kwargs)
    assert not (tmp_path / "invalid").exists()


def test_gpu_sample_parsing_and_error_recording(monkeypatch):
    monkeypatch.setattr(experiment.subprocess, "run", lambda *a, **kw:
                        subprocess.CompletedProcess(a, 0, "GPU-123, RTX 4090D, 570, 90, 9500, 24564\n"))
    assert experiment.gpu_snapshot()[0]["utilization_percent"] == 90
    def fail():
        raise OSError("nvidia-smi missing")
    monkeypatch.setattr(experiment, "gpu_snapshot", fail)
    with experiment.GPUSampler(0.2) as sampler:
        pass
    assert sampler.errors == ["nvidia-smi missing"] and not sampler.samples

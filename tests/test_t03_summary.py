"""Audit-tool checks against synthetic bundles, never GPU performance fixtures."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "t03_summary", Path(__file__).resolve().parents[1] / "scripts/summarize_t03_b5.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def bundle(tmp_path):
    config = {"device": "cuda:0", "dtype": "bfloat16", "model_path": "synthetic",
              "do_sample": False, "prompt_version": "fixture", "model_name": "fixture"}
    manifest = {"is_mock": False, "code_commit": "synthetic-commit", "git_dirty": False,
                "backend": "fixture", "config": config, "status": "preparing",
                "schedule": [256, 512] * 3, "hardware": [{"uuid": "fixture-gpu"}],
                "dependencies": ["torch==fixture", "transformers==fixture"]}
    result = {**manifest, "status": "completed", "completed_runs": 6}
    rows = []
    for i, limit in enumerate(manifest["schedule"]):
        meta = {**config, "inference_performed": True, "torch_version": "fixture",
                "transformers_version": "fixture", "model_loaded_this_call": i == 0,
                "model_load_ms": 1, "max_new_tokens": limit, "generated_tokens": limit,
                "generation_ms": 2, "peak_gpu_memory_mb": 3}
        rows.append({"run_index": i, "max_new_tokens": limit, "generated_tokens": limit,
                     "phase": "cold" if i == 0 else "reuse", "token_limit_reached": True,
                     "wall_ms": 4, "latency_ms": 2, "generation_ms": 2,
                     "torch_peak_allocated_mib": 3, "gpu_sampling_errors": [],
                     "gpu_samples": [{"gpus": [{"uuid": "fixture-gpu", "memory_used_mib": 5}]}],
                     "response": {"is_mock": False, "backend": "fixture", "model_name": "fixture",
                                  "latency_ms": 2, "description": "synthetic", "metadata": meta}})
    return manifest, result, rows


def write_bundle(path, manifest, result, rows):
    for name, data in (("manifest.json", manifest), ("result.json", result)):
        (path / name).write_text(json.dumps(data), encoding="utf-8")
    (path / "runs.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_synthetic_bundle_calculation(tmp_path):
    write_bundle(tmp_path, *bundle(tmp_path))
    report = summary.summarize(tmp_path, "synthetic-commit")
    assert report["validated"]
    assert report["reuse_wall_mean_ms"] == {"256": 4, "512": 4}
    assert report["metrics"][0]["board_sample_max_mib"] == 5
    assert report["metrics"][0]["torch_peak_allocated_mib"] == 3


@pytest.mark.parametrize("fault", ["mock", "count", "commit", "reload", "sample", "metric"])
def test_reject_inconsistent_bundle(tmp_path, fault):
    manifest, result, rows = bundle(tmp_path)
    if fault == "mock":
        rows[1]["response"]["is_mock"] = True
    elif fault == "count":
        result["completed_runs"] = 5
    elif fault == "commit":
        result["code_commit"] = "other"
    elif fault == "reload":
        rows[1]["response"]["metadata"]["model_loaded_this_call"] = True
    elif fault == "sample":
        rows[1]["gpu_sampling_errors"] = ["synthetic error"]
    else:
        rows[1]["torch_peak_allocated_mib"] = 99
    write_bundle(tmp_path, manifest, result, rows)
    with pytest.raises(ValueError):
        summary.summarize(tmp_path, "synthetic-commit")


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', "{"])
def test_strict_json(text):
    with pytest.raises(ValueError):
        summary.strict_json(text)

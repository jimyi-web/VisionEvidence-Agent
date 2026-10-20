"""Real-source gate must reject local synthetic Mock runs."""

import importlib.util
import importlib
import json
from pathlib import Path

import pytest

from visionevidence.batch import run_batch
from visionevidence.batch import sha256
from visionevidence.datasets.flickr8k import adapt_flickr8k
from tests.test_flickr8k import make_raw

spec = importlib.util.spec_from_file_location("audit_v11_dev", Path(__file__).parents[1] / "scripts/audit_v11_dev.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_freeze_audit_refuses_mock_results(tmp_path):
    raw = make_raw(tmp_path)
    (raw / "source.json").write_text(json.dumps({"repository": "intro/flickr8k",
        "revision": "c7a02cb487da0eca729e133194b0f842cb578c08",
        "files": {f"{split}/metadata.csv": {"sha256": sha256(raw / split / "metadata.csv")}
                  for split in ("train", "dev", "test")}}))
    adapter = tmp_path / "adapter"
    adapt_flickr8k(input_root=raw, output=adapter)
    run = tmp_path / "runs"
    run_batch(samples=adapter / "dev/samples.jsonl", corpus=adapter / "dev/corpus.jsonl",
              qrels=adapter / "dev/qrels.jsonl", output=run / "baseline", limit=2,
              query_mode="baseline", task="image_to_caption_retrieval_proxy", top_k=10, max_attempts=1)
    with pytest.raises(ValueError, match="Mock forbidden"):
        module.audit(run, raw)


def test_missing_dev_gate_stops_before_image_or_model_access(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    runner = importlib.import_module("run_v11_test")
    path = tmp_path / "freeze-manifest.json"
    path.write_text(json.dumps({"status": "blocked_missing_real_dev_artifacts", "selected_strategy": None}))
    def unexpected(*args, **kwargs):
        pytest.fail("blocked freeze must not inspect images or reach inference")
    monkeypatch.setattr(runner, "check_image", unexpected)
    with pytest.raises(ValueError, match="formal freeze blocked"):
        runner.preflight(path, 2)

"""Cohort boundaries and unchanged labels/candidates, entirely synthetic."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from visionevidence.batch import sha256
from visionevidence.caption_experiment import run_experiment
from visionevidence.datasets.flickr8k import adapt_flickr8k
from visionevidence.datasets.flickr8k_cohort import prepare_cohort
from visionevidence.observation_policy import ObservationPolicy
from tests.test_flickr8k import make_raw


def source(tmp_path):
    raw = make_raw(tmp_path)
    # Extend metadata only: missing held-out images must never cause a download.
    for split in ("test",):
        path = raw / split / "metadata.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        template = rows[0]
        rows = [{**template, "file_name": f"test_{i:03d}.jpg"} for i in range(30)]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(template))
            writer.writeheader()
            writer.writerows(rows)
    adapted = tmp_path / "adapter"
    adapt_flickr8k(input_root=raw, output=adapted)
    return adapted


def test_dev_copy_preserves_images_full_candidates_and_original_sources(tmp_path):
    adapted = source(tmp_path)
    before = {p.relative_to(adapted).as_posix(): sha256(p) for p in adapted.rglob("*") if p.is_file()}
    output = tmp_path / "dev-cohort"
    result = prepare_cohort(dataset=adapted / "dev", output=output, purpose="development", limit=1)
    assert result["cohort"]["method_selection_allowed"]
    assert result["splits"]["dev"]["images"] == 1
    assert result["splits"]["dev"]["candidate_caption_documents"] == 10
    assert (output / "dev/corpus.jsonl").read_bytes() == (adapted / "dev/corpus.jsonl").read_bytes()
    original_sample = json.loads((adapted / "dev/samples.jsonl").read_text().splitlines()[0])
    selected = json.loads((output / "dev/samples.jsonl").read_text())
    assert (output / "dev" / selected["image_path"]).resolve() == (adapted / "dev" / original_sample["image_path"]).resolve()
    assert json.loads((output / "dev/qrels.jsonl").read_text()) == json.loads((adapted / "dev/qrels.jsonl").read_text().splitlines()[0])
    report = run_experiment(dataset=output / "dev", output=tmp_path / "mock-run", limit=1,
                            observation_policy=ObservationPolicy(True, True))
    assert report["status"] == "completed" and report["cohort"]["purpose"] == "development"
    assert before == {p.relative_to(adapted).as_posix(): sha256(p) for p in adapted.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        prepare_cohort(dataset=adapted / "dev", output=output, purpose="development", limit=1)


def test_heldout_selection_excludes_twenty_and_does_not_require_images(tmp_path):
    adapted = source(tmp_path)
    result = prepare_cohort(dataset=adapted / "test", output=tmp_path / "heldout", purpose="heldout_test", offset=20, limit=5)
    cohort = result["cohort"]
    assert cohort["sample_ids"][0] == "flickr8k:test_020.jpg"
    assert not set(cohort["sample_ids"]) & set(cohort["diagnostic_sample_ids"])
    assert not cohort["method_selection_allowed"] and cohort["heldout_requires_prior_method_freeze"]
    assert result["splits"]["test"]["candidate_caption_documents"] == 150
    assert result["splits"]["test"]["available_images"] == 0


@pytest.mark.parametrize("split,purpose,offset,limit", [
    ("test", "development", 0, 1), ("dev", "heldout_test", 20, 1),
    ("test", "heldout_test", 19, 1), ("test", "diagnostic", 19, 2),
    ("dev", "development", 0, 100), ("dev", "development", -1, 1),
])
def test_invalid_selection_rejected_before_output(tmp_path, split, purpose, offset, limit):
    adapted = source(tmp_path)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        prepare_cohort(dataset=adapted / split, output=output, purpose=purpose, offset=offset, limit=limit)
    assert not output.exists()


def test_cohort_cannot_be_nested_or_reselected(tmp_path):
    adapted = source(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        prepare_cohort(dataset=adapted / "dev", output=adapted / "child", purpose="development", limit=1)
    output = tmp_path / "cohort"
    prepare_cohort(dataset=adapted / "dev", output=output, purpose="development", limit=1)
    with pytest.raises(ValueError, match="original"):
        prepare_cohort(dataset=output / "dev", output=tmp_path / "second", purpose="development", limit=1)


def test_cohort_cli_and_source_fingerprint_guard(tmp_path):
    adapted = source(tmp_path)
    command = [sys.executable, "-m", "visionevidence.datasets.flickr8k_cohort", "--dataset", str(adapted / "dev"),
               "--output", str(tmp_path / "cli"), "--purpose", "development", "--limit", "1"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and json.loads(result.stdout)["purpose"] == "development"
    with (adapted / "dev/corpus.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="fingerprint"):
        prepare_cohort(dataset=adapted / "dev", output=tmp_path / "mutated", purpose="development", limit=1)

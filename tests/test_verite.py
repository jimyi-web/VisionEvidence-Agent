"""Synthetic representations of official CSV format, not real dataset scores."""

import csv
import hashlib
import json

import pytest

from visionevidence.datasets.verite import adapt_verite


def inputs(tmp_path):
    prepared, articles = tmp_path / "VERITE.csv", tmp_path / "articles.csv"
    with prepared.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["", "caption", "image_path", "label"])
        writer.writerows([[0, "Synthetic true caption", "images/true_0.jpg", "true"],
                         [1, "Synthetic wrong caption", "images/true_0.jpg", "miscaptioned"],
                         [2, "Synthetic true caption", "images/false_0.jpg", "out-of-context"]])
    with articles.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "true_url", "false_url", "snopes_url"])
        writer.writerow([0, "https://example.org/true.jpg", "https://example.org/false.jpg", "https://example.org/article"])
    return dict(prepared=prepared, articles=articles, image_root=tmp_path / "local-media",
                output=tmp_path / "adapted", revision="a" * 40)


def test_adapter_preserves_labels_groups_missing_images_and_no_qrels(tmp_path):
    args = inputs(tmp_path)
    report = adapt_verite(**args, limit=3)
    assert report["queries"] == 3 and report["unique_image_paths"] == 2
    assert report["available_images"] == 0 and report["evidence_documents"] == 0
    rows = [json.loads(line) for line in (args["output"] / "samples.jsonl").read_text().splitlines()]
    assert len({r["metadata"]["group_id"] for r in rows}) == 1
    assert all(r["metadata"]["retrieval_qrels"] is None for r in rows)
    assert rows[0]["sample_id"] == "verite:0" and rows[1]["pair_label"] == "miscaptioned"
    assert rows[0]["image_path"] == rows[1]["image_path"]
    with pytest.raises(FileExistsError):
        adapt_verite(**args)


def test_revision_required_and_path_traversal_rejected(tmp_path):
    args = inputs(tmp_path)
    with pytest.raises(ValueError, match="revision"):
        adapt_verite(**{**args, "revision": "master"})
    args["prepared"].write_text(args["prepared"].read_text().replace("images/true_0.jpg", "../private.jpg"))
    with pytest.raises(ValueError, match="path"):
        adapt_verite(**args)


def test_explicit_content_snapshot_revision(tmp_path):
    args = inputs(tmp_path)
    digest = hashlib.sha256(args["prepared"].read_bytes() + args["articles"].read_bytes()).hexdigest()
    report = adapt_verite(**{**args, "revision": f"snapshot-sha256:{digest}"})
    assert report["revision"] == f"snapshot-sha256:{digest}"
    with pytest.raises(ValueError, match="fingerprint"):
        adapt_verite(**{**args, "revision": "snapshot-sha256:" + "0" * 64})

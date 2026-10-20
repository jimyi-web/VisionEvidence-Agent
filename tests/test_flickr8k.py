"""Synthetic imagefolder fixtures; no test metric is a real VLM observation."""

import csv
import io
import json
from pathlib import Path

from PIL import Image
import pytest

from visionevidence.datasets import flickr8k


def make_raw(tmp_path, *, images=True, captions_prefix="HUMAN_REFERENCE_SECRET"):
    root = tmp_path / "raw"
    root.mkdir()
    for split in flickr8k.SPLITS:
        directory = root / split
        directory.mkdir()
        with (directory / "metadata.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["file_name", "split", *(f"caption_{i}" for i in range(5))])
            for i in (1, 0):  # Adapter must select by stable IDs, not source order or caption content.
                name = f"{split}_{i}.jpg"
                writer.writerow([name, split, *(f" {captions_prefix} {split} {i} #{j} " for j in range(5))])
                if images:
                    Image.new("RGB", (32, 32), color=(i * 20, 100, 80)).save(directory / name)
    return root


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_offline_adapter_preserves_five_captions_ids_splits_and_targets(tmp_path):
    raw = make_raw(tmp_path)
    output = tmp_path / "adapted"
    manifest = flickr8k.adapt_flickr8k(input_root=raw, output=output)
    assert manifest["upstream_revision"] is None
    assert manifest["dataset_version"].startswith("offline-sha256:")
    for split in flickr8k.SPLITS:
        info = manifest["splits"][split]
        assert info["images"] == info["available_images"] == 2
        assert info["candidate_caption_documents"] == 10
        samples, captions, corpus, qrels = [load(output / split / f"{name}.jsonl") for name in ("samples", "captions", "corpus", "qrels")]
        assert samples[0]["sample_id"] == f"flickr8k:{split}_0.jpg"
        assert samples[0]["split"] == split
        assert (output / split / samples[0]["image_path"]).is_file()
        assert "\\" not in samples[0]["image_path"]  # Portable to Linux after copying the same tree.
        assert "HUMAN_REFERENCE_SECRET" not in json.dumps(samples)
        assert captions[0]["captions"][0] == f" HUMAN_REFERENCE_SECRET {split} 0 #0 "
        assert [doc["text"] for doc in corpus[:5]] == captions[0]["captions"]
        assert len(qrels[0]["relevant_ids"]) == 5 and qrels[0]["kind"] == "caption_pair_proxy"
        assert set(qrels[0]["relevant_ids"]) == {doc["evidence_id"] for doc in corpus[:5]}
    with pytest.raises(FileExistsError):
        flickr8k.adapt_flickr8k(input_root=raw, output=output)


@pytest.mark.parametrize("issue", ["caption", "split", "path", "overlap", "duplicate"])
def test_malformed_metadata_rejected(tmp_path, issue):
    raw = make_raw(tmp_path, images=False)
    path = raw / "test/metadata.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    if issue == "caption":
        rows[1][-1] = " "
    elif issue == "split":
        rows[1][1] = "train"
    elif issue == "path":
        rows[1][0] = "../escape.jpg"
    elif issue == "overlap":
        rows[1][0] = "train_0.jpg"
    else:
        rows.append(rows[1])
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError):
        flickr8k.adapt_flickr8k(input_root=raw, output=tmp_path / "out")


def test_adapter_all_metadata_with_missing_images(tmp_path):
    raw = make_raw(tmp_path, images=False)
    result = flickr8k.adapt_flickr8k(input_root=raw, output=tmp_path / "out")
    assert all(info["available_images"] == 0 and info["images"] == 2 for info in result["splits"].values())


def test_bounded_download_partial_failure_resume_and_pin(tmp_path, monkeypatch):
    original = make_raw(tmp_path, images=False)
    jpeg = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(jpeg, format="JPEG")
    calls, failed_once = [], False
    def fake_fetch(url, maximum):
        nonlocal failed_once
        calls.append(url)
        if "/api/" in url:
            return json.dumps({"sha": "a" * 40}).encode()
        assert "/" + "a" * 40 + "/" in url
        if url.endswith("README.md"):
            return b"license: cc0-1.0"
        if url.endswith("metadata.csv"):
            return (original / "/".join(url.split("/")[-2:])).read_bytes()
        if url.endswith("test_1.jpg") and not failed_once:
            failed_once = True
            raise OSError("injected transient network failure")
        return jpeg.getvalue()
    monkeypatch.setattr(flickr8k, "fetch", fake_fetch)
    output = tmp_path / "download"
    first = flickr8k.download_subset(output=output, limit=2)
    assert first["downloaded"] == 1 and len(first["errors"]) == 1
    assert not (output / "train/train_0.jpg").exists()
    calls.clear()
    second = flickr8k.download_subset(output=output, limit=2, resume=True)
    assert second["downloaded"] == second["cache_hits"] == 1 and not second["errors"]
    assert len(calls) == 1 and calls[0].endswith("test_1.jpg")
    with pytest.raises(ValueError, match="revision"):
        flickr8k.download_subset(output=output, limit=2, resume=True, revision="b" * 40)
    (output / "test/test_0.jpg").write_bytes(b"corrupt")
    bad = flickr8k.download_subset(output=output, limit=2, resume=True)
    assert len(bad["errors"]) == 1 and "changed" in bad["errors"][0]["message"]


def test_offline_source_fingerprint_is_checked(tmp_path):
    raw = make_raw(tmp_path, images=False)
    (raw / "source.json").write_text(json.dumps({"repository": flickr8k.REPO, "revision": "a" * 40,
                                                "files": {"test/metadata.csv": {"sha256": "0" * 64}}}))
    with pytest.raises(ValueError, match="fingerprint"):
        flickr8k.adapt_flickr8k(input_root=raw, output=tmp_path / "out")


def test_offset_download_fetches_only_fixed_slice_and_resumes(tmp_path, monkeypatch):
    original = make_raw(tmp_path, images=False)
    jpeg = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(jpeg, format="JPEG")
    calls = []
    def fetch(url, maximum):
        calls.append(url)
        if url.endswith("README.md"):
            return b"fixture"
        if url.endswith("metadata.csv"):
            return (original / "/".join(url.split("/")[-2:])).read_bytes()
        return jpeg.getvalue()
    monkeypatch.setattr(flickr8k, "fetch", fetch)
    output = tmp_path / "download"
    receipt = flickr8k.download_subset(output=output, revision="a" * 40, offset=1, limit=1)
    assert receipt["offset"] == 1 and receipt["downloaded"] == 1
    assert [url for url in calls if url.endswith(".jpg")] == [
        "https://huggingface.co/datasets/intro/flickr8k/resolve/" + "a" * 40 + "/test/test_1.jpg"]
    assert not (output / "test/test_0.jpg").exists()
    calls.clear()
    again = flickr8k.download_subset(output=output, offset=1, limit=1, resume=True)
    assert again["cache_hits"] == 1 and calls == []
    for offset, limit in ((-1, 1), (True, 1), (2, 1), (1, 2)):
        with pytest.raises(ValueError):
            flickr8k.download_subset(output=output, offset=offset, limit=limit, resume=True)

"""Pinned, bounded Flickr8k acquisition and offline imagefolder adaptation.

Captions are reference targets for a retrieval proxy, never input observations.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen

from visionevidence.batch import dumps, now, run_lock, sha256
from visionevidence.vlm.factory import VLMConfig
from visionevidence.vlm.qwen3_vl import prepare_image
from visionevidence.vlm.schemas import VLMRequest

REPO = "intro/flickr8k"
SPLITS = ("train", "dev", "test")
FIELDS = {"file_name", "split", *(f"caption_{i}" for i in range(5))}
TASK = "image_to_caption_retrieval_proxy"


def fetch(url: str, maximum: int) -> bytes:
    with urlopen(Request(url, headers={"User-Agent": "VisionEvidence-Agent-research"}), timeout=30) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError(f"download exceeded {maximum} bytes: {url}")
    return data


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(dumps(value) + "\n", encoding="utf-8")
    temp.replace(path)


def rows_for_split(root: Path, split: str) -> list[dict]:
    """Validate raw values without normalizing or stripping reference captions."""
    with (root / split / "metadata.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if len(reader.fieldnames or []) != len(FIELDS) or set(reader.fieldnames or []) != FIELDS:
            raise ValueError(f"{split}: expected exactly file_name,split,caption_0..4")
        rows, seen = [], set()
        for row in reader:
            name = row["file_name"]
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+\.jpg", name):
                raise ValueError(f"unsafe or unsupported image filename: {name!r}")
            if name in seen or row["split"] != split:
                raise ValueError(f"duplicate image or original split mismatch: {name}")
            if None in row or any(not isinstance(row[f"caption_{i}"], str) or not row[f"caption_{i}"].strip() for i in range(5)):
                raise ValueError(f"{name}: all five original captions must be present")
            seen.add(name)
            rows.append(row)
    if not rows:
        raise ValueError(f"empty split: {split}")
    return sorted(rows, key=lambda row: row["file_name"])


def check_image(path: Path) -> None:
    picture = prepare_image(VLMRequest(path, "Validate local image only"), VLMConfig())
    picture.close()


def download_subset(*, output: Path, split: str = "test", limit: int = 20,
                    revision: str | None = None, resume: bool = False,
                    offset: int = 0) -> dict:
    if (split not in SPLITS or type(limit) is not int or limit <= 0
            or type(offset) is not int or offset < 0):
        raise ValueError("choose an original split, positive image limit and nonnegative offset")
    output = Path(output)
    if resume:
        source = json.loads((output / "source.json").read_text(encoding="utf-8"))
        if source.get("repository") != REPO:
            raise ValueError("download provenance repository mismatch")
        if revision is not None and revision != source["revision"]:
            raise ValueError("cannot change dataset revision on resume")
        revision = source["revision"]
    elif revision is None:
        revision = json.loads(fetch(f"https://huggingface.co/api/datasets/{REPO}", 2 * 1024**2))["sha"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("dataset revision must be an exact 40-character commit SHA")
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    with run_lock(output / ".download.lock"):
        if not resume:
            files = {}
            for relative in ("README.md", *(f"{s}/metadata.csv" for s in SPLITS)):
                url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{relative}"
                content = fetch(url, 8 * 1024**2)
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                files[relative] = {"url": url, "sha256": sha256(path), "bytes": len(content)}
            source = {"repository": REPO, "revision": revision, "retrieved_utc": now(), "files": files,
                      "provider_license_claim": "cc0-1.0", "original_photo_rights_verified": False,
                      "usage": "local noncommercial research; do not assume per-image CC0 rights"}
            atomic_json(output / "source.json", source)
        for relative, info in source["files"].items():
            if sha256(output / relative) != info["sha256"]:
                raise ValueError(f"source metadata changed: {relative}")
        rows = rows_for_split(output, split)
        if offset + limit > len(rows):
            raise ValueError(f"requested offset {offset} + {limit} images but {split} has only {len(rows)}; do not mix original splits")
        inventory_path = output / "image-inventory.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.exists() else {}
        receipt = {"started_utc": now(), "repository": REPO, "revision": revision, "split": split,
                   "selection": "ascending ASCII filenames [offset:offset+limit], independent of captions and scores",
                   "offset": offset,
                   "limit": limit, "downloaded": 0, "cache_hits": 0, "errors": [], "image_bytes_downloaded": 0}
        for row in rows[offset:offset + limit]:
            relative = f"{split}/{row['file_name']}"
            path = output / relative
            try:
                if relative in inventory and path.is_file():
                    if sha256(path) != inventory[relative]["sha256"]:
                        raise ValueError(f"cached image changed: {relative}")
                    check_image(path)
                    receipt["cache_hits"] += 1
                    continue
                if path.exists():
                    raise ValueError(f"untracked existing image: {relative}; use offline import or a new directory")
                url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{relative}"
                content = fetch(url, 10 * 1024**2)
                temporary = path.with_name(path.stem + ".download.jpg")
                temporary.write_bytes(content)
                check_image(temporary)
                temporary.replace(path)
                inventory[relative] = {"url": url, "sha256": sha256(path), "bytes": path.stat().st_size,
                                       "downloaded_utc": now()}
                atomic_json(inventory_path, inventory)
                receipt["downloaded"] += 1
                receipt["image_bytes_downloaded"] += len(content)
            except Exception as exc:
                receipt["errors"].append({"image": relative, "type": type(exc).__name__, "message": str(exc)})
        receipt["finished_utc"] = now()
        number = 1
        while (output / f"download-{number:04d}.json").exists():
            number += 1
        with (output / f"download-{number:04d}.json").open("x", encoding="utf-8") as stream:
            stream.write(dumps(receipt) + "\n")
        return receipt


def adapt_flickr8k(*, input_root: Path, output: Path, revision: str | None = None) -> dict:
    """Offline adapter; preserve all three splits even if only 20 images exist.

    Candidate pools use ALL five captions of every image in the SAME split.
    No labels/captions are copied into the inference samples.
    """
    root, output = Path(input_root).resolve(), Path(output).resolve()
    source_path = root / "source.json"
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else None
    if source:
        if source.get("repository") != REPO:
            raise ValueError("source repository mismatch")
        if revision is not None and revision != source["revision"]:
            raise ValueError("explicit revision contradicts source manifest")
        revision = source["revision"]
        for relative, info in source["files"].items():
            if sha256(root / relative) != info["sha256"]:
                raise ValueError("source file fingerprint mismatch")
    tables = {split: rows_for_split(root, split) for split in SPLITS}
    hashes = {split: sha256(root / split / "metadata.csv") for split in SPLITS}
    # Offline imports without upstream proof use content identity, never a guessed commit.
    version = revision or "offline-sha256:" + hashlib.sha256(dumps(hashes).encode()).hexdigest()
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full dataset commit SHA")
    seen = set()
    for rows in tables.values():
        for row in rows:
            if row["file_name"] in seen:
                raise ValueError("image ID overlaps original splits")
            seen.add(row["file_name"])
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"task": TASK, "repository": REPO, "dataset_version": version,
                "upstream_revision": revision, "input_metadata_sha256": hashes,
                "source_manifest_sha256": sha256(source_path) if source else None,
                "selection_order": "ascending filename per original split", "splits": {},
                "qrels_semantics": "five same-image captions are proxy positives; other-image captions are assumed negatives, not fact judgments",
                "source_verified": source is not None, "provider_license_claim": "cc0-1.0",
                "original_photo_rights_verified": False}
    for split, rows in tables.items():
        directory = output / split
        directory.mkdir()
        names = ("samples", "captions", "corpus", "qrels")
        streams = {name: (directory / f"{name}.jsonl").open("x", encoding="utf-8") for name in names}
        available = 0
        image_hashes = {}
        try:
            for row in rows:
                image_id = row["file_name"]
                sample_id = f"flickr8k:{image_id}"
                image = root / split / image_id
                if image.is_file():
                    check_image(image)
                    available += 1
                    image_hashes[image_id] = sha256(image)
                reference = f"https://huggingface.co/datasets/{REPO}/blob/{revision or 'main'}/{split}/metadata.csv"
                sample = {"sample_id": sample_id, "image_path": Path(os.path.relpath(image, directory)).as_posix(),
                          "claim": "Describe the visible image only.", "dataset": "Flickr8k",
                          "dataset_version": version, "split": split, "source": reference,
                          "metadata": {"image_id": image_id, "original_split": split, "task": TASK}}
                streams["samples"].write(dumps(sample) + "\n")
                captions = [row[f"caption_{i}"] for i in range(5)]
                streams["captions"].write(dumps({"image_id": image_id, "split": split, "captions": captions}) + "\n")
                ids = []
                for i, caption in enumerate(captions):
                    document_id = f"{sample_id}#{i}"
                    ids.append(document_id)
                    streams["corpus"].write(dumps({"evidence_id": document_id, "text": caption,
                        "source": reference, "metadata": {"image_id": image_id, "caption_index": i,
                                                             "split": split, "task": TASK}}) + "\n")
                streams["qrels"].write(dumps({"sample_id": sample_id, "relevant_ids": ids,
                    "annotation_source": f"{REPO}@{version}:original same-image caption association (proxy)",
                    "kind": "caption_pair_proxy"}) + "\n")
        finally:
            for stream in streams.values():
                stream.close()
        manifest["splits"][split] = {"images": len(rows), "candidate_caption_documents": 5 * len(rows),
                                    "available_images": available, "available_image_sha256": image_hashes,
                                    "outputs": {name: sha256(directory / f"{name}.jsonl") for name in names}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Fetch split metadata and a fixed filename slice")
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--split", choices=SPLITS, default="test")
    download.add_argument("--limit", type=int, default=20)
    download.add_argument("--offset", type=int, default=0, help="Zero-based offset in ascending original filenames")
    download.add_argument("--revision")
    download.add_argument("--resume", action="store_true")
    adapt = commands.add_parser("adapt", help="Offline import of train/dev/test imagefolder metadata")
    adapt.add_argument("--input-root", type=Path, required=True)
    adapt.add_argument("--output", type=Path, required=True)
    adapt.add_argument("--revision")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        result = download_subset(**args) if command == "download" else adapt_flickr8k(**args)
        print(dumps(result))
        return 1 if result.get("errors") else 0
    except Exception as exc:
        print(dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

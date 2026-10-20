"""Offline cohort copies: select only by original filename order, keep full candidates.

No images are fetched, no model is called, and source bundles are never edited.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

from visionevidence.batch import dumps, sha256
from visionevidence.datasets.flickr8k import TASK
from visionevidence.retrieval.corpus import read_jsonl


def prepare_cohort(*, dataset: Path, output: Path, purpose: str,
                   limit: int = 20, offset: int = 0) -> dict:
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if type(limit) is not int or limit <= 0 or type(offset) is not int or offset < 0:
        raise ValueError("limit must be positive and offset nonnegative")
    if output == dataset.parent or output.is_relative_to(dataset.parent):
        raise ValueError("cohort output must be outside the source adapter bundle")
    manifest_path = dataset.parent / "manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    if original.get("task") != TASK or "cohort" in original or dataset.name not in original["splits"]:
        raise ValueError("select from one original Flickr8k adapter split, not another cohort")
    split = dataset.name
    if purpose not in {"development", "diagnostic", "heldout_test"}:
        raise ValueError("unknown cohort purpose")
    if purpose == "development" and split != "dev":
        raise ValueError("development selection requires the original dev split")
    if purpose == "diagnostic" and (split != "test" or offset + limit > 20):
        raise ValueError("diagnostic cohort must stay within the known test first 20")
    if purpose == "heldout_test" and (split != "test" or offset < 20):
        raise ValueError("heldout test must exclude the known diagnostic first 20")
    info = original["splits"][split]
    for name, expected in info["outputs"].items():
        if sha256(dataset / f"{name}.jsonl") != expected:
            raise ValueError(f"source adapter fingerprint changed: {name}")
    samples = [row for _, row in read_jsonl(dataset / "samples.jsonl")]
    image_ids = [row["metadata"]["image_id"] for row in samples]
    if image_ids != sorted(image_ids) or len(set(image_ids)) != len(image_ids):
        raise ValueError("original samples must have unique ascending filename order")
    if any(row["split"] != split for row in samples) or len(samples) != info["images"]:
        raise ValueError("source split or image count mismatch")
    if offset + limit > len(samples):
        raise ValueError("requested cohort exceeds original split")
    selected = samples[offset:offset + limit]  # Captions and qrels never influence selection.
    selected_ids = {row["sample_id"] for row in selected}
    qrels = [row for _, row in read_jsonl(dataset / "qrels.jsonl") if row["sample_id"] in selected_ids]
    if len(qrels) != limit or {row["sample_id"] for row in qrels} != selected_ids:
        raise ValueError("cohort qrels coverage mismatch")
    output.mkdir(parents=True, exist_ok=False)
    target = output / split
    target.mkdir()
    # Full original corpus and reference text are retained, never mined for query terms.
    for name in ("corpus", "captions"):
        shutil.copyfile(dataset / f"{name}.jsonl", target / f"{name}.jsonl")
    with (target / "samples.jsonl").open("x", encoding="utf-8") as stream:
        for source_row in selected:
            row = deepcopy(source_row)
            image = Path(row["image_path"])
            if not image.is_absolute():
                image = dataset / image
            row["image_path"] = Path(os.path.relpath(image.resolve(), target)).as_posix()
            stream.write(dumps(row) + "\n")
    with (target / "qrels.jsonl").open("x", encoding="utf-8") as stream:
        for row in qrels:
            stream.write(dumps(row) + "\n")
    result = deepcopy(original)
    selected_images = {row["metadata"]["image_id"] for row in selected}
    selected_hashes = {k: v for k, v in info.get("available_image_sha256", {}).items() if k in selected_images}
    result["cohort"] = {
        "version": "flickr8k_v11a_cohort_v1", "purpose": purpose,
        "selection": "original ascending filenames; offset is zero-based; independent of captions/results",
        "offset": offset, "limit": limit, "source_manifest_sha256": sha256(manifest_path),
        "sample_ids": [row["sample_id"] for row in selected],
        "diagnostic_sample_ids": [row["sample_id"] for row in samples[:20]] if split == "test" else [],
        "method_selection_allowed": purpose == "development",
        "heldout_requires_prior_method_freeze": purpose == "heldout_test",
    }
    result["splits"] = {split: {**info, "images": limit,
        "original_split_images": info["images"], "available_images": len(selected_hashes),
        "available_image_sha256": selected_hashes,
        "outputs": {name: sha256(target / f"{name}.jsonl") for name in info["outputs"]}}}
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(dumps(result) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--purpose", choices=("development", "diagnostic", "heldout_test"), required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        result = prepare_cohort(**vars(args))
        print(dumps(result["cohort"]))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

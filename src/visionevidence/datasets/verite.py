"""Adapt official VERITE CSV metadata without downloading third-party images."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

from visionevidence.batch import dumps, sha256, validate_sample

OFFICIAL = "https://github.com/stevejpapad/image-text-verification"


def adapt_verite(*, prepared: Path, articles: Path, image_root: Path, output: Path,
                 revision: str, limit: int = 12) -> dict:
    if type(limit) is not int or limit <= 0:
        raise ValueError("limit must be positive")
    if not re.fullmatch(r"[0-9a-f]{40}|snapshot-sha256:[0-9a-f]{64}", revision):
        raise ValueError("revision must be an official commit SHA or an explicit snapshot-sha256 content fingerprint")
    if revision.startswith("snapshot-sha256:"):
        actual = hashlib.sha256(Path(prepared).read_bytes() + Path(articles).read_bytes()).hexdigest()
        if revision != f"snapshot-sha256:{actual}":
            raise ValueError("revision snapshot fingerprint contradicts actual CSV bytes")
    prepared, articles, image_root = Path(prepared), Path(articles), Path(image_root).resolve()
    with articles.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "true_url", "false_url", "snopes_url"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("official articles CSV is missing required fields")
        mapping = {}
        for row in reader:
            if row["id"] in mapping:
                raise ValueError("duplicate article id")
            mapping[row["id"]] = row
    rows, seen = [], set()
    with prepared.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"", "caption", "image_path", "label"} <= set(reader.fieldnames or []):
            raise ValueError("expected the official processed VERITE.csv columns")
        for row in reader:
            if len(rows) == limit:
                break
            identifier = row[""]
            if not identifier.isdigit() or identifier in seen:
                raise ValueError("VERITE row indices must be unique numeric IDs")
            seen.add(identifier)
            match = re.fullmatch(r"images/(true|false)_([0-9]+)\.jpg", row["image_path"])
            if not match or row["label"] not in {"true", "miscaptioned", "out-of-context"}:
                raise ValueError("unsupported VERITE image path or label")
            role, group = match.groups()
            if group not in mapping:
                raise ValueError(f"missing source article group {group}")
            if (role == "false") != (row["label"] == "out-of-context"):
                raise ValueError("label/image role contradicts official packing convention")
            article = mapping[group]
            image_url = article["true_url" if role == "true" else "false_url"]
            if not image_url.startswith(("https://", "http://")):
                raise ValueError("missing original image URL")
            sample = {"sample_id": f"verite:{identifier}", "claim": row["caption"],
                      "image_path": str(image_root / row["image_path"]), "dataset": "VERITE",
                      "dataset_version": revision, "split": "benchmark_subset",
                      "source": f"{OFFICIAL}/blob/{'master' if revision.startswith('snapshot-') else revision}/VERITE/VERITE.csv",
                      "pair_label": row["label"],
                      "metadata": {"group_id": f"verite-article:{group}", "official_row_id": identifier,
                                   "image_url": image_url, "article_url": article["snopes_url"],
                                   "label_source": "dataset authors; image-caption classification only",
                                   "retrieval_qrels": None, "locally_human_verified": False,
                                   "usage": "research only; third-party image rights remain with owners"}}
            validate_sample(sample)
            rows.append(sample)
    if not rows:
        raise ValueError("empty VERITE subset")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    samples = output / "samples.jsonl"
    with samples.open("x", encoding="utf-8") as stream:
        for sample in rows:
            stream.write(dumps(sample) + "\n")
    manifest = {"dataset": "VERITE", "official_repository": OFFICIAL, "revision": revision,
                "inputs": {"prepared_sha256": sha256(prepared), "articles_sha256": sha256(articles)},
                "samples_sha256": sha256(samples), "subset_rule": "first N official CSV records, no relabeling",
                "requested_limit": limit, "queries": len(rows),
                "unique_image_paths": len({s["image_path"] for s in rows}),
                "available_images": sum(Path(p).is_file() for p in {s["image_path"] for s in rows}),
                "labels": dict(Counter(s["pair_label"] for s in rows)),
                "evidence_documents": 0, "real_vlm_image_calls": 0, "retrieval_qrels": None,
                "split_note": "benchmark only, not training/dev; keep article group together if splitting later",
                "license": {"repository": "Apache-2.0", "dataset_instruction": "research purposes only",
                            "third_party_media": "not covered by assuming repository license; not downloaded"}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "articles", "image-root", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args(argv)
    try:
        result = adapt_verite(**vars(args))
        print(dumps(result))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(dumps({"status": "failed", "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Seal a NEW freeze directory after auditing real dev artifacts; never edit preparation."""

import argparse
import json
from pathlib import Path
import shutil

from prepare_v11_freeze import ROOT, canonical_hash, write
from audit_v11_dev import ARMS, require
from visionevidence.batch import sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    prepared = json.loads(args.prepared.read_text(encoding="utf-8"))
    audited = json.loads(args.audit.read_text(encoding="utf-8"))
    require(not args.output.exists(), "freeze output must be new")
    require(audited["status"] == "audited" and audited["selected_strategy"] == "format_details", "real dev audit required")
    require(set(audited["branches"]) == set(ARMS), "all five arms required")
    require(sha256(args.audit.parent / "dev-originals.tar.gz") == audited["archive_sha256"], "dev archive hash mismatch")
    if "transport" in audited:
        require(sha256(args.audit.parent / "all-dev-originals.tar.gz") == audited["transport"]["sha256"], "complete dev transport hash mismatch")
    for relative, expected in prepared["code_and_config_sha256"].items():
        require(canonical_hash(ROOT / relative) == expected, f"code/config changed since preparation: {relative}")
    # Scientific inference code must remain identical to the audited dev commit.
    for relative, expected in audited["selected_identity"]["source_sha256"].items():
        if relative == "visionevidence/datasets/flickr8k.py":
            continue  # Only fixed-offset acquisition was added; inference does not use this downloader.
        require(canonical_hash(ROOT / "src" / relative) == expected, f"method changed after dev: {relative}")
    for name in ("baseline", "format_details"):
        identity = audited["baseline_identity"] if name == "baseline" else audited["selected_identity"]
        import hashlib
        require(hashlib.sha256(identity["effective_prompt"].encode()).hexdigest() == prepared["prompt_sha256"][name], "prompt mismatch")
    prepared.update(status="frozen", selected_strategy="format_details", blockers=[],
                    method_git_commit=audited["dev_commit"],
                    model_file_sha256=audited["selected_identity"]["model_file_sha256"],
                    vlm_config=audited["selected_identity"]["vlm_config"],
                    measured_dev_environment=audited["environment"],
                    dev_audit_sha256=sha256(args.audit), dev_archive_sha256=audited["archive_sha256"],
                    dev_metrics=audited["branches"], reported_dev_metrics_unverified=None,
                    all_dev_transport=audited.get("transport"),
                    preliminary_dev=audited.get("preliminary"),
                    measured_dev_runtime=audited["branches"]["format_details"]["runtime"],
                    preparation_manifest_sha256=sha256(args.prepared))
    args.output.mkdir(parents=True, exist_ok=False)
    for path in args.prepared.parent.iterdir():
        if path.suffix in (".toml", ".txt"):
            shutil.copyfile(path, args.output / path.name)
    shutil.copyfile(args.audit, args.output / "dev-audit.json")
    write(args.output / "freeze-manifest.json", prepared)
    print(json.dumps({"status": "frozen", "manifest": str(args.output / "freeze-manifest.json"),
                      "next": "commit freeze before GPU authorization; do not change strategy after test"}))


if __name__ == "__main__":
    main()

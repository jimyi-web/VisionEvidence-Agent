"""Fail-closed frozen test runner. Default is CPU preflight only; no model load.

Use --execute only after explicit human authorization of GPU inference.
The output, policy and lifetime attempt budget cannot be changed here.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visionevidence.batch import sha256
from visionevidence.datasets.flickr8k import check_image
from prepare_v11_freeze import canonical_hash


def preflight(manifest_path: Path, limit: int):
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    if frozen["status"] != "frozen" or frozen["selected_strategy"] != "format_details":
        raise ValueError("formal freeze blocked: audited real dev evidence and model hashes required")
    if limit not in (2, 100):
        raise ValueError("only frozen 2-image smoke or 100-image full run is allowed")
    for relative, expected in frozen["code_and_config_sha256"].items():
        if canonical_hash(ROOT / relative) != expected:
            raise ValueError(f"frozen code/config changed: {relative}")
    policy = manifest_path.parent / "format_details.toml"
    if canonical_hash(policy) != canonical_hash(ROOT / "configs/v11/format_details.toml"):
        raise ValueError("frozen policy copy changed")
    for branch, expected in frozen["prompt_sha256"].items():
        if sha256(manifest_path.parent / f"{branch}-prompt.txt") != expected:
            raise ValueError("frozen prompt changed")
    dataset = ROOT / frozen["dataset"]["cohort"] / "test"
    if sha256(dataset.parent / "manifest.json") != frozen["dataset"]["adapter_manifest_sha256"]:
        raise ValueError("cohort manifest changed")
    for label, expected in frozen["dataset"]["output_sha256"].items():
        if sha256(dataset / f"{label}.jsonl") != expected:
            raise ValueError(f"test inputs changed: {label}")
    samples = [json.loads(line) for line in (dataset / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    if [s["sample_id"] for s in samples] != frozen["dataset"]["sample_ids"]:
        raise ValueError("frozen sample IDs/order mismatch")
    raw = ROOT / frozen["dataset"]["raw"]
    for relative, expected in frozen["dataset"]["metadata_sha256"].items():
        if sha256(raw / relative) != expected:
            raise ValueError("raw metadata changed")
    inventory = json.loads((raw / "image-inventory.json").read_text(encoding="utf-8"))
    # Validate ALL 100 files even before smoke; no missing-image failures can be hidden.
    for sample in samples:
        path = (dataset / sample["image_path"]).resolve()
        relative = path.relative_to(raw.resolve()).as_posix()
        if sha256(path) != inventory[relative]["sha256"]:
            raise ValueError(f"image hash mismatch: {relative}")
        check_image(path)
    from visionevidence.vlm.factory import load_vlm_config
    from dataclasses import asdict
    config = load_vlm_config(ROOT / "configs/t05_qwen3_vl.toml")
    values = asdict(config)
    values["model_path"] = str(config.model_path)
    if values != frozen["vlm_config"]:
        raise ValueError("VLM configuration differs from audited dev")
    actual_models = {p.relative_to(config.model_path).as_posix(): sha256(p)
                     for p in sorted(config.model_path.rglob("*")) if p.is_file()}
    if not actual_models or actual_models != frozen["model_file_sha256"]:
        raise ValueError("model hash mismatch; do not download or replace weights")
    environment = frozen["measured_dev_environment"]
    if sys.version != environment["python"]:
        raise ValueError("reuse exact dev Python environment")
    for package in ("torch", "transformers", "Pillow", "visionevidence-agent"):
        if importlib.metadata.version(package) != environment["dependencies"][package]:
            raise ValueError(f"environment dependency drift: {package}; do not upgrade")
    return frozen, dataset, policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, choices=(2, 100), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest_path = args.manifest.resolve()
        # Require a committed and unchanged freeze record before inspecting any run.
        relative = manifest_path.relative_to(ROOT).as_posix()
        git = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
        committed = subprocess.check_output([*git, "show", f"HEAD:{relative}"], cwd=ROOT)
        if committed.replace(b"\r\n", b"\n") != manifest_path.read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError("freeze manifest must be committed and unchanged")
        frozen, dataset, policy = preflight(manifest_path, args.limit)
        output = ROOT / frozen["full_run"]["output"]
        if args.resume != output.exists():
            raise ValueError("new output requires no --resume; existing output requires --resume")
        if args.resume:
            # Read session budgets, never rankings. Increased attempts are forbidden.
            import sqlite3
            from contextlib import closing
            for branch in ("baseline", "structured"):
                state = output / branch / "state.sqlite3"
                if not state.exists():
                    continue
                with closing(sqlite3.connect(state.resolve().as_uri() + "?mode=ro", uri=True)) as db:
                    if any(json.loads(s[0])["max_attempts"] != 1 for s in db.execute("SELECT settings FROM sessions")):
                        raise ValueError("historical attempt budget violates frozen protocol")
                    if db.execute("SELECT max(attempts) FROM samples").fetchone()[0] > 1:
                        raise ValueError("historical retries violate frozen protocol")
        print(json.dumps({"status": "preflight_passed", "images": args.limit, "max_attempts": 1,
                          "output": str(output), "execute": args.execute}), flush=True)
        if not args.execute:
            return 0
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines()
        if not gpu or "4090 D" not in gpu[0].replace("4090D", "4090 D"):
            raise ValueError("expected existing RTX 4090D at cuda:0")
        command = [sys.executable, "-m", "visionevidence.caption_experiment", "--dataset", str(dataset),
                   "--output", str(output), "--mode", "qwen3_vl", "--config", str(ROOT / "configs/t05_qwen3_vl.toml"),
                   "--observation-config", str(policy), "--limit", str(args.limit), "--batch-size", "4", "--max-attempts", "1"]
        if args.resume:
            command.append("--resume")
        import os
        environment = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        return subprocess.call(command, cwd=ROOT, env=environment)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

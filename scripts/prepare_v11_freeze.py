"""Offline preparation. A formal freeze requires a successful five-arm raw audit.

Never reads retrieval outputs from test. All destinations must be new.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visionevidence.batch import sha256
from visionevidence.datasets.flickr8k import adapt_flickr8k
from visionevidence.datasets.flickr8k_cohort import prepare_cohort
from visionevidence.description import PROMPT as BASELINE_PROMPT
from visionevidence.observation import PROMPT
from visionevidence.observation_policy import ObservationPolicy

REVISION = "c7a02cb487da0eca729e133194b0f842cb578c08"
FREEZE = Path("experiments/v11/independent-test-20261020-01")
RAW = Path("data/raw/flickr8k-v11-test-01")
ORIGINAL = Path("data/processed/flickr8k-v11-test-original-01")
COHORT = Path("data/processed/flickr8k-v11-heldout100-01")
PACKAGE = Path("outputs/flickr8k-v11-test-inputs-01.tar.gz")


def canonical_hash(path):
    """Git text identity across Windows CRLF and Linux LF; binary inputs use sha256."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-source", type=Path, required=True)
    args = parser.parse_args(argv)
    source = json.loads((args.raw_source / "source.json").read_text(encoding="utf-8"))
    if source["revision"] != REVISION:
        raise ValueError("dataset revision mismatch")
    for relative, info in source["files"].items():
        if sha256(args.raw_source / relative) != info["sha256"]:
            raise ValueError("source metadata changed")
    for path in (FREEZE, RAW, ORIGINAL, COHORT, PACKAGE):
        if path.exists():
            raise FileExistsError(f"will not overwrite {path}; use a reviewed new preparation version")
    RAW.mkdir(parents=True)
    for relative in ("source.json", *source["files"]):
        target = RAW / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.raw_source / relative, target)
    adapt_flickr8k(input_root=RAW, output=ORIGINAL)
    cohort = prepare_cohort(dataset=ORIGINAL / "test", output=COHORT, purpose="heldout_test", offset=20, limit=100)
    info = cohort["splits"]["test"]
    if info["candidate_caption_documents"] != 5000 or info["available_images"] != 0:
        raise ValueError("unexpected test pool/image availability")
    with tarfile.open(PACKAGE, "x:gz") as stream:
        for path in (RAW, ORIGINAL, COHORT):
            stream.add(path, arcname=path.as_posix())
    FREEZE.mkdir(parents=True)
    for name in ("v10_control", "format_only", "details_only", "format_details"):
        (FREEZE / f"{name}.toml").write_text((ROOT / f"configs/v11/{name}.toml").read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    prompts = {"baseline": BASELINE_PROMPT, "format_details": ObservationPolicy(True, True).prompt(PROMPT)}
    for name, prompt in prompts.items():
        (FREEZE / f"{name}-prompt.txt").write_text(prompt, encoding="utf-8", newline="\n")
    tracked_inputs = [*sorted((ROOT / "src").rglob("*.py")), ROOT / "configs/observation.schema.json", ROOT / "configs/t05_qwen3_vl.toml",
                      ROOT / "configs/v11/evaluation_plan.toml", *sorted((ROOT / "configs/v11").glob("*.toml"))]
    tracked_inputs += [ROOT / "scripts" / name for name in
                       ("audit_v11_dev.py", "prepare_v11_freeze.py", "seal_v11_freeze.py", "run_v11_test.py")]
    tracked_inputs = sorted(set(tracked_inputs))
    manifest = {
        "protocol_version": "flickr8k_v11_independent_test_v1",
        "status": "blocked_missing_real_dev_artifacts",
        "task": "image_to_caption_retrieval_proxy", "is_fact_verification": False,
        "requested_candidate": "format_details", "selected_strategy": None,
        "blockers": ["Five-arm real dev SQLite, manifests, summaries and raw responses are absent; model hashes and measured environment cannot be verified. User summaries are not raw evidence."],
        "method_git_commit": subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sealing_commit": "Resolve with git log -1 --format=%H -- <this manifest>; a commit cannot embed its own SHA",
        "source_hash_convention": "SHA256 of UTF-8 source bytes with CRLF normalized to LF; runtime batch manifests additionally record exact disk byte hashes",
        "code_and_config_sha256": {p.relative_to(ROOT).as_posix(): canonical_hash(p) for p in tracked_inputs},
        "prompt_sha256": {name: hashlib.sha256(prompt.encode()).hexdigest() for name, prompt in prompts.items()},
        "schema_sha256": canonical_hash(ROOT / "configs/observation.schema.json"),
        "format_contract": {"objects_max": 6, "visible_text_max": 2, "uncertainties_max": 2, "string_characters_max": 100,
                            "total_whitespace_words_max": 60, "duplicates": "reject casefold and whitespace-normalized duplicates within each field"},
        "model_file_sha256": None,
        "vlm_config": None,
        "measured_dev_environment": None,
        "generation": {"max_new_tokens": 384, "do_sample": False, "num_beams": 1, "num_return_sequences": 1,
                       "seed": 0, "dtype": "bfloat16", "max_image_side": 1280, "max_image_pixels": 20000000,
                       "max_image_size_mb": 10, "temperature": "unset, not used by greedy decoding", "token_limit": "generated_tokens >= 384 is failure in both branches"},
        "retrieval": {"k1": 1.5, "b": .75, "top_k": 10, "tokenizer": "ASCII letters/digits, lowercase; exact implementation frozen by source hash", "tie_break": "document_id ascending"},
        "dataset": {"repository": "intro/flickr8k", "revision": REVISION,
                    "metadata_sha256": {k: v["sha256"] for k, v in source["files"].items()},
                    "split_provenance_limit": "mirror original split fields; original author lists not independently reacquired",
                    "original_test_images": 1000, "offset": 20, "limit": 100, "order": "ascending ASCII filename",
                    "sample_ids": cohort["cohort"]["sample_ids"], "excluded_diagnostic_sample_ids": cohort["cohort"]["diagnostic_sample_ids"],
                    "candidate_caption_documents": 5000, "cohort": COHORT.as_posix(), "raw": RAW.as_posix(),
                    "output_sha256": info["outputs"], "adapter_manifest_sha256": sha256(COHORT / "manifest.json"),
                    "qrels": "five original same-image captions positive; all other-image captions assumed negative; kind=caption_pair_proxy",
                    "images_to_acquire": 100, "image_sha256": "record in append-only download receipts; verify inventory before every run; never select by pixels, captions or results"},
        "test_arms": ["baseline", "format_details"],
        "failure_policy": "all 100 selected images in denominator, every failed/interrupted/pending query scores zero; no repair, fallback, deletion, sample substitution or increased attempts",
        "max_attempts": 1, "smoke": {"images": 2, "calls": 4, "included_in_full_run": True},
        "full_run": {"images": 100, "calls": 200, "output": "outputs/flickr8k-v11-independent100-01"},
        "metrics": ["Hit@1", "Hit@5", "Hit@10", "MRR@10", "schema_success_rate (structured only)", "format_contract_success_rate", "workflow_success_rate", "failure_rate", "response_latency_ms including failed responses", "attempt_wall_ms", "retrieval_ms successful only", "cold model load separately", "per-image paired differences"],
        "hit10_ci95": {"method": "paired image bootstrap percentile", "repetitions": 20000, "seed": 0,
                        "unit": "one image pair, including failures", "quantiles": [0.025, 0.975], "interpolation": "linear", "direction": "format_details minus baseline"},
        "budget": {"smoke_wall_limit_minutes": 10, "full_resume_wall_limit_minutes": 60,
                   "planning_seconds_per_call": 10, "headroom": .20, "full_planning_minutes_excluding_setup": 40,
                   "cost": "actual billed hours times user-confirmed hourly price; no GPU spending authorized by this manifest",
                   "stop": "OOM, source/model/hash/environment mismatch or timeout: preserve records, no policy changes or extra attempts"},
        "backup": "after process stops, archive entire run including both SQLite DBs, manifests, all numbered snapshots, failures, raw outputs and logs; SHA256 archive; copy to Windows and verify before user stops instance",
        "test_feedback_for_strategy_changes": False,
        "data_package": {"path": PACKAGE.as_posix(), "sha256": sha256(PACKAGE), "contains_images": False, "public_distribution": False},
        "dev_audit_sha256": None,
        "reported_dev_metrics_unverified": {
            "baseline": {"successes": 20, "n": 20, "hit10": .80, "mrr10": .5321},
            "v10_control": {"successes": 17, "n": 20, "hit10": .55},
            "format_only": {"successes": 14, "n": 20, "hit10": .55},
            "details_only": {"successes": 17, "n": 20, "hit10": .65},
            "format_details": {"successes": 20, "n": 20, "hit10": .90, "mrr10": .6655}},
        "dev_metrics": None,
        "selection_rule": "structured 20/20 success gate, then Hit@10, MRR@10, fewer generated tokens, fewer switches, format_only; baseline not a structured candidate",
        "scope_limits": ["not misinformation detection or fact verification", "same-image labels miss cross-image semantic relevance", "five captions are correlated", "model pretraining exposure unknown", "fixed-cohort bootstrap does not establish population-wide significance"]}
    write(FREEZE / "freeze-manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "manifest": str(FREEZE / "freeze-manifest.json"), "images_to_acquire": 100}))


if __name__ == "__main__":
    main()

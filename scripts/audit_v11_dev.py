"""Read-only five-arm dev audit. Never load a model or open test rankings.

--run must contain baseline/, v10_control/, format_only/, details_only/,
format_details/. All original files are archived locally, including failures.
Only compact, caption-free audit output should be version controlled.
"""

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visionevidence.batch import sha256
from visionevidence.caption_experiment import score_branch
from visionevidence.datasets.flickr8k import rows_for_split
from visionevidence.description import PROMPT as BASELINE_PROMPT
from visionevidence.observation import PROMPT, build_query, parse_observation
from visionevidence.observation_policy import ObservationPolicy
from visionevidence.retrieval import BM25Index, load_corpus

ARMS = {"baseline": None, "v10_control": ObservationPolicy(),
        "format_only": ObservationPolicy(True, False), "details_only": ObservationPolicy(False, True),
        "format_details": ObservationPolicy(True, True)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inventory(root):
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def audit_preliminary(path, raw, primary):
    """Earlier dev two-image paired smoke is archived, never pooled into selection."""
    comparison = read(path / "comparison-0001.json")
    require(comparison["is_mock"] is False and comparison["original_split"] == "dev", "preliminary source mismatch")
    require(comparison["scale"]["evaluation_images"] == 2, "unexpected preliminary cohort")
    expected = ["flickr8k:" + row["file_name"] for row in rows_for_split(raw, "dev")[:2]]
    branches = {}
    for mode in ("baseline", "structured"):
        branch = path / mode
        manifest = read(branch / "manifest.json")
        identity = manifest["identity"]
        for field in ("corpus_sha256", "vlm_config", "model_file_sha256", "source_sha256", "seed", "retrieval"):
            require(identity[field] == primary["baseline_identity"][field], "preliminary controls differ")
        require(manifest["git"]["commit"] == primary["dev_commit"] and manifest["git"]["dirty"] is False, "preliminary source drift")
        require(identity["effective_prompt"] == (BASELINE_PROMPT if mode == "baseline" else PROMPT), "preliminary prompt drift")
        for label in ("samples", "corpus", "qrels"):
            require(sha256(branch / f"input-{label}.jsonl") == identity[f"{label}_sha256"], "preliminary input changed")
        with closing(sqlite3.connect((branch / "state.sqlite3").resolve().as_uri()+"?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            require(db.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "preliminary SQLite corrupted")
            samples = list(db.execute("SELECT * FROM samples WHERE position<=2 ORDER BY position"))
            attempts = list(db.execute("SELECT * FROM attempts ORDER BY id"))
        require([s["id"] for s in samples] == expected and len(attempts) == 2, "preliminary counts mismatch")
        require(all(s["attempts"] == 1 for s in samples), "preliminary retries")
        index = BM25Index(load_corpus(branch / "input-corpus.jsonl"), k1=1.5, b=.75)
        for sample, attempt in zip(samples, attempts):
            response = json.loads(attempt["raw_response"])
            require(attempt["sample_id"] == sample["id"] == response["request_id"] and
                    attempt["backend_called"] == attempt["real_response"] == 1 and
                    response["is_mock"] is False and response["backend"] == "qwen3_vl" and
                    response["metadata"]["inference_performed"] is True, "preliminary response provenance mismatch")
            require(attempt["image_sha256"] == sample["image_sha256"] == sha256(raw / "dev" / sample["id"].removeprefix("flickr8k:")), "preliminary image mismatch")
            if sample["result"]:
                result = json.loads(sample["result"])
                query = response["description"] if mode == "baseline" else build_query(parse_observation(response["description"]))[0]
                require(query == result["query"] and response == result["vlm_response"], "preliminary query mismatch")
                replay = index.search(query, k=10)
                require([d.document_id for d in replay] == [d["document_id"] for d in result["evidence"]], "preliminary replay mismatch")
                require(all(math.isclose(a.score, b["score"], abs_tol=1e-10) for a,b in zip(replay, result["evidence"])), "preliminary score mismatch")
        scores = score_branch(branch, 2, (1, 5, 10))
        require(scores == comparison["scores"][mode], "preliminary comparison mismatch")
        branches[mode] = scores
    return {"images": 2, "calls": 4, "overlap_with_primary": "same dev first two; not additional independent images",
            "used_for_final_strategy_selection": False, "branches": branches}


def audit(run: Path, raw: Path):
    require(not list(run.rglob("*-wal")) and not list(run.rglob("*-journal")), "live SQLite sidecar")
    source = read(raw / "source.json")
    require(source["repository"] == "intro/flickr8k"
            and source["revision"] == "c7a02cb487da0eca729e133194b0f842cb578c08", "unpinned raw dataset source")
    require(all(sha256(raw / name) == info["sha256"] for name, info in source["files"].items()), "raw metadata changed")
    expected_ids = ["flickr8k:" + r["file_name"] for r in rows_for_split(raw, "dev")[:20]]
    before = inventory(run)
    branches, identities, manifests = {}, {}, {}
    for arm, policy in ARMS.items():
        path = run / arm
        manifest = read(path / "manifest.json")
        identity = manifest["identity"]
        config = identity["vlm_config"]
        require(config["backend"] == "qwen3_vl" and config["model_name"] == "Qwen/Qwen3-VL-4B-Instruct",
                f"{arm}: real Qwen3-VL required, Mock forbidden")
        require(config["max_new_tokens"] == 384 and config["do_sample"] is False and identity["seed"] == 0,
                f"{arm}: generation protocol mismatch")
        require(identity["model_file_sha256"] and any(k.endswith(".safetensors") for k in identity["model_file_sha256"]),
                f"{arm}: missing local model weight hashes")
        require(identity["retrieval"] == {"top_k": 10, "k1": 1.5, "b": .75}, "BM25 drift")
        require(identity["task"] == "image_to_caption_retrieval_proxy", "wrong task")
        require(identity["effective_prompt"] == (policy.prompt(PROMPT) if policy is not None else BASELINE_PROMPT),
                f"{arm}: prompt drift")
        require(identity["observation_policy"] == (policy.identity() if policy is not None else None), "policy drift")
        require(identity["query_mode"] == ("structured" if policy is not None else "baseline"), "query mode drift")
        for label in ("samples", "corpus", "qrels"):
            require(sha256(path / f"input-{label}.jsonl") == identity[f"{label}_sha256"], "input snapshot drift")
        with closing(sqlite3.connect((path / "state.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            require(db.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "corrupt SQLite")
            samples = list(db.execute("SELECT * FROM samples WHERE position<=20 ORDER BY position"))
            attempts = list(db.execute("SELECT * FROM attempts ORDER BY id"))
            sessions = list(db.execute("SELECT * FROM sessions"))
            qrels = {r[0]: json.loads(r[1]) for r in db.execute("SELECT sample_id,relevant_ids FROM qrels")}
        require([s["id"] for s in samples] == expected_ids, f"{arm}: dev prefix mismatch")
        require(all(json.loads(s["payload"])["split"] == "dev"
                    and json.loads(s["payload"])["dataset_version"] == source["revision"] for s in samples), "non-dev or wrong-version input")
        require(len(attempts) == 20 and all(s["attempts"] == 1 for s in samples), "unexpected retry/count")
        require(all(json.loads(s["settings"])["max_attempts"] == 1 for s in sessions), "retry budget changed")
        exported_attempts = [json.loads(line) for line in sorted(path.glob("attempts-*.jsonl"))[-1].read_text(encoding="utf-8").splitlines()]
        expected_attempts = [{**dict(a), "raw_response": json.loads(a["raw_response"]) if a["raw_response"] else None} for a in attempts]
        require(exported_attempts == expected_attempts, "attempt export/SQLite mismatch")
        exported_results = [json.loads(line) for line in sorted(path.glob("results-*.jsonl"))[-1].read_text(encoding="utf-8").splitlines()]
        require(exported_results == [json.loads(s["result"]) for s in samples if s["result"]], "result export/SQLite mismatch")
        runtimes = [read(p) for p in sorted(path.glob("runtime-*.json"))]
        require(runtimes and all(r == runtimes[0] for r in runtimes), "runtime snapshots missing or changed")
        require(runtimes[0]["gpu_name"] == "NVIDIA GeForce RTX 4090 D" and runtimes[0]["seed"] == 0, "GPU/seed mismatch")
        by_id = {a["sample_id"]: a for a in attempts}
        require(len(by_id) == 20 and set(by_id) == set(expected_ids), "missing/duplicate attempts")
        documents = load_corpus(path / "input-corpus.jsonl")
        require(len(documents) == 5000, "candidate pool must have 5000 captions")
        expected_documents = {f"flickr8k:{row['file_name']}#{i}": row[f"caption_{i}"]
                              for row in rows_for_split(raw, "dev") for i in range(5)}
        require({d.document_id: d.text for d in documents} == expected_documents, "candidate text/source mismatch")
        snapshot_qrels = [json.loads(line) for line in (path / "input-qrels.jsonl").read_text(encoding="utf-8").splitlines()]
        require(all(row["kind"] == "caption_pair_proxy" and qrels[row["sample_id"]] == row["relevant_ids"]
                    for row in snapshot_qrels) and len(snapshot_qrels) == len(qrels), "DB/snapshot qrels mismatch")
        index = BM25Index(documents, k1=1.5, b=.75)
        categories, details, tokens, latencies, hashes = Counter(), [], [], [], []
        schema_count = contract_count = replayed = 0
        for sample in samples:
            sid = sample["id"]
            attempt = by_id[sid]
            require(attempt["backend_called"] == attempt["real_response"] == 1, "not a real response")
            response = json.loads(attempt["raw_response"])
            meta = response["metadata"]
            require(response["backend"] == "qwen3_vl" and response["is_mock"] is False
                    and meta["inference_performed"] is True and response["request_id"] == sid, "response provenance mismatch")
            require(response["model_name"] == config["model_name"] and meta["max_new_tokens"] == 384
                    and meta["do_sample"] is False and meta["dtype"] == config["dtype"], "response config mismatch")
            image_hash = sha256(raw / "dev" / sid.removeprefix("flickr8k:"))
            require(attempt["image_sha256"] == sample["image_sha256"] == image_hash, "dev image hash mismatch")
            hashes.append(image_hash)
            require(set(qrels[sid]) == {f"{sid}#{i}" for i in range(5)}, "qrels association mismatch")
            tokens.append(meta["generated_tokens"])
            latencies.append(response["latency_ms"])
            query = None
            category = "accepted"
            if policy is not None:
                try:
                    observation = parse_observation(response["description"])
                    schema_count += 1
                    policy.validate(observation)
                    contract_count += 1
                    query = build_query(observation)[0]
                except ValueError as exc:
                    error = str(exc)
                    category = ("json_syntax" if error.startswith("invalid observation JSON") else
                                "format_contract" if error.startswith("v11 format") else
                                "query" if error.startswith("observation has no supported") else "schema")
            else:
                query = response["description"]
                if not query.strip() or len(query) > 8000:
                    category = "query"
            if meta["generated_tokens"] >= 384:
                category = "token_limit"
            categories[category] += 1
            require((category == "accepted") == (sample["status"] == "completed"), f"{arm}: failure classification mismatch {sid}")
            if sample["status"] == "completed":
                result = json.loads(sample["result"])
                require(result["query"] == query and result["vlm_response"] == response, "saved query/response mismatch")
                computed = index.search(query, k=10)
                require([d.document_id for d in computed] == [d["document_id"] for d in result["evidence"]], "ranking replay mismatch")
                require(all(math.isclose(a.score, b["score"], abs_tol=1e-10) for a, b in zip(computed, result["evidence"])), "score mismatch")
                replayed += 1
            else:
                require(sample["status"] == attempt["status"] == "failed" and not sample["result"], "invalid failed state")
                details.append({"sample_id": sid, "category": category, "error_type": attempt["error_type"],
                                "error": attempt["error"], "attempt_id": attempt["id"]})
        score = score_branch(path, 20, (1, 5, 10))
        summary = read(sorted(path.glob("summary-*.json"))[-1])
        require(summary["is_mock"] is False and summary["attempts"] == 20, "summary provenance/count mismatch")
        require(summary["sample_status"].get("completed", 0) == replayed, "summary success mismatch")
        require(math.isclose(summary["retrieval_metrics"]["mrr_at_k"], score["metrics"]["10"]["mrr"], abs_tol=1e-12), "summary MRR mismatch")
        require(math.isclose(summary["retrieval_metrics"]["recall_at_k"], score["metrics"]["10"]["caption_recall"], abs_tol=1e-12), "summary recall mismatch")
        branches[arm] = {"scores": score, "successes": replayed, "failures": 20-replayed,
            "categories": dict(categories), "failure_records": details, "generated_tokens_sum": sum(tokens),
            "latency_ms_mean": sum(latencies)/20, "schema_successes": schema_count if policy is not None else None,
            "contract_successes": contract_count if policy is not None else None,
            "image_sha256": hashes, "real_response_count": 20, "cpu_ranking_replays": replayed}
        branches[arm]["runtime"] = runtimes[0]
        identities[arm], manifests[arm] = identity, manifest
    base = identities["baseline"]
    for arm, identity in identities.items():
        for field in ("samples_sha256", "corpus_sha256", "qrels_sha256", "vlm_config", "model_file_sha256", "source_sha256", "seed", "retrieval"):
            require(identity[field] == base[field], f"cross-arm control mismatch: {arm}/{field}")
    commits = {m["git"]["commit"] for m in manifests.values()}
    require(len(commits) == 1 and all(m["git"]["dirty"] is False for m in manifests.values()), "nonuniform or dirty dev source")
    commit = commits.pop()
    for name, expected in base["source_sha256"].items():
        blob = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "show", f"{commit}:src/{name}"],
                              cwd=ROOT, capture_output=True, check=True).stdout
        require(hashlib.sha256(blob).hexdigest() == expected, f"source not identical to committed dev implementation: {name}")
    eligible = [a for a in ARMS if a != "baseline" and branches[a]["successes"] == 20]
    require(eligible, "no structured candidate passes preregistered reliability gate")
    chosen = sorted(eligible, key=lambda a: (-branches[a]["scores"]["metrics"]["10"]["hit_rate"],
        -branches[a]["scores"]["metrics"]["10"]["mrr"], branches[a]["generated_tokens_sum"],
        int(ARMS[a].format_constraints) + int(ARMS[a].retain_visual_details), a != "format_only", a))[0]
    require(chosen == "format_details", "audited selection differs from expected candidate; do not freeze silently")
    require(inventory(run) == before, "audit modified input")
    require(all(branches[a]["runtime"] == branches["baseline"]["runtime"] for a in ARMS), "cross-arm runtime mismatch")
    return {"status": "audited", "task": "image_to_caption_retrieval_proxy", "selected_strategy": chosen,
            "dev_commit": commit, "source_file_sha256": before, "branches": branches,
            "selected_identity": identities[chosen], "baseline_identity": base,
            "environment": manifests[chosen]["environment"], "gpu_calls_during_audit": 0,
            "provenance_limit": "internally consistent saved real-backend records and committed code; not independent hardware attestation"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Untouched transport archive; verifies every sibling dev run byte")
    parser.add_argument("--preliminary", type=Path, help="Earlier two-image dev paired smoke, preserved separately")
    args = parser.parse_args(argv)
    require(not args.output.resolve().is_relative_to(args.run.resolve()), "output must be outside original run")
    require(not args.output.exists(), "output must be new")
    value = audit(args.run, args.raw)
    if args.preliminary:
        preliminary_before = inventory(args.preliminary)
        value["preliminary"] = audit_preliminary(args.preliminary, args.raw, value)
        require(inventory(args.preliminary) == preliminary_before, "preliminary audit changed original files")
    if args.archive:
        source_files = inventory(args.run.parent)
        with tarfile.open(args.archive) as stream:
            members = [m for m in stream.getmembers() if m.isfile()]
            require({m.name for m in members} == set(source_files), "transport/extracted file list mismatch")
            for member in members:
                with stream.extractfile(member) as source:
                    require(hashlib.file_digest(source, "sha256").hexdigest() == source_files[member.name], "transport byte mismatch")
        value["transport"] = {"sha256": sha256(args.archive), "files": source_files,
                              "file_count": len(source_files), "all_members_verified": True}
    args.output.mkdir(parents=True, exist_ok=False)
    archive = args.output / "dev-originals.tar.gz"
    with tarfile.open(archive, "x:gz") as stream:
        stream.add(args.run, arcname="dev-originals")
    require(inventory(args.run) == value["source_file_sha256"], "source changed during archival")
    value["archive_sha256"] = sha256(archive)
    if args.archive:
        shutil.copyfile(args.archive, args.output / "all-dev-originals.tar.gz")
        require(sha256(args.output / "all-dev-originals.tar.gz") == value["transport"]["sha256"], "archive copy changed")
    with (args.output / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": value["status"], "selected_strategy": value["selected_strategy"], "output": str(args.output)}))


if __name__ == "__main__":
    main()

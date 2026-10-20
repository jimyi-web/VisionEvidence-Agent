"""Read-only, CPU-only audit of a frozen Flickr8k GPU run; never regenerates queries.

Run with the project's Python. Outputs must be a new directory outside the run.
Full responses and caption text stay in ignored outputs, not the public report.
"""

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visionevidence.observation import build_query, parse_observation
from visionevidence.retrieval import BM25Index, load_corpus, tokenize


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def snapshot(run: Path) -> dict:
    return {p.relative_to(run).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(run.rglob("*")) if p.is_file()}


def metrics(ranked: list[str], relevant: frozenset[str]) -> dict:
    result = {}
    for k in (1, 5, 10):
        top = ranked[:k]
        rr = next((1 / i for i, doc in enumerate(top, 1) if doc in relevant), 0.0)
        result[str(k)] = {"hit_rate": float(rr > 0), "mrr": rr,
                          "caption_recall": len(set(top) & relevant) / len(relevant)}
    return result


def aggregate(rows):
    assert rows, "empty metric cohort"
    return {"denominator": len(rows), "metrics": {
        str(k): {key: sum(r["metrics"][str(k)][key] for r in rows) / len(rows)
                 for key in ("hit_rate", "mrr", "caption_recall")} for k in (1, 5, 10)}}


def check_metrics(actual, expected):
    for k, values in actual.items():
        for key, value in values.items():
            assert math.isclose(value, expected[k][key], abs_tol=1e-12), (k, key, value, expected)


def strict_json(text):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError(f"duplicate key: {key}")
            obj[key] = value
        return obj

    def constant(value):
        raise ValueError(f"non-JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def response_checks(raw: dict) -> dict:
    text, meta = raw["description"], raw["metadata"]
    row = {"generated_tokens": meta["generated_tokens"], "max_new_tokens": meta["max_new_tokens"],
           "token_limit_reached": meta["generated_tokens"] >= meta["max_new_tokens"],
           "json_valid": False, "schema_valid": False, "query_valid": False}
    try:
        obj = strict_json(text)
        row["json_valid"] = True
        if isinstance(obj, dict):
            row["field_lengths"] = {k: len(v) if isinstance(v, (list, str, dict)) else None
                                    for k, v in obj.items()}
    except (ValueError, RecursionError) as exc:
        row["json_error"] = str(exc)
    try:
        observation = parse_observation(text)
        row["schema_valid"] = True
        build_query(observation)
        row["query_valid"] = True
    except ValueError as exc:
        row["schema_or_query_error"] = str(exc)
    # Root category and symptom are separate: truncation also causes invalid JSON.
    row["category"] = ("token_limit" if row["token_limit_reached"] else
                       "json_syntax" if not row["json_valid"] else
                       "schema" if not row["schema_valid"] else
                       "query" if not row["query_valid"] else "accepted")
    return row


def distribution(values):
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "max": max(values), "sum": sum(values)}


def contributions(query, document, frequencies, size, average_length, k1, b):
    counts = Counter(tokenize(document.text))
    result = {}
    for term in sorted(set(tokenize(query)) & counts.keys()):
        df, tf = frequencies[term], counts[term]
        idf = math.log1p((size - df + 0.5) / (df + 0.5))
        result[term] = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * sum(counts.values()) / average_length))
    assert math.isclose(sum(result.values()), document.score, abs_tol=1e-10)
    return dict(sorted(result.items(), key=lambda item: (-item[1], item[0])))


def audit(run: Path, archive: Path, output: Path) -> None:
    run, archive, output = run.resolve(), archive.resolve(), output.resolve()
    assert run.is_dir() and archive.is_file()
    assert not output.is_relative_to(run), "analysis must be outside frozen run"
    if output.exists():
        raise FileExistsError(f"audit output already exists: {output}")
    assert not list(run.rglob("*-wal")) and not list(run.rglob("*-journal")), "live SQLite sidecar"
    before, archive_hash = snapshot(run), digest(archive)
    # Verify every extracted byte against the untouched transport archive.
    with tarfile.open(archive, "r:gz") as bundle:
        prefix = "outputs/" + run.name + "/"
        members = [m for m in bundle.getmembers() if m.isfile()]
        assert len(members) == len(before)
        for member in members:
            assert member.name.startswith(prefix)
            relative = member.name[len(prefix):]
            with bundle.extractfile(member) as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == before[relative]["sha256"]
    comparison = read_json(run / "comparison-0002.json")
    assert comparison["task"] == "image_to_caption_retrieval_proxy" and not comparison["is_mock"]
    limit = comparison["scale"]["evaluation_images"]
    manifests = {mode: read_json(run / mode / "manifest.json") for mode in ("baseline", "structured")}
    controls = ("samples_sha256", "corpus_sha256", "qrels_sha256", "retrieval", "vlm_config",
                "model_file_sha256", "source_sha256", "seed")
    for key in controls:
        assert manifests["baseline"]["identity"][key] == manifests["structured"]["identity"][key], key
    for name, expected in manifests["baseline"]["identity"]["source_sha256"].items():
        assert digest(ROOT / "src" / name) == expected, f"audit code drift: {name}"
    documents = load_corpus(run / "baseline" / "input-corpus.jsonl")
    lookup = {doc.document_id: doc for doc in documents}
    retrieval = manifests["baseline"]["identity"]["retrieval"]
    index = BM25Index(documents, k1=retrieval["k1"], b=retrieval["b"])
    frequencies = Counter(term for doc in documents for term in set(tokenize(doc.text)))
    cohorts, branches, failures = {}, {}, []
    verified_searches = 0
    for mode in ("baseline", "structured"):
        branch = run / mode
        for key in ("samples", "corpus", "qrels"):
            assert digest(branch / f"input-{key}.jsonl") == manifests[mode]["identity"][f"{key}_sha256"]
        summary = read_json(branch / "summary-0002.json")
        assert summary == comparison["branches"][mode]
        exported = read_jsonl(branch / "results-0002.jsonl")
        with closing(sqlite3.connect((branch / "state.sqlite3").as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
            db.row_factory = sqlite3.Row
            assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            samples = [dict(r) for r in db.execute("SELECT * FROM samples WHERE position<=? ORDER BY position", (limit,))]
            attempts = [dict(r) for r in db.execute("SELECT a.* FROM attempts a JOIN samples s ON s.id=a.sample_id WHERE s.position<=? ORDER BY a.id", (limit,))]
            qrels = {r["sample_id"]: frozenset(json.loads(r["relevant_ids"])) for r in db.execute("SELECT * FROM qrels")}
            qrel_snapshot = read_jsonl(branch / "input-qrels.jsonl")
            assert all(qrels[r["sample_id"]] == frozenset(r["relevant_ids"]) for r in qrel_snapshot)
            sessions = [{"id": r["id"], "started": r["started"], "finished": r["finished"],
                         "settings": {k: v for k, v in json.loads(r["settings"]).items() if k != "git"}}
                        for r in db.execute("SELECT * FROM sessions ORDER BY id")]
            assert json.loads(db.execute("SELECT summary FROM sessions WHERE id=2").fetchone()[0]) == summary
        assert len(samples) == limit and len(attempts) == summary["attempts"]
        assert dict(Counter(a["error_type"] for a in attempts if a["error_type"])) == summary["error_counts"]
        assert len({a["sample_id"] for a in attempts}) == len(attempts), "multiple attempts require explicit audit extension"
        by_id = {a["sample_id"]: a for a in attempts}
        assert exported == [json.loads(s["result"]) for s in samples if s["status"] == "completed"]
        rows = []
        for sample in samples:
            sid = sample["id"]
            attempt = by_id[sid]
            raw = json.loads(attempt["raw_response"])
            assert attempt["backend_called"] == attempt["real_response"] == 1
            assert raw["backend"] == "qwen3_vl" and raw["is_mock"] is False
            assert raw["metadata"]["inference_performed"] is True and raw["request_id"] == sid
            assert attempt["image_sha256"] == sample["image_sha256"]
            relevant = qrels[sid]
            assert len(relevant) == 5 and relevant <= lookup.keys()
            result = json.loads(sample["result"]) if sample["result"] else None
            ranked = [d["document_id"] for d in result["evidence"]] if result else []
            row = {"position": sample["position"], "sample_id": sid, "status": sample["status"],
                   "attempt_id": attempt["id"], "image_sha256": sample["image_sha256"],
                   "metrics": metrics(ranked, relevant), "raw_response": raw,
                   "target_captions": [{"id": doc, "text": lookup[doc].text} for doc in sorted(relevant)],
                   "error_type": attempt["error_type"], "error": attempt["error"]}
            if mode == "structured":
                row["response_checks"] = response_checks(raw)
                assert (row["response_checks"]["category"] == "accepted") == (sample["status"] == "completed")
                if sample["status"] == "failed":
                    failures.append({"sample_id": sid, "attempt_id": attempt["id"], "error_type": attempt["error_type"],
                                     "error": attempt["error"], "checks": row["response_checks"], "raw_response": raw})
            if result:
                assert result["vlm_response"] == raw
                query = result["query"]
                if mode == "structured":
                    assert build_query(parse_observation(raw["description"]))[0] == query
                else:
                    assert query == raw["description"]
                # Only unchanged original queries are replayed, on CPU, without VLM calls.
                replay = index.search(query, k=len(documents))
                assert ranked == [d.document_id for d in replay[:10]]
                for saved, computed in zip(result["evidence"], replay):
                    assert math.isclose(saved["score"], computed.score, abs_tol=1e-10)
                verified_searches += 1
                first = next(((i, doc) for i, doc in enumerate(replay, 1) if doc.document_id in relevant), None)
                target_terms = {t for doc in relevant for t in tokenize(lookup[doc].text)}
                terms = set(tokenize(query))
                row.update(query=query, unique_query_terms=sorted(terms), query_token_count=len(tokenize(query)),
                           target_overlap_terms=sorted(terms & target_terms),
                           non_target_terms=sorted(terms - target_terms),
                           first_relevant_rank_full=first[0] if first else None,
                           observation=result.get("observation"), query_sources=result["query_sources"],
                           top10=result["evidence"])
                for name, doc in (("top1", replay[0]), ("best_target", first[1] if first else None)):
                    if doc:
                        row[name] = {"document_id": doc.document_id, "text": doc.text, "score": doc.score,
                                     "term_contributions": contributions(query, doc, frequencies, len(documents),
                                                                          index.average_length, index.k1, index.b)}
            else:
                assert sample["status"] == "failed" and attempt["status"] == "failed"
            expected = next(p for p in comparison["scores"][mode]["per_query"] if p["sample_id"] == sid)
            assert expected["status"] == row["status"]
            check_metrics(row["metrics"], expected["metrics"])
            rows.append(row)
        cohorts[mode] = rows
        main = aggregate(rows)
        check_metrics(main["metrics"], comparison["scores"][mode]["metrics"])
        assert math.isclose(main["metrics"]["10"]["mrr"], summary["retrieval_metrics"]["mrr_at_k"], abs_tol=1e-12)
        assert math.isclose(main["metrics"]["10"]["caption_recall"], summary["retrieval_metrics"]["recall_at_k"], abs_tol=1e-12)
        branches[mode] = {"all_selected": main, "success_only": aggregate([r for r in rows if r["status"] == "completed"]),
                          "status_counts": dict(Counter(r["status"] for r in rows)), "attempt_count": len(attempts),
                          "exported_results_count": len(exported), "sessions": sessions,
                          "latency_ms_all_responses": distribution([r["raw_response"]["latency_ms"] for r in rows]),
                          "generated_tokens": distribution([r["raw_response"]["metadata"]["generated_tokens"] for r in rows]),
                          "peak_gpu_allocated_mib": distribution([r["raw_response"]["metadata"]["peak_gpu_memory_mb"] for r in rows]),
                          "model_loaded_this_call": sum(r["raw_response"]["metadata"]["model_loaded_this_call"] for r in rows),
                          "unique_query_terms_success": distribution([len(r["unique_query_terms"]) for r in rows if "query" in r])}
    common = {r["sample_id"] for r in cohorts["structured"] if r["status"] == "completed"}
    for mode in branches:
        branches[mode]["common_success"] = aggregate([r for r in cohorts[mode] if r["sample_id"] in common])
        branches[mode]["structured_failed_cohort"] = aggregate([r for r in cohorts[mode] if r["sample_id"] not in common])
    paired = []
    for base, structured in zip(cohorts["baseline"], cohorts["structured"]):
        assert base["sample_id"] == structured["sample_id"] and base["image_sha256"] == structured["image_sha256"]
        row = {"position": base["position"], "sample_id": base["sample_id"], "baseline": base, "structured": structured,
               "mrr10_delta": structured["metrics"]["10"]["mrr"] - base["metrics"]["10"]["mrr"]}
        if structured["status"] == "completed":
            bt, st = set(base["unique_query_terms"]), set(structured["unique_query_terms"])
            target = {t for doc in base["target_captions"] for t in tokenize(doc["text"])}
            row.update(baseline_only_terms=sorted(bt - st), structured_only_terms=sorted(st - bt),
                       baseline_only_target_terms=sorted((bt - st) & target),
                       structured_only_non_target_terms=sorted((st - bt) - target))
        paired.append(row)
    checks = [r["response_checks"] for r in cohorts["structured"]]
    audit_result = {"task": comparison["task"], "verified": False, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "archive_sha256": archive_hash, "run": str(run), "source_files": before,
                    "scale": comparison["scale"], "branches": branches,
                    "structured_format": {"responses": len(checks),
                        "json_valid": sum(c["json_valid"] for c in checks),
                        "schema_valid": sum(c["schema_valid"] for c in checks),
                        "query_valid": sum(c["query_valid"] for c in checks),
                        "categories": dict(Counter(c["category"] for c in checks))},
                    "paired_mrr10_outcomes_all": dict(Counter("structured_better" if r["mrr10_delta"] > 0 else
                        "baseline_better" if r["mrr10_delta"] < 0 else "tied" for r in paired)),
                    "paired_mrr10_outcomes_common": dict(Counter("structured_better" if r["mrr10_delta"] > 0 else
                        "baseline_better" if r["mrr10_delta"] < 0 else "tied" for r in paired if r["sample_id"] in common)),
                    "failures": failures,
                    "validation": {"archive_members_match": True, "sqlite_integrity": "ok in both branches",
                        "local_source_hashes_match": True, "comparison_summary_db_exports_match": True,
                        "input_snapshot_hashes_match": True, "qrels_snapshot_db_match": True,
                        "original_queries_replayed_cpu": verified_searches, "gpu_calls_during_audit": 0,
                        "unchanged_after_audit": snapshot(run) == before and digest(archive) == archive_hash}}
    assert audit_result["validation"]["unchanged_after_audit"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "audit.json", audit_result)
    with (output / "paired-samples.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in paired:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(output / "failures.json", failures)
    print(json.dumps({"output": str(output), "validation": audit_result["validation"],
                      "structured_format": audit_result["structured_format"],
                      "branches": {k: {s: v[s] for s in ("all_selected", "common_success", "structured_failed_cohort")}
                                   for k, v in branches.items()}}, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.archive, args.output)

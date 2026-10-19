"""Resumable single-worker observation/retrieval runs; SQLite is the durable journal.

The batch size bounds scheduling, not GPU tensor batching. No labels enter queries.
"""

import argparse
from contextlib import closing, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sqlite3
import shutil
import subprocess
import sys
from time import perf_counter

from visionevidence.observation import PROMPT, PROMPT_VERSION, observe_and_retrieve
from visionevidence.evaluation.retrieval import metrics_at_k
from visionevidence.retrieval.bm25 import BM25Index
from visionevidence.retrieval.corpus import load_corpus, read_jsonl
from visionevidence.vlm.factory import VLMConfig, create_vlm, load_vlm_config
from visionevidence.vlm.schemas import VLMRequest, VLMResponse


def dumps(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        result = subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}", *args],
                                cwd=root, capture_output=True, text=True, check=True, timeout=10)
        return result.stdout.strip()
    try:
        commit, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    sources = {p.relative_to(root / "src").as_posix(): sha256(p)
               for p in sorted((root / "src" / "visionevidence").rglob("*.py"))}
    return {"commit": commit, "dirty": dirty, "source_sha256": sources}


@contextmanager
def run_lock(path):
    """OS releases the advisory lock after a crash; lock file is not a stale flag."""
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("run directory is already locked by another worker") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class StructuredMock:
    """Fixed synthetic observation; never derives visual content from a caption."""
    def generate(self, request):
        request.validate_image()
        if request.prompt != PROMPT:
            return VLMResponse(request.request_id, "[MOCK] A road sign. No image analysis performed.",
                               "structured-mock-v1", "mock", True, 0.0,
                               {"inference_performed": False, "verified": False,
                                "content_source": "fixed_synthetic_description_fixture"})
        return VLMResponse(request.request_id, dumps({"objects": ["road sign"],
                           "visible_text": [], "uncertainties": ["[MOCK] no image analysis"]}),
                           "structured-mock-v1", "mock", True, 0.0,
                           {"inference_performed": False, "verified": False,
                            "content_source": "fixed_synthetic_structured_fixture"})


class Capture:
    def __init__(self, backend, expected):
        self.backend, self.expected = backend, expected
        self.response = None
        self.called = False

    def generate(self, request):
        self.called = True
        self.response = self.backend.generate(request)
        if self.response.backend != self.expected or self.response.is_mock != (self.expected == "mock"):
            raise ValueError("backend provenance mismatch; refusing mixed mock/real results")
        return self.response


def validate_sample(row: dict) -> None:
    required = {"sample_id", "image_path", "claim", "dataset", "dataset_version", "split", "source"}
    allowed = required | {"metadata", "pair_label"}
    if required - row.keys() or row.keys() - allowed:
        raise ValueError(f"sample fields missing={sorted(required-row.keys())}, unknown={sorted(row.keys()-allowed)}")
    for field in required:
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError(f"{field} must be nonempty text")
    raw = row["image_path"]
    if "://" in raw or raw.startswith(("//", "\\\\")):
        raise ValueError("image_path must be local")
    if "metadata" in row and not isinstance(row["metadata"], dict):
        raise ValueError("metadata must be an object")


def initialize(db, samples):
    db.executescript("""
      CREATE TABLE IF NOT EXISTS samples (
        position INTEGER UNIQUE, id TEXT PRIMARY KEY, payload TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
        image_sha256 TEXT, result TEXT);
      CREATE TABLE IF NOT EXISTS attempts (
        id INTEGER PRIMARY KEY, sample_id TEXT NOT NULL, started TEXT NOT NULL,
        finished TEXT, status TEXT NOT NULL, error_type TEXT, error TEXT,
        raw_response TEXT, backend_called INTEGER NOT NULL DEFAULT 0,
        real_response INTEGER NOT NULL DEFAULT 0, image_sha256 TEXT);
      CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY, started TEXT, finished TEXT, settings TEXT, summary TEXT);
    """)
    if db.execute("SELECT count(*) FROM samples").fetchone()[0]:
        return
    with db:
        for line, row in read_jsonl(samples):
            validate_sample(row)
            try:
                db.execute("INSERT INTO samples(position,id,payload) VALUES(?,?,?)",
                           (line, row["sample_id"], dumps(row)))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"duplicate sample_id at line {line}: {row['sample_id']}") from exc
        if not db.execute("SELECT count(*) FROM samples").fetchone()[0]:
            raise ValueError("samples must not be empty")


def summary(db, limit, documents, backend, top_k):
    selected = "SELECT id FROM samples ORDER BY position LIMIT ?"
    counts = dict(db.execute(f"SELECT status,count(*) FROM samples WHERE id IN ({selected}) GROUP BY status", (limit,)))
    attempts = db.execute(f"SELECT count(*),coalesce(sum(backend_called),0),coalesce(sum(real_response),0) "
                          f"FROM attempts WHERE sample_id IN ({selected})", (limit,)).fetchone()
    real_images = db.execute(f"SELECT count(DISTINCT a.image_sha256) FROM samples s JOIN attempts a "
                            f"ON s.id=a.sample_id WHERE a.real_response=1 AND s.id IN ({selected})", (limit,)).fetchone()[0]
    errors = dict(db.execute(f"SELECT coalesce(error_type,'Interrupted'),count(*) FROM attempts "
                            f"WHERE status!='completed' AND sample_id IN ({selected}) GROUP BY error_type", (limit,)))
    evaluated = skipped = unannotated = 0
    recall = rr = 0.0
    for result, labels in db.execute("SELECT s.result,q.relevant_ids FROM samples s LEFT JOIN qrels q ON s.id=q.sample_id WHERE s.position<=?", (limit,)):
        if labels is None:
            unannotated += 1
            continue
        relevant = frozenset(json.loads(labels))
        if not relevant:
            skipped += 1
            continue
        ranked = [x["document_id"] for x in json.loads(result)["evidence"]] if result else []
        r, m = metrics_at_k(ranked, relevant, top_k)
        recall += r
        rr += m
        evaluated += 1
    return {"status": "completed" if not (counts.get("failed", 0) or counts.get("pending", 0)) else "incomplete",
            "backend": backend, "is_mock": backend == "mock", "verified": False,
            "scale": {"evidence_documents": documents, "evaluation_queries": sum(counts.values()),
                      "successful_queries": counts.get("completed", 0),
                      "real_vlm_image_calls": attempts[1] if backend == "qwen3_vl" else 0,
                      "real_vlm_responses": attempts[2], "unique_real_vlm_images": real_images},
            "sample_status": counts, "attempts": attempts[0], "error_counts": errors,
            "retrieval_metrics": {"recall_at_k": recall/evaluated, "mrr_at_k": rr/evaluated,
                                  "k": top_k, "evaluated_queries": evaluated,
                                  "failure_policy": "annotated failed/pending queries count as zero"} if evaluated else None,
            "qrels_counts": {"evaluated": evaluated, "empty_excluded": skipped, "unannotated": unannotated},
            "qrels_status": "explicit relevance annotations only; pair labels are never consumed",
            "real_call_scope": "calls entered backend, including failures; interrupted call completion may be unknown"}


def run_batch(*, samples: Path, corpus: Path, output: Path, config: VLMConfig | None = None,
              limit: int = 100, batch_size: int = 32, max_attempts: int = 2,
              resume: bool = False, top_k: int = 3, k1: float = 1.5, b: float = 0.75,
              qrels: Path | None = None, backend_factory=None, progress: bool = False,
              query_mode: str = "structured", task: str = "evidence_retrieval") -> dict:
    """Fixed dataset, growing prefix. max_attempts is a lifetime per-sample budget.

    Increasing it explicitly on resume permits recovery after repairing missing images.
    Successful results cannot be replaced; changed inputs/config/code require a new run.
    """
    for name, value in (("limit", limit), ("batch_size", batch_size), ("max_attempts", max_attempts), ("top_k", top_k)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    samples, corpus, output = Path(samples).resolve(), Path(corpus).resolve(), Path(output).resolve()
    config = config or VLMConfig()
    if query_mode not in {"structured", "baseline"}:
        raise ValueError("query_mode must be structured or baseline")
    if task not in {"evidence_retrieval", "image_to_caption_retrieval_proxy"}:
        raise ValueError("unsupported retrieval task")
    from visionevidence.description import PROMPT as BASELINE_PROMPT, PROMPT_VERSION as BASELINE_VERSION, describe_and_retrieve
    effective_prompt = PROMPT if query_mode == "structured" else BASELINE_PROMPT
    prompt_version = PROMPT_VERSION if query_mode == "structured" else BASELINE_VERSION
    processor = observe_and_retrieve if query_mode == "structured" else describe_and_retrieve
    if config.do_sample:
        raise ValueError("batch v1 requires deterministic do_sample=false")
    index = BM25Index(load_corpus(corpus), k1=k1, b=b)
    code = provenance()
    # Hash local model contents, including weights, once per invocation: revision is not guessed.
    model_files = None
    if config.backend == "qwen3_vl":
        if config.model_path is None or not Path(config.model_path).is_dir():
            raise ValueError("real mode requires existing local model_path; no downloads")
        model_files = {p.relative_to(config.model_path).as_posix(): sha256(p)
                       for p in sorted(Path(config.model_path).rglob("*")) if p.is_file()}
        if not model_files:
            raise ValueError("model directory is empty")
    effective_config = asdict(config)
    effective_config["model_path"] = str(config.model_path) if config.model_path is not None else None
    identity = {"schema_version": 1, "samples_sha256": sha256(samples), "corpus_sha256": sha256(corpus),
                "qrels_sha256": sha256(Path(qrels)) if qrels else None,
                "samples_path": str(samples), "corpus_path": str(corpus),
                "vlm_config": effective_config,
                "effective_prompt": effective_prompt, "effective_prompt_version": prompt_version,
                "query_mode": query_mode, "task": task,
                "source_sha256": code["source_sha256"], "model_file_sha256": model_files,
                "retrieval": {"top_k": top_k, "k1": k1, "b": b}, "seed": 0}
    if resume:
        if not (output / "manifest.json").is_file() or not (output / "state.sqlite3").is_file():
            raise ValueError("resume requires an existing initialized run")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with run_lock(output / ".lock"):
        manifest_path = output / "manifest.json"
        if resume:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["identity"] != identity:
                raise ValueError("input/config/model/source fingerprint changed; create a new run")
        else:
            dependencies = {}
            for name in ("visionevidence-agent", "torch", "transformers", "Pillow"):
                try:
                    dependencies[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    dependencies[name] = None
            manifest = {"created_utc": now(), "identity": identity, "git": code,
                        "environment": {"python": sys.version, "executable": sys.executable,
                                        "platform": platform.platform(), "processor": platform.processor(),
                                        "dependencies": dependencies},
                        "model_revision": "local content hashes" if model_files else None,
                        "device": "cpu" if config.backend == "mock" else config.device}
            manifest_path.write_text(dumps(manifest) + "\n", encoding="utf-8")
        for label, path in (("samples", samples), ("corpus", corpus), ("qrels", Path(qrels) if qrels else None)):
            if path:
                target = output / f"input-{label}.jsonl"
                if not resume:
                    shutil.copyfile(path, target)
                if sha256(target) != identity[f"{label}_sha256"]:
                    raise ValueError(f"{label} snapshot fingerprint changed")
        with closing(sqlite3.connect(output / "state.sqlite3")) as db:
            db.execute("PRAGMA synchronous=FULL")
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("SQLite integrity check failed")
            initialize(db, output / "input-samples.jsonl")
            with db:
                db.execute("CREATE TABLE IF NOT EXISTS qrels(sample_id TEXT PRIMARY KEY,relevant_ids TEXT,annotation_source TEXT,kind TEXT)")
                if qrels and not db.execute("SELECT count(*) FROM qrels").fetchone()[0]:
                    for _, row in read_jsonl(qrels):
                        if row.keys() != {"sample_id", "relevant_ids", "annotation_source", "kind"}:
                            raise ValueError("qrels require sample_id,relevant_ids,annotation_source,kind")
                        permitted = ("caption_pair_proxy",) if task == "image_to_caption_retrieval_proxy" else ("human_relevance", "synthetic_relevance")
                        if row["kind"] not in permitted:
                            raise ValueError("pair/classification labels are not retrieval relevance judgments")
                        ids = row["relevant_ids"]
                        if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
                            raise ValueError("relevant_ids must be a string array")
                        if len(ids) != len(set(ids)) or set(ids) - index.document_ids:
                            raise ValueError("duplicate or unknown evidence IDs in qrels")
                        if not isinstance(row["annotation_source"], str) or not row["annotation_source"].strip():
                            raise ValueError("qrels require a documented annotation source")
                        if not db.execute("SELECT 1 FROM samples WHERE id=?", (row["sample_id"],)).fetchone():
                            raise ValueError("qrels sample absent from dataset")
                        try:
                            db.execute("INSERT INTO qrels VALUES(?,?,?,?)", (row["sample_id"], dumps(ids), row["annotation_source"], row["kind"]))
                        except sqlite3.IntegrityError as exc:
                            raise ValueError("duplicate qrels sample_id") from exc
            # A committed running attempt means the previous process was interrupted.
            with db:
                db.execute("UPDATE attempts SET status='interrupted',error_type='Interrupted',finished=? WHERE status='running'", (now(),))
                db.execute("UPDATE samples SET status='failed' WHERE status='running'")
                cursor = db.execute("INSERT INTO sessions(started,settings) VALUES(?,?)", (now(), dumps({
                    "limit": limit, "batch_size": batch_size, "max_attempts": max_attempts, "git": code})))
                session = cursor.lastrowid
            backend = None
            started = perf_counter()
            position = 0
            while True:
                rows = db.execute("SELECT position,id,payload,status,attempts,image_sha256 FROM samples "
                                  "WHERE position>? AND position<=? ORDER BY position LIMIT ?",
                                  (position, limit, batch_size)).fetchall()
                if not rows:
                    break
                for position, sample_id, payload, status, tries, old_hash in rows:
                    row = json.loads(payload)
                    image = Path(row["image_path"])
                    if not image.is_absolute():
                        image = samples.parent / image
                    try:
                        image_hash = sha256(image) if image.is_file() else None
                    except OSError:
                        image_hash = None  # The request below records a per-sample I/O failure.
                    if status == "completed":
                        if image_hash != old_hash:
                            raise ValueError(f"cached image changed: {sample_id}; create a new run")
                        continue
                    while tries < max_attempts:
                        tries += 1
                        capture = None
                        with db:
                            db.execute("UPDATE samples SET status='running',attempts=?,image_sha256=? WHERE id=?",
                                       (tries, image_hash, sample_id))
                            attempt = db.execute("INSERT INTO attempts(sample_id,started,status,image_sha256) VALUES(?,?,'running',?)",
                                                 (sample_id, now(), image_hash)).lastrowid
                        try:
                            request = VLMRequest(image, effective_prompt, sample_id, config.max_image_size_mb)
                            if backend is None:
                                backend = backend_factory() if backend_factory else (
                                    StructuredMock() if config.backend == "mock" else create_vlm(config.backend, config=config))
                                if config.backend == "qwen3_vl":
                                    import torch
                                    torch.manual_seed(0)
                                    torch.cuda.manual_seed_all(0)
                                    runtime = {"torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                                               "gpu_name": torch.cuda.get_device_name(config.device),
                                               "total_memory_bytes": torch.cuda.get_device_properties(config.device).total_memory,
                                               "seed": 0, "determinism": "greedy; cross-device bitwise equality not guaranteed"}
                                    with (output / f"runtime-{session:04d}.json").open("x", encoding="utf-8") as stream:
                                        stream.write(dumps(runtime) + "\n")
                            capture = Capture(backend, config.backend)
                            # Record intent BEFORE entering the backend so hard crashes are auditable.
                            with db:
                                db.execute("UPDATE attempts SET backend_called=1 WHERE id=?", (attempt,))
                            result = processor(request, capture, index, k=top_k)
                            if result["trace"]["image_sha256"] != image_hash:
                                raise ValueError("image changed during inference")
                            result["input"] = row
                            result["content_source"] = "mock_fixture" if config.backend == "mock" else "unverified_model_generation"
                        except Exception as exc:
                            with db:
                                response = capture.response if capture else None
                                db.execute("UPDATE attempts SET status='failed',finished=?,error_type=?,error=?,raw_response=?,real_response=? WHERE id=?",
                                           (now(), type(exc).__name__, str(exc), dumps(response.to_dict()) if response else None,
                                            int(config.backend == "qwen3_vl" and response is not None
                                                and response.backend == "qwen3_vl" and not response.is_mock
                                                and response.metadata.get("inference_performed") is True), attempt))
                                db.execute("UPDATE samples SET status='failed' WHERE id=?", (sample_id,))
                        else:
                            with db:
                                db.execute("UPDATE attempts SET status='completed',finished=?,raw_response=?,real_response=? WHERE id=?",
                                           (now(), dumps(capture.response.to_dict()), int(not capture.response.is_mock), attempt))
                                db.execute("UPDATE samples SET status='completed',result=? WHERE id=?", (dumps(result), sample_id))
                            break
                if progress:
                    print(dumps({"session": session, "processed_prefix": position, "target": limit}), file=sys.stderr, flush=True)
            report = summary(db, limit, len(index.document_ids), config.backend, top_k)
            report.update({"task": task, "query_mode": query_mode})
            if task == "image_to_caption_retrieval_proxy":
                report["qrels_status"] = "same-image caption association proxy; not independent relevance or factual truth"
            report.update({"session": session, "elapsed_seconds": perf_counter() - started,
                           "updated_utc": now(), "requested_limit": limit})
            # Numbered snapshots are immutable. SQLite holds all earlier sessions and errors.
            with (output / f"summary-{session:04d}.json").open("x", encoding="utf-8") as stream:
                stream.write(dumps(report) + "\n")
            with (output / f"results-{session:04d}.jsonl").open("x", encoding="utf-8") as stream:
                for result, in db.execute("SELECT result FROM samples WHERE position<=? AND status='completed' ORDER BY position", (limit,)):
                    stream.write(result + "\n")
            with db:
                db.execute("UPDATE sessions SET finished=?,summary=? WHERE id=?", (now(), dumps(report), session))
            return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("samples", "corpus", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--qrels", type=Path, help="Independent retrieval relevance judgments, never pair labels")
    parser.add_argument("--mode", choices=("mock", "qwen3_vl"), required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--query-mode", choices=("structured", "baseline"), default="structured")
    parser.add_argument("--task", choices=("evidence_retrieval", "image_to_caption_retrieval_proxy"), default="evidence_retrieval")
    args = parser.parse_args(argv)
    try:
        config = load_vlm_config(args.config)
        if config.backend != args.mode:
            raise ValueError("--mode must match config backend; no implicit fallback")
        report = run_batch(samples=args.samples, corpus=args.corpus, output=args.output, config=config,
                           limit=args.limit, batch_size=args.batch_size, max_attempts=args.max_attempts,
                           resume=args.resume, top_k=args.top_k, k1=args.k1, b=args.b, qrels=args.qrels, progress=True,
                           query_mode=args.query_mode, task=args.task)
        print(dumps(report))
        return 0 if report["status"] == "completed" else 1
    except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
        print(dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

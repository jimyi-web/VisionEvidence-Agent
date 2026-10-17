"""CPU-only structured fixture runner; never loads a real model."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

from visionevidence import __version__
from visionevidence.observation import PROMPT, observe_and_retrieve, parse_observation
from visionevidence.retrieval import BM25Index, load_corpus
from visionevidence.vlm.schemas import VLMRequest, VLMResponse


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n")


class StructuredFixtureVLM:
    """Replay a caller-supplied synthetic observation without reading pixels."""

    def __init__(self, text: str):
        observation = parse_observation(text)
        if not any("[MOCK]" in item for item in observation.uncertainties):
            raise ValueError("synthetic input requires [MOCK] in uncertainties")
        self.text = text

    def generate(self, request: VLMRequest) -> VLMResponse:
        return VLMResponse(request.request_id, self.text, "structured-fixture-v1", "mock", True, 0,
                           {"inference_performed": False, "pixels_analyzed": False,
                            "purpose": "synthetic_interface_test_only", "model_revision": None})


def code_provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "visionevidence").rglob("*.py"))
    result = {"source_sha256": {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(root), *args], check=True,
                                  capture_output=True, text=True, timeout=10).stdout.strip()
        result.update(commit=git("rev-parse", "HEAD"), git_status=git("status", "--porcelain"))
        result["git_dirty"] = bool(result["git_status"])
    except (OSError, subprocess.SubprocessError) as exc:
        result.update(commit=None, git_dirty=None, unavailable_reason=str(exc))
    return result


def run_observation(args, options_loader) -> dict:
    """Reserve a new directory and preserve both successful and failed attempts."""
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1, "experiment": "T04-3", "status": "started",
        "request_id": args.request_id, "started_utc": datetime.now(timezone.utc).isoformat(),
        "is_mock": True, "inference_performed": False, "purpose": "synthetic_interface_test_only",
        "claim": None, "claim_note": "No claim accepted or verified in this observation-only workflow",
        "dataset": {"name": args.dataset_name, "version": args.dataset_version, "split": args.split},
        "sample_count": 1, "completed_samples": 0, "inputs": {},
        "random_seed": None, "randomness": "none; deterministic fixture replay and BM25",
        "model": {"identifier": "structured-fixture-v1", "revision": None, "backend": "mock"},
        "peak_memory_bytes": None, "peak_memory_note": "not measured; GPU not used",
        "metrics": None,
    }
    try:
        manifest["code"] = code_provenance()
        manifest["environment"] = {
            "package_version": __version__, "python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "device": "cpu", "dependencies": sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()}),
        }
        # Snapshot the exact bytes consumed, so later source-file edits cannot invalidate provenance.
        for name, path in (("observation", args.observation), ("corpus", args.corpus),
                           ("image", args.observation_image), ("config", args.config)):
            if path is None:
                continue
            data = path.read_bytes()
            snapshot = args.output / (name + path.suffix)
            with snapshot.open("xb") as stream:
                stream.write(data)
            manifest["inputs"][name] = {"path": str(path.resolve()), "snapshot": snapshot.name,
                                         "sha256": hashlib.sha256(data).hexdigest()}
        def saved(name):
            return args.output / manifest["inputs"][name]["snapshot"]
        options = options_loader(saved("config") if args.config else None)
        manifest["config"] = {"bm25": options, "top_k": args.top_k, "prompt": PROMPT}
        model = StructuredFixtureVLM(saved("observation").read_text(encoding="utf-8"))
        request = VLMRequest(saved("image"), PROMPT, request_id=args.request_id)
        index_started = perf_counter()
        index = BM25Index(load_corpus(saved("corpus")), **options)
        manifest["index_build_ms"] = (perf_counter() - index_started) * 1000
        manifest["corpus_document_count"] = len(index.document_ids)
        result = observe_and_retrieve(request, model, index, k=args.top_k)
        write_json(args.output / "result.json", result)
        manifest.update(status="completed", completed_samples=1,
                        result_sha256=hashlib.sha256((args.output / "result.json").read_bytes()).hexdigest())
        return result
    except Exception as exc:
        manifest.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["total_ms"] = (perf_counter() - started) * 1000
        write_json(args.output / "manifest.json", manifest)

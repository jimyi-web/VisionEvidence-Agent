"""Input validation and CPU retrieval CLI. No model inference or verification."""

import argparse
import json
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import sys
from time import perf_counter
import tomllib

from visionevidence import __version__
from visionevidence.evaluation.retrieval import evaluate_retrieval, load_qrels
from visionevidence.retrieval import BM25Index, evidence_to_dict, load_corpus
from visionevidence.retrieval.bm25 import TOKENIZER
from visionevidence.schemas import ClaimInput


def _bm25_options(path: Path | None) -> dict[str, float]:
    if path is None:
        return {"k1": 1.5, "b": 0.75}
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if config.keys() != {"bm25"} or not isinstance(config["bm25"], dict):
        raise ValueError("config must contain only a [bm25] table")
    if config["bm25"].keys() != {"k1", "b"}:
        raise ValueError("[bm25] config must contain exactly k1 and b")
    return config["bm25"]


def _fingerprint(path: Path) -> dict[str, str]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest}


def _run_retrieval(args: argparse.Namespace) -> None:
    started = perf_counter()
    parameters = _bm25_options(args.config)
    documents = load_corpus(args.corpus)
    index_started = perf_counter()
    index = BM25Index(documents, **parameters)
    index_seconds = perf_counter() - index_started
    config = {
        "method": "bm25", "idf": "log1p((N-df+0.5)/(df+0.5))",
        "k1": index.k1, "b": index.b, "top_k": args.top_k,
        "tokenizer": TOKENIZER, "query_term_frequency": "binary",
        "tie_break": "evidence_id ascending", "matching_only": True,
    }
    if args.command == "retrieve":
        results = index.search(args.query, args.top_k)
        if args.json:
            print(json.dumps({"query": args.query, "config": config,
                              "results": [evidence_to_dict(item) for item in results]}, allow_nan=False))
        elif not results:
            print("No matching evidence.")
        else:
            for item in results:
                # ASCII escapes keep arbitrary UTF-8 corpus text printable on Windows consoles.
                print(json.dumps(evidence_to_dict(item), allow_nan=False))
        return

    queries = load_qrels(args.qrels)
    report = evaluate_retrieval(index, queries, args.top_k)
    config["empty_qrels_policy"] = "exclude and count; error if none evaluable"
    report.update({
        "schema_version": 1, "config": config,
        "dataset": {"name": args.dataset_name, "version": args.dataset_version, "split": args.split},
        "inputs": {"corpus": _fingerprint(args.corpus), "qrels": _fingerprint(args.qrels),
                   "config": _fingerprint(args.config) if args.config else None},
        "environment": {"package_version": __version__, "python": sys.version,
                        "python_executable": sys.executable, "platform": platform.platform(),
                        "machine": platform.machine(), "processor": platform.processor(),
                        "device": "cpu", "runtime_dependencies": []},
        "random_seed": None, "randomness": "none; deterministic lexical retrieval",
        "model": None, "peak_memory_bytes": None, "peak_memory_note": "not measured",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "timing": {"index_build_seconds": index_seconds, "total_seconds": perf_counter() - started},
    })
    serialized = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False)
    if args.output:
        # Exclusive create: never overwrite previous experiments, including source inputs.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
    print(serialized)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="visionevidence",
        description="Validate input or retrieve text evidence; model inference is not implemented.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--text", help="Non-empty text claim to validate")
    parser.add_argument("--image", help="Optional path to an existing local image file")
    commands = parser.add_subparsers(dest="command")
    for name, description in (("retrieve", "Retrieve BM25 evidence"), ("evaluate", "Evaluate retrieval against explicit qrels")):
        command = commands.add_parser(name, help=description)
        command.add_argument("--corpus", type=Path, required=True, help="UTF-8 evidence JSONL")
        command.add_argument("--top-k", type=int, default=3, help="Positive cutoff (default: 3)")
        command.add_argument("--config", type=Path, help="TOML with [bm25] k1 and b; default: 1.5, 0.75")
        if name == "retrieve":
            command.add_argument("--query", required=True)
            command.add_argument("--json", action="store_true", help="Emit one machine-readable JSON object")
        else:
            command.add_argument("--qrels", type=Path, required=True)
            command.add_argument("--output", type=Path, help="New UTF-8 JSON file; existing files are never overwritten")
            command.add_argument("--dataset-name", default="unspecified")
            command.add_argument("--dataset-version", default="unspecified")
            command.add_argument("--split", default="unspecified")
    args = parser.parse_args(argv)
    try:
        if args.command:
            if args.text is not None or args.image is not None:
                raise ValueError("--text/--image cannot be combined with retrieval subcommands")
            _run_retrieval(args)
            return 0
        if args.text is None:
            parser.error("--text is required unless using retrieve or evaluate")
        claim = ClaimInput(text=args.text, image_path=args.image)
    except (TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))

    print(json.dumps({
        "status": "input_validated",
        "inference_performed": False,
        "message": "Input validation only; model inference is not implemented.",
        "input": {
            "text": claim.text,
            "image_path": str(claim.image_path) if claim.image_path is not None else None,
        },
        "metadata": {"package_version": __version__},
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

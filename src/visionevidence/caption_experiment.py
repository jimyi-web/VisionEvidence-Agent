"""Paired description/structured image-to-caption retrieval proxy on one split.

This is caption association evaluation, not evidence-grounded fact verification.
"""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

from visionevidence.batch import StructuredMock, dumps, now, run_batch, run_lock, sha256
from visionevidence.datasets.flickr8k import TASK
from visionevidence.evaluation.retrieval import metrics_at_k
from visionevidence.vlm.factory import VLMConfig, create_vlm, load_vlm_config


def score_branch(path: Path, limit: int, cutoffs: tuple[int, ...]) -> dict:
    """All selected queries form the denominator, including failed generations."""
    totals = {str(k): {"hit_rate": 0.0, "caption_recall": 0.0, "mrr": 0.0} for k in cutoffs}
    per_query = []
    with closing(sqlite3.connect(path / "state.sqlite3")) as db:
        for sample_id, status, result, labels in db.execute(
            "SELECT s.id,s.status,s.result,q.relevant_ids FROM samples s LEFT JOIN qrels q ON s.id=q.sample_id "
            "WHERE s.position<=? ORDER BY s.position", (limit,)):
            if labels is None or len(json.loads(labels)) != 5:
                raise ValueError("proxy requires exactly five original same-image caption targets per query")
            relevant = frozenset(json.loads(labels))
            ranked = [doc["document_id"] for doc in json.loads(result)["evidence"]] if status == "completed" else []
            metrics = {}
            for k in cutoffs:
                recall, rr = metrics_at_k(ranked, relevant, k)
                metrics[str(k)] = {"hit_rate": float(rr > 0), "caption_recall": recall, "mrr": rr}
                for name, value in metrics[str(k)].items():
                    totals[str(k)][name] += value
            per_query.append({"sample_id": sample_id, "status": status, "metrics": metrics})
    if not per_query:
        raise ValueError("empty proxy evaluation")
    for metrics in totals.values():
        for name in metrics:
            metrics[name] /= len(per_query)
    return {"denominator": len(per_query), "metrics": totals, "per_query": per_query,
            "recall_naming": "hit_rate is conventional image-to-text R@K (any of five captions); caption_recall is fraction of all five targets"}


def run_experiment(*, dataset: Path, output: Path, config: VLMConfig | None = None,
                   limit: int = 20, batch_size: int = 4, max_attempts: int = 1,
                   resume: bool = False, cutoffs: tuple[int, ...] = (1, 5, 10),
                   backend_factory=None, progress: bool = False) -> dict:
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if not cutoffs or any(type(k) is not int or k <= 0 for k in cutoffs) or len(cutoffs) != len(set(cutoffs)):
        raise ValueError("cutoffs must be distinct positive integers")
    config = config or VLMConfig()
    adapter_manifest = dataset.parent / "manifest.json"
    manifest = json.loads(adapter_manifest.read_text(encoding="utf-8"))
    if manifest.get("task") != TASK or dataset.name not in manifest["splits"]:
        raise ValueError("dataset must be one original split of a Flickr8k proxy adapter bundle")
    split = manifest["splits"][dataset.name]
    for name, digest in split["outputs"].items():
        if sha256(dataset / f"{name}.jsonl") != digest:
            raise ValueError(f"adapter output fingerprint changed: {name}")
    if type(limit) is not int or not 0 < limit <= split["images"]:
        raise ValueError("limit must fit the selected original split; 8000 requires separate train/dev/test runs")
    identity = {"task": TASK, "adapter_manifest_sha256": sha256(adapter_manifest),
                "split": dataset.name, "cutoffs": list(cutoffs)}
    if resume:
        if json.loads((output / "experiment.json").read_text(encoding="utf-8")) != identity:
            raise ValueError("proxy protocol identity changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "experiment.json").write_text(dumps(identity) + "\n", encoding="utf-8")
    model = None
    def shared_backend():
        nonlocal model
        if model is None:
            model = backend_factory() if backend_factory else (
                StructuredMock() if config.backend == "mock" else create_vlm(config.backend, config=config))
        return model
    with run_lock(output / ".experiment.lock"):
        branches, scores = {}, {}
        for mode in ("baseline", "structured"):
            branch = output / mode
            branches[mode] = run_batch(samples=dataset / "samples.jsonl", corpus=dataset / "corpus.jsonl",
                qrels=dataset / "qrels.jsonl", output=branch, config=config, limit=limit,
                batch_size=batch_size, max_attempts=max_attempts, top_k=max(cutoffs),
                task=TASK, query_mode=mode, backend_factory=shared_backend,
                resume=resume and (branch / "manifest.json").exists(), progress=progress)
            scores[mode] = score_branch(branch, limit, cutoffs)
        # Explicitly verify comparison controls rather than trust caller settings.
        identities = [json.loads((output / mode / "manifest.json").read_text())["identity"] for mode in branches]
        for field in ("samples_sha256", "corpus_sha256", "qrels_sha256", "retrieval", "vlm_config", "model_file_sha256", "source_sha256", "seed"):
            if identities[0][field] != identities[1][field]:
                raise ValueError(f"comparison control mismatch: {field}")
        real_images = set()
        for mode in branches:
            with closing(sqlite3.connect(output / mode / "state.sqlite3")) as db:
                real_images.update(row[0] for row in db.execute(
                    "SELECT DISTINCT a.image_sha256 FROM attempts a JOIN samples s ON s.id=a.sample_id "
                    "WHERE a.real_response=1 AND s.position<=?", (limit,)) if row[0])
        result = {"task": TASK, "verified": False, "is_mock": config.backend == "mock", "backend": config.backend,
                  "status": "completed" if all(b["status"] == "completed" for b in branches.values()) else "incomplete",
                  "created_utc": now(), "original_split": dataset.name,
                  "scale": {"candidate_caption_documents": split["candidate_caption_documents"],
                            "evaluation_images": limit, "retrieval_query_slots": 2 * limit,
                            "real_vlm_image_calls": sum(b["scale"]["real_vlm_image_calls"] for b in branches.values()),
                            "unique_real_vlm_images": len(real_images)},
                  "branches": branches, "scores": scores,
                  "limitations": ["same-image caption association proxy, not external fact verification",
                                  "other-image captions can also be semantically relevant (unjudged negatives)",
                                  "five captions are correlated; captions are not five independent images",
                                  "model pretraining contamination cannot be ruled out"],
                  "failure_policy": "all selected queries in both branches; failed or pending queries score zero"}
        number = 1
        while (output / f"comparison-{number:04d}.json").exists():
            number += 1
        with (output / f"comparison-{number:04d}.json").open("x", encoding="utf-8") as stream:
            stream.write(dumps(result) + "\n")
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Adapter output train, dev or test directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("mock", "qwen3_vl"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_vlm_config(args.config)
        if config.backend != args.mode:
            raise ValueError("mode/config mismatch; no automatic mock fallback")
        result = run_experiment(dataset=args.dataset, output=args.output, config=config, limit=args.limit,
                               batch_size=args.batch_size, max_attempts=args.max_attempts, resume=args.resume, progress=True)
        # Full per-query metrics are in the immutable comparison JSON, not the console.
        print(dumps({key: result[key] for key in ("task", "status", "is_mock", "scale")}))
        return 0 if result["status"] == "completed" else 1
    except Exception as exc:
        print(dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

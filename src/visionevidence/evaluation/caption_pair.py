"""CPU-only paired reporting. Invoke only after the independent run is authorized."""

import argparse
from contextlib import closing
from datetime import datetime
import json
from pathlib import Path
import random
import sqlite3
import statistics

from visionevidence.batch import dumps, sha256
from visionevidence.caption_experiment import score_branch
from visionevidence.observation import parse_observation
from visionevidence.observation_policy import ObservationPolicy


def paired_hit_interval(baseline, structured, *, repetitions=20000, seed=0):
    """Percentile bootstrap over image pairs; fixed seed, no successful-only filtering."""
    if (not baseline or len(baseline) != len(structured)
            or any(x not in (0, 1) for x in [*baseline, *structured])
            or type(repetitions) is not int or repetitions < 100):
        raise ValueError("require equal nonempty binary pairs and at least 100 replicates")
    differences = [s - b for b, s in zip(baseline, structured)]
    rng = random.Random(seed)
    n = len(differences)
    draws = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(repetitions))
    def quantile(p):
        position = (repetitions - 1) * p
        lower = int(position)
        upper = min(lower + 1, repetitions - 1)
        return draws[lower] + (draws[upper] - draws[lower]) * (position - lower)
    return {"difference": statistics.mean(differences), "lower": quantile(.025),
            "upper": quantile(.975), "confidence": .95, "repetitions": repetitions, "seed": seed,
            "method": "paired image bootstrap percentile, linear empirical quantiles",
            "limitation": "conditional on this fixed cohort; small or constant differences can yield a degenerate interval; no correction for model pretraining exposure"}


def distribution(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": min(values), "max": max(values)} if values else {"n": 0}


def report(run: Path, limit: int = 100):
    branches, paired = {}, []
    controls = []
    image_hashes = []
    for name in ("baseline", "structured"):
        path = run / name
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        identity = manifest["identity"]
        controls.append(identity)
        scores = score_branch(path, limit, (1, 5, 10))
        if scores["denominator"] != limit:
            raise ValueError("missing selected queries")
        with closing(sqlite3.connect((path / "state.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            samples = list(db.execute("SELECT * FROM samples WHERE position<=? ORDER BY position", (limit,)))
            attempts = list(db.execute("SELECT a.* FROM attempts a JOIN samples s ON s.id=a.sample_id WHERE s.position<=? ORDER BY a.id", (limit,)))
        if any(s["status"] not in ("completed", "failed") for s in samples):
            raise ValueError("finish or resolve interrupted sessions before final reporting")
        if any(s["attempts"] != 1 for s in samples):
            raise ValueError("frozen protocol allows exactly one attempt per selected query")
        image_hashes.append([s["image_sha256"] for s in samples])
        raws = [json.loads(a["raw_response"]) for a in attempts if a["raw_response"]]
        if any(r["is_mock"] or r["backend"] != "qwen3_vl" for r in raws):
            raise ValueError("real run required")
        schema = contract = 0
        policy = identity.get("observation_policy") or {}
        for raw in raws:
            try:
                observation = parse_observation(raw["description"])
                schema += 1
                ObservationPolicy(policy.get("format_constraints", False),
                                  policy.get("retain_visual_details", False)).validate(observation)
                contract += 1
            except ValueError:
                pass
        results = [json.loads(s["result"]) for s in samples if s["result"]]
        branches[name] = {"scores": scores, "success_rate": sum(s["status"] == "completed" for s in samples) / limit,
            "failure_rate": sum(s["status"] == "failed" for s in samples) / limit,
            "schema_success_rate": schema / limit if name == "structured" else None,
            "format_contract_success_rate": contract / limit if name == "structured" else None,
            "schema_scope": "all selected images, evaluated independently of token-limit acceptance; not applicable to prose baseline",
            "response_latency_ms_including_failed_responses": distribution([r["latency_ms"] for r in raws]),
            "attempt_wall_ms_all": distribution([
                (datetime.fromisoformat(a["finished"]) -
                 datetime.fromisoformat(a["started"])).total_seconds() * 1000
                for a in attempts if a["finished"]]),
            "retrieval_ms_successful_only": distribution([r["trace"]["retrieval_ms"] for r in results]),
            "model_load_ms_cold_calls": [r["metadata"]["model_load_ms"] for r in raws if r["metadata"]["model_loaded_this_call"]],
            "failure_records": [{k: a[k] for k in ("sample_id", "status", "error_type", "error")} for a in attempts if a["status"] != "completed"]}
    for field in ("samples_sha256", "corpus_sha256", "qrels_sha256", "retrieval", "vlm_config", "model_file_sha256", "source_sha256", "seed"):
        if controls[0][field] != controls[1][field]:
            raise ValueError(f"paired control mismatch: {field}")
    if image_hashes[0] != image_hashes[1]:
        raise ValueError("paired image bytes mismatch")
    base = branches["baseline"]["scores"]["per_query"]
    structured = branches["structured"]["scores"]["per_query"]
    for b, s in zip(base, structured):
        if b["sample_id"] != s["sample_id"]:
            raise ValueError("paired IDs mismatch")
        paired.append({"sample_id": b["sample_id"], "baseline_status": b["status"], "structured_status": s["status"],
            "delta": {k: {metric: s["metrics"][k][metric] - b["metrics"][k][metric]
                           for metric in ("hit_rate", "mrr", "caption_recall")} for k in ("1", "5", "10")}})
    return {"task": "image_to_caption_retrieval_proxy", "is_fact_verification": False,
            "failure_policy": "all selected queries; failures score zero", "branches": branches, "paired": paired,
            "hit10_difference_ci95": paired_hit_interval([r["metrics"]["10"]["hit_rate"] for r in base],
                                                         [r["metrics"]["10"]["hit_rate"] for r in structured]),
            "manifest_sha256": {name: sha256(run / name / "manifest.json") for name in branches}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)
    value = report(args.run, args.limit)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(dumps(value) + "\n")


if __name__ == "__main__":
    main()

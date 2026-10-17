"""Validate and summarize the six-run T03-B-5 artifact bundle, without GPU imports."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(directory: Path, expected_commit: str):
    raw = {name: (directory / name).read_bytes()
           for name in ("manifest.json", "runs.jsonl", "result.json")}
    manifest = strict_json(raw["manifest.json"].decode("utf-8"))
    result = strict_json(raw["result.json"].decode("utf-8"))
    rows = [strict_json(line) for line in raw["runs.jsonl"].decode("utf-8").splitlines()]
    require(manifest["is_mock"] is False and result["is_mock"] is False, "mock artifacts")
    require(result["status"] == "completed" and result["completed_runs"] == len(rows) == 6,
            "incomplete experiment")
    require(manifest["code_commit"] == expected_commit, "unexpected commit")
    require(manifest["git_dirty"] is False, "dirty experiment code")
    for key, value in manifest.items():
        if key != "status":
            require(result.get(key) == value, f"manifest/result mismatch: {key}")
    require(manifest["schedule"] == [256, 512, 256, 512, 256, 512], "unexpected schedule")
    uuid = manifest["hardware"][0]["uuid"]
    require(len(manifest["hardware"]) == 1, "multiple GPUs require explicit device mapping")
    metrics = []
    for i, row in enumerate(rows):
        response, limit = row["response"], manifest["schedule"][i]
        meta = response["metadata"]
        require(row["run_index"] == i and row["max_new_tokens"] == limit, "run order mismatch")
        require(response["is_mock"] is False and meta["inference_performed"] is True,
                f"run {i}: not real inference")
        require(response["backend"] == manifest["backend"] and
                response["model_name"] == manifest["config"]["model_name"], "backend mismatch")
        for key in ("device", "dtype", "model_path", "do_sample", "prompt_version"):
            require(meta[key] == manifest["config"][key], f"run {i}: {key} mismatch")
        for key, package in (("torch_version", "torch"), ("transformers_version", "transformers")):
            require(f"{package}=={meta[key]}" in manifest["dependencies"], "version mismatch")
        require(meta["model_loaded_this_call"] is (i == 0), "model reload mismatch")
        require(row["phase"] == ("cold" if i == 0 else "reuse"), "phase mismatch")
        require(meta["max_new_tokens"] == limit and
                row["generated_tokens"] == meta["generated_tokens"], "token mismatch")
        require(row["token_limit_reached"] is (row["generated_tokens"] >= limit), "limit mismatch")
        require(row["latency_ms"] == response["latency_ms"] and
                row["generation_ms"] == meta["generation_ms"] and
                row["torch_peak_allocated_mib"] == meta["peak_gpu_memory_mb"], "metric mismatch")
        require(not row["gpu_sampling_errors"] and row["gpu_samples"], "sampling failure")
        samples = [g for sample in row["gpu_samples"] for g in sample["gpus"] if g["uuid"] == uuid]
        require(len(samples) == len(row["gpu_samples"]), "GPU UUID mismatch")
        metrics.append({**{k: row[k] for k in ("run_index", "phase", "max_new_tokens", "wall_ms",
                        "latency_ms", "generation_ms", "generated_tokens", "torch_peak_allocated_mib")},
                        "board_sample_max_mib": max(g["memory_used_mib"] for g in samples),
                        "sample_count": len(samples),
                        "generation_tokens_per_second": row["generated_tokens"] * 1000 / row["generation_ms"],
                        "output_tail": response["description"][-100:]})
    return {"validated": True, "code_commit": manifest["code_commit"],
            "artifact_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()},
            "model_load_ms": rows[0]["response"]["metadata"]["model_load_ms"],
            "metrics": metrics,
            "reuse_wall_mean_ms": {str(t): mean(r["wall_ms"] for r in rows[1:] if r["max_new_tokens"] == t)
                                   for t in (256, 512)},
            "later_512_wall_mean_ms": mean(rows[i]["wall_ms"] for i in (3, 5)),
            "same_text_within_token_group": all(rows[i]["response"]["description"] ==
                                                rows[i % 2]["response"]["description"] for i in range(6))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory, args.expected_commit), indent=2, ensure_ascii=True))

"""T03-B-5 single-process experiment; no downloads or dependency installation."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import random
import subprocess
from threading import Event, Thread
from time import perf_counter

from .factory import create_vlm, load_vlm_config
from .prompts import get_prompt
from .schemas import VLMRequest


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_manifest(directory: Path) -> list[dict]:
    # Hash actual bytes, including symlink targets in local HF snapshots.
    files = sorted(p for p in directory.rglob("*") if p.is_file() and ".cache" not in p.parts)
    if not files:
        raise ValueError("local model directory contains no files")
    return [{"path": p.relative_to(directory).as_posix(), "bytes": p.stat().st_size,
             "sha256": sha256(p)} for p in files]


def gpu_snapshot() -> list[dict]:
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=uuid,name,driver_version,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True, timeout=5)
    rows = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        uuid, name, driver, utilization, used, total = row
        rows.append({"uuid": uuid, "name": name, "driver_version": driver,
                     "utilization_percent": float(utilization), "memory_used_mib": float(used),
                     "memory_total_mib": float(total)})
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU samples")
    return rows


class GPUSampler:
    """Board-wide samples of all physical GPUs; UUIDs avoid CUDA index ambiguity."""

    def __init__(self, interval: float):
        self.interval = interval
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self.stop = Event()
        self.thread = Thread(target=self._run, daemon=True)

    def _sample(self):
        try:
            self.samples.append({"elapsed_ms": (perf_counter() - self.started) * 1000,
                                 "gpus": gpu_snapshot()})
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            self.errors.append(str(exc))

    def _run(self):
        while not self.stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self.started = perf_counter()
        self._sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def write_json(path: Path, data: dict):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run_experiment(config_path: Path, image: Path, output: Path, *,
                   tokens=(256, 512), repeats=3, seed=0, interval=0.2,
                   prompt: str | None = None, model_revision="unspecified") -> dict:
    if repeats < 3 or len(tokens) != 2 or set(tokens) != {256, 512}:
        raise ValueError("requires at least 3 repeats and token limits 256 and 512")
    if not 0.05 <= interval <= 10:
        raise ValueError("sample interval must be between 0.05 and 10 seconds")
    cfg = load_vlm_config(config_path)
    if cfg.do_sample:
        raise ValueError("T03-B-5 requires do_sample=false for greedy comparison")
    request = VLMRequest(image, prompt or get_prompt(cfg.prompt_version),
                         max_image_size_mb=cfg.max_image_size_mb)
    root = Path(__file__).resolve().parents[3]
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    commit, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "experiment": "T03-B-5",
                "started_utc": datetime.now(timezone.utc).isoformat(), "status": "preparing",
                "is_mock": cfg.backend == "mock", "backend": cfg.backend,
                "purpose": "interface_test_only" if cfg.backend == "mock" else "single_image_gpu_smoke",
                "code_commit": commit, "git_dirty": bool(status), "git_status": status,
                "python": platform.python_version(), "platform": platform.platform(),
                "dependencies": sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()}),
                "pid": os.getpid(), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                "config": {**asdict(cfg), "model_path": str(cfg.model_path) if cfg.model_path else None},
                "config_sha256": sha256(config_path), "image_path": str(image.resolve()),
                "image_sha256": sha256(image), "prompt": request.prompt,
                "model_revision": model_revision, "seed": seed, "tokens": list(tokens),
                "repeats_per_limit": repeats, "sample_interval_seconds": interval,
                "schedule": [t for _ in range(repeats) for t in tokens],
                "dataset": "user_supplied_single_image", "split": "smoke_only", "sample_count": 1,
                "gpu_scope": "all physical boards, not process-specific; includes loading on first call"}
    count = 0
    try:
        manifest["model_files"] = (model_manifest(Path(cfg.model_path))
                                   if cfg.backend == "qwen3_vl" and cfg.model_path else [])
        if cfg.backend == "qwen3_vl":
            manifest["hardware"] = gpu_snapshot()  # Fail before expensive loading if telemetry is unavailable.
            import torch
            torch.manual_seed(seed)
            manifest["cuda_runtime"] = torch.version.cuda
        random.seed(seed)
        backend = create_vlm(cfg.backend, config=cfg)
        write_json(output / "manifest.json", manifest)
        with (output / "runs.jsonl").open("x", encoding="utf-8") as stream:
            for index, limit in enumerate(manifest["schedule"]):
                sampler = GPUSampler(interval) if cfg.backend == "qwen3_vl" else None
                def infer():
                    started = perf_counter()
                    response = (backend.generate(request, max_new_tokens=limit) if sampler
                                else backend.generate(request))
                    return response, (perf_counter() - started) * 1000
                if sampler:
                    with sampler:
                        response, wall_ms = infer()
                else:
                    response, wall_ms = infer()
                meta = response.metadata
                record = {"run_index": index, "max_new_tokens": limit,
                          "phase": "mock" if response.is_mock else ("cold" if index == 0 else "reuse"),
                          "wall_ms": wall_ms, "latency_ms": response.latency_ms,
                          "generation_ms": meta.get("generation_ms"),
                          "generated_tokens": meta.get("generated_tokens"),
                          "token_limit_reached": (meta["generated_tokens"] >= limit
                                                  if "generated_tokens" in meta else None),
                          "torch_peak_allocated_mib": meta.get("peak_gpu_memory_mb"),
                          "gpu_samples": sampler.samples if sampler else [],
                          "gpu_sampling_errors": sampler.errors if sampler else [],
                          "response": response.to_dict()}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                count += 1
                if sampler and (sampler.errors or not sampler.samples):
                    raise RuntimeError("GPU telemetry incomplete; see runs.jsonl")
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["completed_runs"] = count
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "result.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory; never overwritten")
    parser.add_argument("--tokens", type=int, nargs=2, default=[256, 512])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--model-revision", default="unspecified")
    args = parser.parse_args(argv)
    result = run_experiment(args.config, args.image, args.output, tokens=args.tokens,
                            repeats=args.repeats, seed=args.seed, interval=args.interval,
                            prompt=args.prompt, model_revision=args.model_revision)
    print(json.dumps({"status": result["status"], "is_mock": result["is_mock"],
                      "completed_runs": result["completed_runs"], "output": str(args.output)}))


if __name__ == "__main__":
    main()

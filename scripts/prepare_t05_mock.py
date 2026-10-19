"""Create synthetic scheduler inputs; no claim of independent visual samples."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--count", type=int, default=10000)
args = parser.parse_args()
if args.count <= 0:
    parser.error("count must be positive")
image = (Path(__file__).resolve().parents[1] / "examples/vlm_demo_fixture.png").resolve()
with args.output.open("x", encoding="utf-8") as stream:
    for i in range(args.count):
        stream.write(json.dumps({"sample_id": f"mock-{i:06d}", "claim": "Synthetic scheduler input",
                                 "image_path": str(image), "dataset": "t05-synthetic-scheduler",
                                 "dataset_version": "1", "split": "smoke_only", "source": "synthetic://t05",
                                 "metadata": {"synthetic": True, "independent_visual_sample": False}}) + "\n")

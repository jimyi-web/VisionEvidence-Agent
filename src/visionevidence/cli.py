"""Input validation CLI. No inference, retrieval, or verification is performed."""

import argparse
import json
from collections.abc import Sequence

from visionevidence import __version__
from visionevidence.schemas import ClaimInput


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="visionevidence",
        description="Validate a claim input only; model inference is not implemented.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--text", required=True, help="Non-empty text claim to validate")
    parser.add_argument("--image", help="Optional path to an existing local image file")
    args = parser.parse_args(argv)
    try:
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

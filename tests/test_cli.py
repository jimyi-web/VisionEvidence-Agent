import json
from pathlib import Path
import subprocess
import sys

import pytest

from visionevidence import __version__


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    # sys.executable preserves the project's interpreter in child processes.
    return subprocess.run(
        [sys.executable, "-m", "visionevidence.cli", *args],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def test_help() -> None:
    result = run_cli("--help")
    assert result.returncode == 0
    assert "--text" in result.stdout
    assert "--image" in result.stdout
    assert "inference is not implemented" in result.stdout


def test_version() -> None:
    result = run_cli("--version")
    assert result.returncode == 0
    assert result.stdout.strip() == f"visionevidence {__version__}"


def test_text_input_does_not_produce_prediction() -> None:
    result = run_cli("--text", "  这是一条测试声明。  ")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "input_validated"
    assert payload["inference_performed"] is False
    assert payload["input"] == {"text": "这是一条测试声明。", "image_path": None}
    assert "predicted_label" not in payload
    assert "evidence_ids" not in payload


def test_image_input(tmp_path: Path) -> None:
    image = tmp_path / "sample image.png"
    image.write_bytes(b"path-only fixture")
    result = run_cli("--text", "A test claim.", "--image", str(image))
    assert result.returncode == 0
    assert json.loads(result.stdout)["input"]["image_path"] == str(image)


@pytest.mark.parametrize("args", [[], ["--text", ""], ["--text", " \t"], ["--unknown"]])
def test_invalid_arguments(args: list[str]) -> None:
    result = run_cli(*args)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert result.stdout == ""


def test_missing_image(tmp_path: Path) -> None:
    result = run_cli("--text", "A test claim.", "--image", str(tmp_path / "missing.png"))
    assert result.returncode == 2
    assert "existing file" in result.stderr
    assert result.stdout == ""

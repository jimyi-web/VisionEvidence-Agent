import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = str(ROOT / "examples/vlm_demo_fixture.png")
CONFIG = str(ROOT / "configs/vlm_mock.toml")


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "visionevidence.cli", *args],
                          capture_output=True, text=True, timeout=15, check=False)


def test_vlm_cli_text() -> None:
    result = run_cli("vlm", "--image", IMAGE, "--backend", "mock")
    assert result.returncode == 0, result.stderr
    for marker in ("MOCK", "backend=mock", "is_mock=True", "No visual analysis", "not an observation"):
        assert marker in result.stdout


def test_vlm_cli_json_and_config() -> None:
    result = run_cli("vlm", "--image", IMAGE, "--config", CONFIG, "--request-id", "demo-1", "--json")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["request_id"] == "demo-1"
    assert data["is_mock"] is True and data["backend"] == "mock"
    assert data["model_name"] == "mock-vlm"
    assert "[MOCK]" in data["description"]
    assert data["latency_ms"] >= 0
    assert data["metadata"]["inference_performed"] is False
    assert data["metadata"]["verified"] is False
    assert data["metadata"]["prompt_version"] == "visual_observation_v1"
    assert "predicted_label" not in data


def test_vlm_custom_prompt() -> None:
    result = run_cli("vlm", "--image", IMAGE, "--prompt", "Observe visible objects.", "--json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["metadata"]["prompt_version"] == "custom"


@pytest.mark.parametrize("args, message", [
    (["vlm"], "required"),
    (["vlm", "--image", "missing-vlm-fixture.png"], "existing regular file"),
    (["vlm", "--image", IMAGE, "--prompt", " "], "prompt must not be empty"),
    (["vlm", "--image", IMAGE, "--backend", "qwen3_vl"], "model_path"),
    (["vlm", "--image", IMAGE, "--backend", "other"], "invalid choice"),
    (["vlm", "--image", IMAGE, "--config", "missing-vlm-config.toml"], "invalid VLM config"),
    (["--text", "claim", "vlm", "--image", IMAGE], "cannot be combined"),
    (["--image", IMAGE, "vlm", "--image", IMAGE], "cannot be combined"),
    (["vlm", "--image", "https://example.org/image.png"], "must be local"),
])
def test_vlm_cli_errors(args: list[str], message: str) -> None:
    result = run_cli(*args)
    assert result.returncode == 2
    assert message in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


def test_vlm_cli_bad_image(tmp_path: Path) -> None:
    path = tmp_path / "fake.png"
    path.write_text("not an image", encoding="utf-8")
    result = run_cli("vlm", "--image", str(path), "--json")
    assert result.returncode == 2
    assert "content format" in result.stderr
    assert result.stdout == ""


def test_vlm_cli_config_values_and_override(tmp_path: Path) -> None:
    config = tmp_path / "vlm.toml"
    config.write_text(Path(CONFIG).read_text().replace('backend = "mock"', 'backend = "qwen3_vl"')
                      .replace('model_name = "mock-vlm"', 'model_name = "custom-mock"'), encoding="utf-8")
    result = run_cli("vlm", "--image", IMAGE, "--config", str(config))
    assert result.returncode == 2 and "model_path" in result.stderr
    overridden = run_cli("vlm", "--image", IMAGE, "--config", str(config), "--backend", "mock", "--json")
    assert overridden.returncode == 0, overridden.stderr
    assert json.loads(overridden.stdout)["model_name"] == "custom-mock"
    config.write_text(Path(CONFIG).read_text().replace("max_image_size_mb = 10", "max_image_size_mb = 0.000001"),
                      encoding="utf-8")
    too_small = run_cli("vlm", "--image", IMAGE, "--config", str(config))
    assert too_small.returncode == 2 and "exceeds" in too_small.stderr
    config.write_text("[broken", encoding="utf-8")
    invalid = run_cli("vlm", "--image", IMAGE, "--config", str(config))
    assert invalid.returncode == 2 and "invalid VLM config" in invalid.stderr
    assert "Traceback" not in invalid.stderr


def test_vlm_help() -> None:
    result = run_cli("vlm", "--help")
    assert result.returncode == 0
    assert all(option in result.stdout for option in ("--image", "--backend", "--config", "--json"))

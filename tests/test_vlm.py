import base64
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import struct
import subprocess
import sys
import zlib

import pytest

from visionevidence.vlm import (
    BaseVLM, MockVLM, VLMConfig, VLMRequest,
    VLMResponse, create_vlm, load_vlm_config,
)
from visionevidence.vlm.prompts import PROMPT_VERSION, get_prompt

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "examples/vlm_demo_fixture.png"


def test_demo_fixture_is_complete_valid_one_pixel_png() -> None:
    data = IMAGE.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    offset = 8
    chunks = []
    pixels = b""
    while offset < len(data):
        length = int.from_bytes(data[offset:offset + 4], "big")
        kind = data[offset + 4:offset + 8]
        body = data[offset + 8:offset + 8 + length]
        crc = int.from_bytes(data[offset + 8 + length:offset + 12 + length], "big")
        assert zlib.crc32(kind + body) == crc
        chunks.append(kind)
        if kind == b"IHDR":
            assert struct.unpack(">IIBBBBB", body) == (1, 1, 8, 2, 0, 0, 0)
        elif kind == b"IDAT":
            pixels += body
        offset += length + 12
    assert chunks == [b"IHDR", b"IDAT", b"IEND"]
    assert offset == len(data)
    assert zlib.decompress(pixels) == b"\x00\xff\xff\xff"


@pytest.mark.parametrize("path", [IMAGE, str(IMAGE)])
def test_request(path: str | Path) -> None:
    request = VLMRequest(path, " Describe visible content. ", " sample-1 ")
    assert request.image_path == IMAGE
    assert request.prompt == "Describe visible content."
    assert request.request_id == "sample-1"
    assert request.max_image_size_mb == 10
    with pytest.raises(FrozenInstanceError):
        request.prompt = "changed"


@pytest.mark.parametrize("suffix", [".png", ".PNG"])
def test_png_extensions(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / ("image" + suffix)
    path.write_bytes(IMAGE.read_bytes())
    VLMRequest(path, "Describe.")


@pytest.mark.parametrize("suffix, content", [
    # JPEG marker fixture: tests header recognition, not pixel decoding.
    (".jpg", b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 16),
    (".JPEG", b"\xff\xd8\xff\xe1\x00\x10Exif\x00" + b"\x00" * 16),
    (".webp", base64.b64decode("UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoBAAEADsD+JaQAA3AAAAAA")),
])
def test_other_supported_format_headers(tmp_path: Path, suffix: str, content: bytes) -> None:
    path = tmp_path / ("image" + suffix)
    path.write_bytes(content)
    VLMRequest(path, "Describe.")


def test_missing_path_and_directory(tmp_path: Path) -> None:
    for path in (tmp_path / "missing.png", tmp_path):
        with pytest.raises(ValueError, match="existing regular file"):
            VLMRequest(path, "Describe.")


@pytest.mark.parametrize("value", ["https://example.org/image.png", "file:///image.png",
                                        "//server/image.png", "\\\\server\\image.png"])
def test_reject_remote_paths_before_access(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("remote input must not reach filesystem access")
    monkeypatch.setattr(Path, "stat", forbidden)
    with pytest.raises(ValueError, match="local"):
        VLMRequest(value, "Describe.")


@pytest.mark.parametrize("value", [None, 123, b"image.png"])
def test_invalid_path_types(value) -> None:
    with pytest.raises(TypeError, match="image_path"):
        VLMRequest(value, "Describe.")


def test_invalid_extension(tmp_path: Path) -> None:
    path = tmp_path / "image.gif"
    path.write_bytes(IMAGE.read_bytes())
    with pytest.raises(ValueError, match="extension"):
        VLMRequest(path, "Describe.")


@pytest.mark.parametrize("suffix, content", [
    (".jpg", IMAGE.read_bytes()), (".png", b"not an image"), (".png", b""),
    (".png", b"\x89PNG\r\n\x1a\n"), (".webp", b"RIFF" + b"\x00" * 29),
])
def test_format_mismatch_and_truncated_headers(tmp_path: Path, suffix: str, content: bytes) -> None:
    path = tmp_path / ("image" + suffix)
    path.write_bytes(content)
    with pytest.raises(ValueError, match="content format"):
        VLMRequest(path, "Describe.")


def test_image_size_boundary() -> None:
    size_mb = IMAGE.stat().st_size / (1024 * 1024)
    VLMRequest(IMAGE, "Describe.", max_image_size_mb=size_mb)
    with pytest.raises(ValueError, match="exceeds"):
        VLMRequest(IMAGE, "Describe.", max_image_size_mb=size_mb / 2)


def test_default_size_rejects_large_file_before_reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "oversized.png"
    with path.open("wb") as stream:
        stream.truncate(10 * 1024 * 1024 + 1)
    def forbidden(*args, **kwargs):
        pytest.fail("oversized image must not be opened")
    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(ValueError, match="exceeds"):
        VLMRequest(path, "Describe.")


@pytest.mark.parametrize("value", [0, -1, True, "10", float("inf"), float("nan"), 10**400])
def test_bad_size_limits(value) -> None:
    with pytest.raises((TypeError, ValueError), match="max_image_size_mb"):
        VLMRequest(IMAGE, "Describe.", max_image_size_mb=value)


@pytest.mark.parametrize("value", ["", " \t\n", None, 123])
def test_invalid_prompt(value) -> None:
    with pytest.raises((TypeError, ValueError), match="prompt"):
        VLMRequest(IMAGE, value)


def response(**overrides) -> VLMResponse:
    fields = dict(request_id="R1", description="[MOCK] Simulated text.", model_name="test-mock",
                  backend="mock", is_mock=True, latency_ms=0,
                  metadata={"nested": [{"text": "测试", "value": None}], "verified": False})
    fields.update(overrides)
    return VLMResponse(**fields)


def test_response_json_serialization_and_independent_metadata() -> None:
    original = {"nested": [1, True, None, "text", 1.5]}
    result = response(metadata=original)
    original["nested"].append("changed")
    assert json.loads(result.to_json()) == result.to_dict()
    assert result.metadata == {"nested": [1, True, None, "text", 1.5]}
    exported = result.to_dict()
    exported["metadata"]["nested"].clear()
    assert result.metadata["nested"]


@pytest.mark.parametrize("metadata", [[], {"x": Path("x")}, {"x": float("nan")},
    {"x": float("inf")}, {1: "value"}, {"nested": {1: "value"}}, {"x": {1, 2}}, {"x": (1, 2)}])
def test_response_rejects_non_json_metadata(metadata) -> None:
    with pytest.raises((TypeError, ValueError), match="metadata"):
        response(metadata=metadata)


def test_response_rejects_cycles_and_invalid_mutation() -> None:
    cyclic = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="metadata"):
        response(metadata=cyclic)
    result = response()
    result.metadata["bad"] = float("nan")
    with pytest.raises(ValueError, match="metadata"):
        result.to_json()


@pytest.mark.parametrize("overrides", [
    {"description": None}, {"description": "A real observation"}, {"is_mock": False},
    {"is_mock": 1}, {"backend": "qwen3_vl"}, {"model_name": " "}, {"request_id": ""},
    {"latency_ms": -1}, {"latency_ms": float("inf")}, {"latency_ms": float("nan")},
    {"latency_ms": True}, {"latency_ms": "1"}, {"latency_ms": 10**400},
])
def test_invalid_response(overrides: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        response(**overrides)


def test_real_backend_contract_does_not_require_mock_label() -> None:
    result = response(backend="future-backend", is_mock=False, description="Unverified generated text.")
    assert result.is_mock is False


def test_mock_deterministic_content_and_dependency_injection(tmp_path: Path) -> None:
    def consumer(backend: BaseVLM, request: VLMRequest) -> VLMResponse:
        return backend.generate(request)
    backend = MockVLM()
    assert isinstance(backend, BaseVLM)
    request = VLMRequest(IMAGE, get_prompt(), request_id="R1")
    first = consumer(backend, request)
    second = consumer(backend, request)
    left, right = first.to_dict(), second.to_dict()
    assert left.pop("latency_ms") >= 0
    assert right.pop("latency_ms") >= 0
    assert left == right
    assert first.request_id == "R1"
    assert first.backend == "mock" and first.is_mock is True
    assert first.metadata["inference_performed"] is False
    path = tmp_path / "flood_in_Paris_2026.png"
    path.write_bytes(IMAGE.read_bytes())
    different = backend.generate(VLMRequest(path, "The claim is true."))
    assert first.description == different.description
    assert "Paris" not in different.description


def test_mock_revalidates_changed_file(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(IMAGE.read_bytes())
    request = VLMRequest(path, get_prompt())
    path.write_bytes(b"replaced with invalid contents")
    with pytest.raises(ValueError, match="content format"):
        MockVLM().generate(request)


def test_mock_cold_import_and_generate_without_gpu_or_network() -> None:
    # Fresh interpreter: guards apply before the VLM package is imported.
    script = '''
import builtins, socket, subprocess, sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'accelerate', 'cuda', 'cupy',
                              'requests', 'http', 'pickle', 'ctypes'} or name == 'urllib.request':
        raise AssertionError('Forbidden dependency: ' + name)
    return original_import(name, *args, **kwargs)
def forbidden(*args, **kwargs):
    raise AssertionError('Network or external process access is forbidden')
builtins.__import__ = guarded_import
socket.socket = socket.create_connection = socket.getaddrinfo = forbidden
subprocess.Popen = forbidden
def audit(event, args):
    if event.startswith('socket.') or event in {'os.system', 'os.spawn', 'subprocess.Popen'}:
        forbidden()
sys.addaudithook(audit)
from visionevidence.vlm import VLMRequest, create_vlm, load_vlm_config
config = load_vlm_config(sys.argv[2])
result = create_vlm(config.backend, model_name=config.model_name).generate(VLMRequest(sys.argv[1], 'Describe.'))
assert result.is_mock and result.backend == 'mock'
print(result.to_json())
'''
    result = subprocess.run([sys.executable, "-c", script, str(IMAGE), str(ROOT / "configs/vlm_mock.toml")],
                            capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["is_mock"] is True


def test_factory_and_default_config() -> None:
    assert isinstance(create_vlm("mock"), MockVLM)
    assert create_vlm("mock", model_name="custom-mock").model_name == "custom-mock"
    assert load_vlm_config(ROOT / "configs/vlm_mock.toml") == load_vlm_config() == VLMConfig()
    with pytest.raises(ValueError, match="model_path"):
        create_vlm("qwen3_vl")
    with pytest.raises(ValueError, match="unsupported backend"):
        create_vlm("unknown")


@pytest.mark.parametrize("content", [
    "[broken", "backend='mock'", "[vlm]\nbackend='mock'", "[vlm]\nunknown=1",
    '[vlm]\nbackend="unknown"\nmodel_name="mock"\nmax_image_size_mb=10\nprompt_version="visual_observation_v1"',
    '[vlm]\nbackend="mock"\nmodel_name=""\nmax_image_size_mb=10\nprompt_version="visual_observation_v1"',
    '[vlm]\nbackend="mock"\nmodel_name="mock"\nmax_image_size_mb=-1\nprompt_version="visual_observation_v1"',
    '[vlm]\nbackend="mock"\nmodel_name="mock"\nmax_image_size_mb=true\nprompt_version="visual_observation_v1"',
    '[vlm]\nbackend="mock"\nmodel_name="mock"\nmax_image_size_mb=10\nprompt_version="unknown"',
])
def test_bad_config(tmp_path: Path, content: str) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid VLM config") as error:
        load_vlm_config(path)
    assert str(path) in str(error.value)


def test_missing_or_invalid_utf8_config(tmp_path: Path) -> None:
    path = tmp_path / "missing.toml"
    with pytest.raises(ValueError, match="invalid VLM config"):
        load_vlm_config(path)
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="invalid VLM config"):
        load_vlm_config(path)


def test_prompt_version() -> None:
    assert get_prompt(PROMPT_VERSION) == get_prompt()
    with pytest.raises(ValueError, match="prompt_version"):
        get_prompt("unknown")

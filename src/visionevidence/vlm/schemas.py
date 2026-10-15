"""VLM contracts and bounded local image checks; no pixel decoding."""

from dataclasses import dataclass, field, asdict
import json
import math
from pathlib import Path
import stat
from typing import Any

from visionevidence.schemas import _nonempty_text


def _positive_size(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("max_image_size_mb must be a finite positive number (MiB)")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError("max_image_size_mb must be finite") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("max_image_size_mb must be a finite positive number (MiB)")
    return value


def _image_format(header: bytes, size: int) -> str | None:
    """Recognize signatures and basic headers, not complete codec validity."""
    if (len(header) >= 33 and header[:8] == b"\x89PNG\r\n\x1a\n"
            and header[8:16] == b"\x00\x00\x00\rIHDR"
            and int.from_bytes(header[16:20], "big") > 0
            and int.from_bytes(header[20:24], "big") > 0):
        return "png"
    if (len(header) >= 6 and header[:3] == b"\xff\xd8\xff"
            and 0xC0 <= header[3] <= 0xFE and not 0xD0 <= header[3] <= 0xD9
            and 2 <= int.from_bytes(header[4:6], "big") <= size - 4):
        return "jpeg"
    if (len(header) >= 20 and header[:4] == b"RIFF" and header[8:12] == b"WEBP"
            and header[12:16] in (b"VP8 ", b"VP8L", b"VP8X")
            and int.from_bytes(header[4:8], "little") + 8 == size
            and 0 < int.from_bytes(header[16:20], "little") <= size - 20):
        return "webp"
    return None


@dataclass(frozen=True)
class VLMRequest:
    """A local image and prompt. max_image_size_mb uses MiB (1024**2 bytes)."""

    image_path: str | Path
    prompt: str
    request_id: str | None = None
    max_image_size_mb: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", _nonempty_text(self.prompt, "prompt"))
        if self.request_id is not None:
            object.__setattr__(self, "request_id", _nonempty_text(self.request_id, "request_id"))
        object.__setattr__(self, "max_image_size_mb", _positive_size(self.max_image_size_mb))
        if not isinstance(self.image_path, (str, Path)):
            raise TypeError("image_path must be a string or Path")
        raw_path = str(self.image_path)
        _nonempty_text(raw_path, "image_path")
        # Reject URL and UNC/network paths before any filesystem access.
        if "://" in raw_path or raw_path.startswith(("\\\\", "//")):
            raise ValueError("image_path must be local; URLs and UNC paths are not supported")
        object.__setattr__(self, "image_path", Path(self.image_path))
        self.validate_image()

    def validate_image(self) -> None:
        """Recheck before use because files can change after request creation."""
        path = Path(self.image_path)
        try:
            info = path.stat()
        except FileNotFoundError as exc:
            raise ValueError(f"image_path must point to an existing regular file: {path}") from exc
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"image_path must point to an existing regular file: {path}")
        supported = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}
        expected = supported.get(path.suffix.lower())
        if expected is None:
            raise ValueError("unsupported image extension; expected PNG, JPEG (.jpg/.jpeg), or WEBP")
        if info.st_size / (1024 * 1024) > self.max_image_size_mb:
            raise ValueError(f"image exceeds max_image_size_mb={self.max_image_size_mb:g} MiB: {path}")
        with path.open("rb") as stream:
            header = stream.read(33)
        actual = _image_format(header, info.st_size)
        if actual != expected:
            raise ValueError(f"image content format does not match {path.suffix} or has an invalid header: {path}")


def _validate_json_metadata(metadata: dict[str, Any]) -> None:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a JSON object")
    # Reject cycles/non-finite values before walking the JSON tree.
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"metadata must be safely JSON serializable: {exc}") from exc

    def check(value: Any) -> None:
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("metadata JSON object keys must be strings")
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
        elif value is not None and not isinstance(value, (str, bool, int, float)):
            raise TypeError("metadata values must use JSON types")

    check(metadata)


@dataclass(frozen=True)
class VLMResponse:
    """Generated text is unverified, including when is_mock is False."""

    request_id: str | None
    description: str
    model_name: str
    backend: str
    is_mock: bool
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.request_id is not None:
            object.__setattr__(self, "request_id", _nonempty_text(self.request_id, "request_id"))
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")
        for name in ("model_name", "backend"):
            object.__setattr__(self, name, _nonempty_text(getattr(self, name), name))
        if not isinstance(self.is_mock, bool):
            raise TypeError("is_mock must be a boolean")
        if (self.backend == "mock") != self.is_mock:
            raise ValueError("backend='mock' and is_mock=True must be used together")
        if self.is_mock and "[MOCK]" not in self.description:
            raise ValueError("mock description must contain [MOCK]")
        if isinstance(self.latency_ms, bool) or not isinstance(self.latency_ms, (int, float)):
            raise TypeError("latency_ms must be a finite non-negative number")
        try:
            latency = float(self.latency_ms)
        except OverflowError as exc:
            raise ValueError("latency_ms must be finite") from exc
        if not math.isfinite(latency) or latency < 0:
            raise ValueError("latency_ms must be a finite non-negative number")
        object.__setattr__(self, "latency_ms", latency)
        _validate_json_metadata(self.metadata)
        # Detach nested containers from caller-owned data.
        object.__setattr__(self, "metadata", json.loads(json.dumps(self.metadata, allow_nan=False)))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe data; recheck metadata in case callers mutated it."""
        _validate_json_metadata(self.metadata)
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, allow_nan=False)

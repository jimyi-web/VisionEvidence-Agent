"""Explicit backend selection and strict standard-library TOML configuration."""

from dataclasses import dataclass, replace
from pathlib import Path
import re
import tomllib

from visionevidence.schemas import _nonempty_text

from .base import BaseVLM
from .mock import MockVLM
from .prompts import PROMPT_VERSION, get_prompt
from .schemas import _positive_size


class BackendUnavailableError(NotImplementedError):
    """A backend's optional runtime dependencies are unavailable."""


class VLMInferenceError(RuntimeError):
    """Loading or inference failed; no successful response is available."""


@dataclass(frozen=True)
class VLMConfig:
    backend: str = "mock"
    model_name: str = "mock-vlm"
    max_image_size_mb: float = 10.0
    prompt_version: str = PROMPT_VERSION
    model_path: str | Path | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    max_new_tokens: int = 256
    do_sample: bool = False
    max_image_pixels: int = 20_000_000
    max_image_side: int = 1280

    def __post_init__(self) -> None:
        for name in ("backend", "model_name", "prompt_version"):
            object.__setattr__(self, name, _nonempty_text(getattr(self, name), name))
        if self.backend not in ("mock", "qwen3_vl"):
            raise ValueError(f"unsupported backend: {self.backend!r}; expected mock or qwen3_vl")
        object.__setattr__(self, "max_image_size_mb", _positive_size(self.max_image_size_mb))
        get_prompt(self.prompt_version)
        if self.model_path is not None:
            if not isinstance(self.model_path, (str, Path)):
                raise TypeError("model_path must be a local path")
            raw = _nonempty_text(str(self.model_path), "model_path")
            if "://" in raw or raw.startswith(("\\\\", "//")):
                raise ValueError("model_path must be local; URLs and UNC paths are not supported")
            object.__setattr__(self, "model_path", Path(raw))
        if not isinstance(self.device, str) or not re.fullmatch(r"cuda(?::[0-9]+)?", self.device):
            raise ValueError("device must be cuda or cuda:N; this backend requires a single CUDA GPU")
        if self.dtype != "bfloat16":
            raise ValueError("dtype must be bfloat16 for this backend")
        for name in ("max_new_tokens", "max_image_pixels", "max_image_side"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_image_side < 32:
            raise ValueError("max_image_side must be at least 32 for Qwen3-VL patches")
        if not isinstance(self.do_sample, bool):
            raise TypeError("do_sample must be a boolean")


def load_vlm_config(path: str | Path | None = None) -> VLMConfig:
    if path is None:
        return VLMConfig()
    path = Path(path)
    try:
        with path.open("rb") as stream:
            config = tomllib.load(stream)
        if config.keys() != {"vlm"} or not isinstance(config["vlm"], dict):
            raise ValueError("config must contain only a [vlm] table")
        required = {"backend", "model_name", "prompt_version"}
        optional = {"max_image_size_mb", "model_path", "device", "dtype", "max_new_tokens",
                    "do_sample", "max_image_pixels", "max_image_side"}
        if config["vlm"].get("backend") != "qwen3_vl":
            required.add("max_image_size_mb")
        missing = required - config["vlm"].keys()
        extra = config["vlm"].keys() - required - optional
        if missing or extra:
            raise ValueError(f"[vlm] missing fields: {sorted(missing)}; unknown fields: {sorted(extra)}")
        return VLMConfig(**config["vlm"])
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid VLM config {path}: {exc}") from exc


def create_vlm(backend: str = "mock", *, model_name: str | None = None,
               config: VLMConfig | None = None) -> BaseVLM:
    selected = config or VLMConfig()
    if model_name is not None:
        selected = replace(selected, model_name=model_name)
    if backend == "mock":
        return MockVLM(model_name=selected.model_name)
    if backend == "qwen3_vl":
        from .qwen3_vl import Qwen3VLBackend

        if config is None and model_name is None:
            selected = replace(selected, model_name="Qwen/Qwen3-VL-4B-Instruct")
        return Qwen3VLBackend(replace(selected, backend=backend))
    raise ValueError(f"unsupported backend: {backend!r}; expected mock or qwen3_vl")

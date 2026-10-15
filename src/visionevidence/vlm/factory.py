"""Explicit backend selection and strict standard-library TOML configuration."""

from dataclasses import dataclass
from pathlib import Path
import tomllib

from visionevidence.schemas import _nonempty_text

from .base import BaseVLM
from .mock import MockVLM
from .prompts import PROMPT_VERSION, get_prompt
from .schemas import _positive_size


class BackendUnavailableError(NotImplementedError):
    """A recognized backend has no implementation in this milestone."""


@dataclass(frozen=True)
class VLMConfig:
    backend: str = "mock"
    model_name: str = "mock-vlm"
    max_image_size_mb: float = 10.0
    prompt_version: str = PROMPT_VERSION

    def __post_init__(self) -> None:
        for name in ("backend", "model_name", "prompt_version"):
            object.__setattr__(self, name, _nonempty_text(getattr(self, name), name))
        if self.backend not in ("mock", "qwen3_vl"):
            raise ValueError(f"unsupported backend: {self.backend!r}; expected mock or qwen3_vl")
        object.__setattr__(self, "max_image_size_mb", _positive_size(self.max_image_size_mb))
        get_prompt(self.prompt_version)


def load_vlm_config(path: str | Path | None = None) -> VLMConfig:
    if path is None:
        return VLMConfig()
    path = Path(path)
    try:
        with path.open("rb") as stream:
            config = tomllib.load(stream)
        if config.keys() != {"vlm"} or not isinstance(config["vlm"], dict):
            raise ValueError("config must contain only a [vlm] table")
        required = {"backend", "model_name", "max_image_size_mb", "prompt_version"}
        missing = required - config["vlm"].keys()
        extra = config["vlm"].keys() - required
        if missing or extra:
            raise ValueError(f"[vlm] missing fields: {sorted(missing)}; unknown fields: {sorted(extra)}")
        return VLMConfig(**config["vlm"])
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid VLM config {path}: {exc}") from exc


def create_vlm(backend: str = "mock", *, model_name: str = "mock-vlm") -> BaseVLM:
    if backend == "mock":
        return MockVLM(model_name=model_name)
    if backend == "qwen3_vl":
        raise BackendUnavailableError(
            "qwen3_vl is not implemented in T03-A; no model was loaded. "
            "Use backend=mock for interface testing."
        )
    raise ValueError(f"unsupported backend: {backend!r}; expected mock or qwen3_vl")

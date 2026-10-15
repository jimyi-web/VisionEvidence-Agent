"""Synchronous dependency-injection boundary for VLM implementations."""

from typing import Protocol, runtime_checkable

from .schemas import VLMRequest, VLMResponse


@runtime_checkable
class BaseVLM(Protocol):
    """Backends return unverified generated text, never verification decisions."""

    def generate(self, request: VLMRequest) -> VLMResponse:
        ...

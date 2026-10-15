"""Predictable test backend with no model, GPU, or network dependencies."""

from time import perf_counter

from visionevidence.schemas import _nonempty_text

from .schemas import VLMRequest, VLMResponse


class MockVLM:
    """Structurally implements BaseVLM; inject an instance into future consumers."""

    def __init__(self, model_name: str = "mock-vlm") -> None:
        self.model_name = _nonempty_text(model_name, "model_name")

    def generate(self, request: VLMRequest) -> VLMResponse:
        started = perf_counter()
        if not isinstance(request, VLMRequest):
            raise TypeError("request must be a VLMRequest")
        request.validate_image()
        description = (
            "[MOCK] Simulated VLM response for interface testing only. "
            "No visual analysis or real model inference was performed. "
            "This text is not an observation or a verified fact."
        )
        return VLMResponse(
            request_id=request.request_id, description=description,
            model_name=self.model_name, backend="mock", is_mock=True,
            latency_ms=(perf_counter() - started) * 1000,
            metadata={"inference_performed": False, "verified": False,
                      "content_source": "fixed_mock_template",
                      "latency_scope": "local validation and mock preparation; excludes response serialization"},
        )

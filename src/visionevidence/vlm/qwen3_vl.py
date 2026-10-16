"""Local-only, single-GPU Qwen3-VL inference. Heavy imports are deliberately lazy."""

from collections.abc import Callable
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any
import warnings

from .factory import BackendUnavailableError, VLMConfig, VLMInferenceError
from .prompts import get_prompt
from .schemas import VLMRequest, VLMResponse


def _import_runtime() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
        # Resolve lazy imports now, so incompatible installations fail clearly.
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: F401
    except (ImportError, OSError, RuntimeError) as exc:
        raise BackendUnavailableError(
            "Qwen3-VL runtime unavailable; see docs/qwen3_vl_deployment.md for the pinned GPU environment: "
            f"{exc}"
        ) from exc
    return torch, transformers


def prepare_image(request: VLMRequest, config: VLMConfig) -> Any:
    """Fully decode a bounded still image; return a detached, EXIF-correct RGB copy."""
    request.validate_image()
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise BackendUnavailableError("Qwen3-VL image decoding requires Pillow==11.3.0") from exc
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(request.image_path) as source:
                if source.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError("only PNG, JPEG, and WEBP are supported")
                if source.width * source.height > config.max_image_pixels:
                    raise ValueError(f"image exceeds max_image_pixels={config.max_image_pixels}")
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("animated/multiframe images are not supported")
                source.verify()
            # verify() invalidates the decoder; reopen and force the entire pixel stream to load.
            with Image.open(request.image_path) as source:
                if source.width * source.height > config.max_image_pixels:
                    raise ValueError(f"image exceeds max_image_pixels={config.max_image_pixels}")
                source.load()
                result = ImageOps.exif_transpose(source).convert("RGB")
                result.thumbnail((config.max_image_side, config.max_image_side), Image.Resampling.LANCZOS)
                return result
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombWarning,
            Image.DecompressionBombError) as exc:
        raise ValueError(f"invalid or unsafe image {request.image_path}: {exc}") from exc


class Qwen3VLBackend:
    """Implements BaseVLM; serialize calls to protect loading and CUDA peak statistics.

    runtime_loader is a test seam for CPU-only fakes, never an automatic fallback.
    """

    def __init__(self, config: VLMConfig, *,
                 runtime_loader: Callable[[], tuple[Any, Any]] = _import_runtime) -> None:
        if config.backend != "qwen3_vl":
            raise ValueError("Qwen3VLBackend requires backend=qwen3_vl")
        if config.model_path is None or not Path(config.model_path).is_dir():
            raise ValueError("qwen3_vl model_path must point to an existing local model directory; "
                             "set model_path in --config. No model will be downloaded.")
        self.config = config
        self.model_path = Path(config.model_path).resolve()
        self._runtime_loader = runtime_loader
        self._lock = Lock()
        self._model = self._processor = self._torch = self._transformers = None
        self._device = ""
        self._load_ms = 0.0

    def _load(self) -> None:
        if self._model is not None:
            return
        started = perf_counter()
        torch, transformers = self._runtime_loader()
        if not torch.cuda.is_available():
            raise VLMInferenceError("qwen3_vl requires CUDA; no GPU is available. No CPU/Mock fallback was used.")
        index = int(self.config.device.split(":")[1]) if ":" in self.config.device else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise VLMInferenceError(f"CUDA device index {index} is unavailable")
        device = f"cuda:{index}"
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise VLMInferenceError(f"{device} does not support native BF16")
            processor = transformers.AutoProcessor.from_pretrained(
                str(self.model_path), local_files_only=True, trust_remote_code=False,
            )
            model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
                str(self.model_path), local_files_only=True, trust_remote_code=False,
                use_safetensors=True, dtype=torch.bfloat16, device_map={"": device},
                attn_implementation="sdpa",
            )
            model.eval()
            torch.cuda.synchronize(device)
        # Publish only a completely loaded pair. A failed load never produces a response.
        self._torch, self._transformers = torch, transformers
        self._processor, self._model, self._device = processor, model, device
        self._load_ms = (perf_counter() - started) * 1000

    def generate(self, request: VLMRequest) -> VLMResponse:
        if not isinstance(request, VLMRequest):
            raise TypeError("request must be a VLMRequest")
        with self._lock:
            started = perf_counter()
            picture = prepare_image(request, self.config)
            preprocess_ms = (perf_counter() - started) * 1000
            loaded_now = self._model is None
            phase = "model loading"
            try:
                self._load()
                torch, processor, model = self._torch, self._processor, self._model
                phase = "inference"
                torch.cuda.synchronize(self._device)
                torch.cuda.reset_peak_memory_stats(self._device)
                started = perf_counter()
                with torch.inference_mode():
                    # Template emits image placeholders; pass only the already decoded PIL image
                    # to the processor. No paths or URLs can be fetched by the processor.
                    messages = [{"role": "user", "content": [
                        {"type": "image"}, {"type": "text", "text": request.prompt},
                    ]}]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(
                        text=[text], images=[picture], return_tensors="pt",
                        images_kwargs={"min_pixels": 32 * 32, "max_pixels": self.config.max_image_side ** 2},
                    ).to(self._device)
                    torch.cuda.synchronize(self._device)
                    generation_started = perf_counter()
                    output = model.generate(
                        **inputs, max_new_tokens=self.config.max_new_tokens,
                        do_sample=self.config.do_sample, num_beams=1, num_return_sequences=1,
                        return_dict_in_generate=False,
                    )
                    torch.cuda.synchronize(self._device)
                    generation_ms = (perf_counter() - generation_started) * 1000
                    if len(output) != 1 or len(inputs["input_ids"]) != 1:
                        raise ValueError("expected exactly one generated sequence")
                    if len(output[0]) < len(inputs["input_ids"][0]):
                        raise ValueError("generated sequence is shorter than the input prompt")
                    trimmed = [output[0][len(inputs["input_ids"][0]):]]
                    description = processor.batch_decode(
                        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                    )[0]
                latency_ms = preprocess_ms + (perf_counter() - started) * 1000
                peak_mb = torch.cuda.max_memory_allocated(self._device) / (1024 ** 2)
                return VLMResponse(
                    model_name=self.config.model_name, backend="qwen3_vl", is_mock=False,
                    description=description, latency_ms=latency_ms, request_id=request.request_id,
                    metadata={
                        "inference_performed": True, "verified": False,
                        "content_source": "unverified_model_generation",
                        "device": self._device, "dtype": self.config.dtype,
                        "torch_version": str(torch.__version__),
                        "transformers_version": str(self._transformers.__version__),
                        "model_path": str(self.model_path),
                        "max_new_tokens": self.config.max_new_tokens,
                        "do_sample": self.config.do_sample, "generated_tokens": len(trimmed[0]),
                        "peak_gpu_memory_mb": peak_mb,
                        "peak_gpu_memory_scope": "torch allocated MiB, includes resident model; excludes load peak",
                        "prompt_version": (self.config.prompt_version
                                           if request.prompt == get_prompt(self.config.prompt_version) else "custom"),
                        "model_load_ms": self._load_ms, "model_loaded_this_call": loaded_now,
                        "generation_ms": generation_ms, "image_preprocessing_ms": preprocess_ms,
                        "latency_scope": "image decoding, processor, transfer, generate and decode; excludes model loading and lock wait",
                        "processed_image_size": list(picture.size),
                        "max_image_pixels": self.config.max_image_pixels,
                        "max_image_side": self.config.max_image_side,
                    },
                )
            except (BackendUnavailableError, VLMInferenceError):
                raise
            except Exception as exc:
                # Preserve the original cause, including CUDA OOM; never substitute mock text.
                raise VLMInferenceError(f"qwen3_vl {phase} failed: {exc}") from exc
            finally:
                picture.close()

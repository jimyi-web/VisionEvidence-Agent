"""Offline CPU tests: real Pillow codecs, injected model/processor/CUDA doubles."""

import builtins
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import json
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image
import pytest

from visionevidence.cli import main
from visionevidence.vlm import (
    BackendUnavailableError, BaseVLM, Qwen3VLBackend, VLMConfig, VLMInferenceError,
    VLMRequest, create_vlm, load_vlm_config,
)
from visionevidence.vlm.prompts import get_prompt
from visionevidence.vlm.qwen3_vl import _import_runtime, prepare_image

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "examples/vlm_demo_fixture.png"


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unit tests must not access the network")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


@pytest.fixture
def config(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    return VLMConfig(backend="qwen3_vl", model_name="Qwen/Qwen3-VL-4B-Instruct", model_path=model_dir)


@pytest.fixture
def runtime():
    state = SimpleNamespace(in_inference=False, moved_to=None, picture=None)

    @contextmanager
    def inference_mode():
        assert not state.in_inference
        state.in_inference = True
        try:
            yield
        finally:
            state.in_inference = False

    cuda = SimpleNamespace(
        is_available=Mock(return_value=True), current_device=Mock(return_value=0),
        device_count=Mock(return_value=2), device=Mock(side_effect=lambda _: nullcontext()),
        is_bf16_supported=Mock(return_value=True), synchronize=Mock(),
        reset_peak_memory_stats=Mock(), max_memory_allocated=Mock(return_value=6 * 1024**2),
    )
    torch = SimpleNamespace(cuda=cuda, bfloat16="bf16-test", __version__="2.8.0+cu128-test",
                            inference_mode=Mock(side_effect=inference_mode))

    class Inputs(dict):
        def to(self, device):
            state.moved_to = device
            return self

    class Processor:
        def __init__(self):
            self.apply_chat_template = Mock(return_value="template with image placeholder")
            self.batch_decode = Mock(return_value=["Unverified generated description."])
            self.calls = []

        def __call__(self, **kwargs):
            assert state.in_inference
            self.calls.append(kwargs)
            state.picture = kwargs["images"][0]
            assert state.picture.mode == "RGB"
            return Inputs(input_ids=[[10, 11, 12]], pixel_values="pixel-tensor", image_grid_thw="grid")

    processor = Processor()

    def generate(**kwargs):
        assert state.in_inference
        assert kwargs["pixel_values"] == "pixel-tensor"
        assert kwargs["image_grid_thw"] == "grid"
        return [[10, 11, 12, 91, 92]]

    model = SimpleNamespace(eval=Mock(), generate=Mock(side_effect=generate))
    transformers = SimpleNamespace(
        __version__="4.57.1-test",
        AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
        Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=Mock(return_value=model)),
    )
    loader = Mock(return_value=(torch, transformers))
    return SimpleNamespace(torch=torch, transformers=transformers, model=model,
                           processor=processor, loader=loader, state=state)


@pytest.mark.parametrize("field,value", [
    ("device", "cpu"), ("device", "auto"), ("device", "cuda:-1"), ("device", None),
    ("dtype", "float16"), ("max_new_tokens", 0), ("max_new_tokens", True),
    ("max_image_pixels", -1), ("max_image_pixels", 1.5), ("max_image_side", 31),
    ("max_image_side", "1280"), ("do_sample", "false"), ("model_path", ""),
    ("model_path", "https://example.com/model"), ("model_path", "//server/model"),
    ("model_path", 123),
])
def test_config_rejects_invalid_settings(field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        VLMConfig(**{field: value})


def test_config_parse_defaults_and_unknown_fields(tmp_path):
    text = (ROOT / "configs/vlm_qwen3_vl.toml").read_text(encoding="utf-8")
    cfg = load_vlm_config(ROOT / "configs/vlm_qwen3_vl.toml")
    assert cfg.model_path == Path("/root/autodl-tmp/VisionEvidence-Agent/models/qwen3-vl-4b-instruct")
    assert cfg.backend == "qwen3_vl" and cfg.dtype == "bfloat16" and not cfg.do_sample
    assert cfg.max_new_tokens == 256 and cfg.max_image_pixels == 20_000_000
    path = tmp_path / "config.toml"
    path.write_text(text + "unexpected = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_vlm_config(path)
    path.write_text(text.replace("do_sample = false", 'do_sample = "false"'), encoding="utf-8")
    with pytest.raises(ValueError, match="do_sample"):
        load_vlm_config(path)


def test_factory_is_lazy_and_implements_protocol(config, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("must not import the GPU runtime during backend selection")
    monkeypatch.setattr("visionevidence.vlm.qwen3_vl._import_runtime", forbidden)
    backend = create_vlm("qwen3_vl", config=config)
    assert isinstance(backend, (Qwen3VLBackend, BaseVLM))
    assert backend._model is None
    assert create_vlm("mock", config=config).model_name == config.model_name


@pytest.mark.parametrize("kind", ["missing", "file", "none"])
def test_model_path_fails_before_loading(config, runtime, tmp_path, kind):
    path = tmp_path / "missing"
    if kind == "file":
        path.write_text("not a directory")
    with pytest.raises(ValueError, match="existing local model directory"):
        Qwen3VLBackend(replace(config, model_path=None if kind == "none" else path), runtime_loader=runtime.loader)
    runtime.loader.assert_not_called()


@pytest.mark.parametrize("suffix,mode", [("png", "RGBA"), ("jpg", "L"), ("webp", "RGB")])
def test_decode_formats_rgb_and_resize_without_modifying_original(config, tmp_path, suffix, mode):
    path = tmp_path / f"image.{suffix}"
    Image.new(mode, (128, 64)).save(path)
    original = path.read_bytes()
    result = prepare_image(VLMRequest(path, "Describe."), replace(config, max_image_side=64))
    assert result.mode == "RGB" and result.size == (64, 32)
    assert path.read_bytes() == original
    result.close()


def test_exif_orientation(config, tmp_path):
    path = tmp_path / "rotated.jpg"
    source = Image.new("RGB", (16, 8), (255, 0, 0))
    exif = source.getexif()
    exif[274] = 6
    source.save(path, exif=exif)
    before = path.read_bytes()
    result = prepare_image(VLMRequest(path, "Describe."), config)
    assert result.size == (8, 16) and result.getexif().get(274) is None
    assert path.read_bytes() == before
    result.close()


def test_pixel_limit_checked_before_decoding(config, tmp_path, monkeypatch):
    path = tmp_path / "large.png"
    Image.new("RGB", (10, 10)).save(path)
    request = VLMRequest(path, "Describe.")
    with prepare_image(request, replace(config, max_image_pixels=100)) as decoded:
        assert decoded.size == (10, 10)
    monkeypatch.setattr("PIL.PngImagePlugin.PngImageFile.load", lambda *a: pytest.fail("oversize decoded"))
    with pytest.raises(ValueError, match="max_image_pixels"):
        prepare_image(request, replace(config, max_image_pixels=99))


@pytest.mark.parametrize("bomb_limit", [60, 30])
def test_decompression_bomb_warning_and_error(config, tmp_path, monkeypatch, bomb_limit):
    path = tmp_path / "bomb.png"
    Image.new("RGB", (10, 10)).save(path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", bomb_limit)
    with pytest.raises(ValueError, match="decompression bomb"):
        prepare_image(VLMRequest(path, "Describe."), config)
    assert Image.MAX_IMAGE_PIXELS == bomb_limit


@pytest.mark.parametrize("suffix", ["png", "jpg", "webp"])
def test_corrupt_images_rejected_before_model_loading(config, runtime, tmp_path, suffix):
    path = tmp_path / f"corrupt.{suffix}"
    Image.new("RGB", (64, 64)).save(path)
    content = path.read_bytes()
    # Retain valid signature so the full decoder, not only VLMRequest, must reject it.
    if suffix == "webp":
        content = content[:24] + b"\xff" * (len(content) - 24)
    else:
        content = content[:-12]
    path.write_bytes(content)
    request = VLMRequest(path, "Describe.")
    backend = Qwen3VLBackend(config, runtime_loader=runtime.loader)
    with pytest.raises(ValueError, match="invalid or unsafe image"):
        backend.generate(request)
    runtime.loader.assert_not_called()


def test_reject_animated_image(config, tmp_path):
    path = tmp_path / "animation.png"
    Image.new("RGB", (8, 8), "red").save(
        path, save_all=True, append_images=[Image.new("RGB", (8, 8), "blue")], duration=100,
    )
    with pytest.raises(ValueError, match="animated"):
        prepare_image(VLMRequest(path, "Describe."), config)


def test_generation_contract_trimming_metadata_and_reuse(config, runtime):
    backend = Qwen3VLBackend(config, runtime_loader=runtime.loader)
    request = VLMRequest(IMAGE, get_prompt(), request_id="request-1")
    first = backend.generate(request)
    second = backend.generate(request)
    assert json.loads(first.to_json())["backend"] == "qwen3_vl"
    assert first.request_id == "request-1" and first.is_mock is False
    assert first.description == "Unverified generated description."
    metadata = first.metadata
    assert metadata["inference_performed"] is True and metadata["verified"] is False
    assert metadata["device"] == "cuda:0" and metadata["generated_tokens"] == 2
    assert metadata["peak_gpu_memory_mb"] == 6
    assert metadata["prompt_version"] == config.prompt_version
    assert metadata["torch_version"] == "2.8.0+cu128-test"
    assert metadata["transformers_version"] == "4.57.1-test"
    assert metadata["model_loaded_this_call"] and not second.metadata["model_loaded_this_call"]
    assert metadata["model_load_ms"] == second.metadata["model_load_ms"] >= 0
    assert 0 <= metadata["generation_ms"] <= first.latency_ms
    runtime.loader.assert_called_once()
    runtime.model.eval.assert_called_once()
    assert runtime.model.generate.call_count == runtime.torch.inference_mode.call_count == 2
    runtime.processor.batch_decode.assert_called_with(
        [[91, 92]], skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    runtime.transformers.AutoProcessor.from_pretrained.assert_called_once_with(
        str(config.model_path.resolve()), local_files_only=True, trust_remote_code=False,
    )
    runtime.transformers.Qwen3VLForConditionalGeneration.from_pretrained.assert_called_once_with(
        str(config.model_path.resolve()), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=runtime.torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa",
    )
    generated_args = runtime.model.generate.call_args.kwargs
    assert generated_args["max_new_tokens"] == 256 and generated_args["do_sample"] is False
    assert generated_args["num_beams"] == generated_args["num_return_sequences"] == 1
    assert generated_args["return_dict_in_generate"] is False
    messages = runtime.processor.apply_chat_template.call_args.args[0]
    assert messages[0]["content"] == [{"type": "image"}, {"type": "text", "text": get_prompt()}]
    assert runtime.state.moved_to == "cuda:0"
    runtime.torch.cuda.reset_peak_memory_stats.assert_called_with("cuda:0")
    assert runtime.torch.cuda.reset_peak_memory_stats.call_count == 2
    assert not runtime.state.in_inference


def test_load_time_excluded_from_latency(config, runtime, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("visionevidence.vlm.qwen3_vl.perf_counter", lambda: clock[0])
    def load():
        clock[0] += 10.0
        return runtime.torch, runtime.transformers
    original_generate = runtime.model.generate.side_effect
    def generate(**kwargs):
        clock[0] += 0.5
        return original_generate(**kwargs)
    runtime.model.generate.side_effect = generate
    response = Qwen3VLBackend(config, runtime_loader=load).generate(VLMRequest(IMAGE, get_prompt()))
    assert response.metadata["model_load_ms"] == 10000
    assert response.latency_ms == response.metadata["generation_ms"] == 500


def test_token_comparison_reuses_one_loaded_model(config, runtime):
    backend = Qwen3VLBackend(config, runtime_loader=runtime.loader)
    results = [backend.generate(VLMRequest(IMAGE, get_prompt()), max_new_tokens=limit)
               for limit in [256, 512] * 3]
    runtime.loader.assert_called_once()
    runtime.transformers.Qwen3VLForConditionalGeneration.from_pretrained.assert_called_once()
    assert [r.metadata["model_loaded_this_call"] for r in results] == [True] + [False] * 5
    assert [c.kwargs["max_new_tokens"] for c in runtime.model.generate.call_args_list] == [256, 512] * 3
    assert backend.config.max_new_tokens == 256
    with pytest.raises(ValueError, match="positive integer"):
        backend.generate(VLMRequest(IMAGE, get_prompt()), max_new_tokens=True)


def test_experiment_real_path_with_cpu_doubles(config, runtime, tmp_path, monkeypatch):
    # These are synthetic runtime doubles, NOT real GPU experimental evidence.
    import sys
    from visionevidence.vlm import experiment

    (config.model_path / "config.json").write_text('{}')
    runtime.torch.manual_seed = Mock()
    runtime.torch.version = SimpleNamespace(cuda="synthetic")
    monkeypatch.setitem(sys.modules, "torch", runtime.torch)
    monkeypatch.setattr(experiment, "load_vlm_config", lambda _: config)
    backend = Qwen3VLBackend(config, runtime_loader=runtime.loader)
    factory = Mock(return_value=backend)
    monkeypatch.setattr(experiment, "create_vlm", factory)
    monkeypatch.setattr(experiment, "gpu_snapshot", lambda: [{"uuid": "synthetic-GPU"}])
    output = tmp_path / "synthetic-results"
    result = experiment.run_experiment(ROOT / "configs/vlm_qwen3_vl.toml", IMAGE, output)
    records = [json.loads(line) for line in (output / "runs.jsonl").read_text().splitlines()]
    assert result["status"] == "completed"
    assert [r["phase"] for r in records] == ["cold"] + ["reuse"] * 5
    assert [r["response"]["metadata"]["max_new_tokens"] for r in records] == [256, 512] * 3
    assert all(r["gpu_samples"] and not r["gpu_sampling_errors"] for r in records)
    factory.assert_called_once()
    runtime.loader.assert_called_once()
    runtime.torch.manual_seed.assert_called_once_with(0)


def test_custom_generation_and_current_cuda_device(config, runtime):
    cfg = replace(config, device="cuda", max_new_tokens=17, do_sample=True)
    runtime.torch.cuda.current_device.return_value = 1
    result = Qwen3VLBackend(cfg, runtime_loader=runtime.loader).generate(VLMRequest(IMAGE, "Custom prompt"))
    assert result.metadata["prompt_version"] == "custom"
    assert result.metadata["device"] == runtime.state.moved_to == "cuda:1"
    assert result.metadata["do_sample"] is True and result.metadata["max_new_tokens"] == 17
    assert runtime.model.generate.call_args.kwargs["do_sample"] is True
    runtime.torch.cuda.device.assert_called_once_with("cuda:1")
    runtime.torch.cuda.is_bf16_supported.assert_called_once_with(including_emulation=False)


@pytest.mark.parametrize("failure,match", [("no_gpu", "no GPU"), ("no_bf16", "BF16"), ("invalid_device", "index")])
def test_gpu_preflight_does_not_load_model(config, runtime, failure, match):
    if failure == "no_gpu":
        runtime.torch.cuda.is_available.return_value = False
    elif failure == "no_bf16":
        runtime.torch.cuda.is_bf16_supported.return_value = False
    else:
        config = replace(config, device="cuda:7")
    backend = Qwen3VLBackend(config, runtime_loader=runtime.loader)
    with pytest.raises(VLMInferenceError, match=match):
        backend.generate(VLMRequest(IMAGE, get_prompt()))
    runtime.transformers.Qwen3VLForConditionalGeneration.from_pretrained.assert_not_called()


@pytest.mark.parametrize("phase", ["processor_load", "model_load", "generation", "decode"])
def test_failures_propagate_without_mock_success(config, runtime, phase):
    target = {
        "processor_load": runtime.transformers.AutoProcessor.from_pretrained,
        "model_load": runtime.transformers.Qwen3VLForConditionalGeneration.from_pretrained,
        "generation": runtime.model.generate,
        "decode": runtime.processor.batch_decode,
    }[phase]
    target.side_effect = RuntimeError("simulated CUDA OOM or corrupt model")
    backend = Qwen3VLBackend(config, runtime_loader=runtime.loader)
    with pytest.raises(VLMInferenceError, match="failed.*simulated") as error:
        backend.generate(VLMRequest(IMAGE, get_prompt()))
    assert isinstance(error.value.__cause__, RuntimeError)
    assert not runtime.state.in_inference
    if "load" in phase:
        assert backend._model is None and backend._processor is None


def test_missing_optional_dependencies_are_actionable(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch intentionally unavailable")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(BackendUnavailableError, match="qwen3_vl_deployment.md"):
        _import_runtime()


def test_missing_pillow_is_actionable(config, monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "PIL":
            raise ImportError("Pillow intentionally unavailable")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(BackendUnavailableError, match="Pillow"):
        prepare_image(VLMRequest(IMAGE, get_prompt()), config)


@pytest.mark.parametrize("output", [[], [[10, 11]], [[10, 11, 12, 91], [10, 11, 12, 92]]])
def test_malformed_generation_rejected(config, runtime, output):
    runtime.model.generate.side_effect = None
    runtime.model.generate.return_value = output
    backend = Qwen3VLBackend(config, runtime_loader=runtime.loader)
    with pytest.raises(VLMInferenceError, match="sequence"):
        backend.generate(VLMRequest(IMAGE, get_prompt()))


def test_deleted_image_and_bad_request_do_not_load(config, runtime, tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(IMAGE.read_bytes())
    request = VLMRequest(path, get_prompt())
    backend = Qwen3VLBackend(config, runtime_loader=runtime.loader)
    path.unlink()
    with pytest.raises(ValueError, match="existing regular file"):
        backend.generate(request)
    with pytest.raises(TypeError, match="VLMRequest"):
        backend.generate(None)
    runtime.loader.assert_not_called()


@pytest.mark.parametrize("as_json", [False, True])
def test_cli_real_backend_contract_with_injected_runtime(config, runtime, monkeypatch, capsys, as_json):
    backend = Qwen3VLBackend(config, runtime_loader=runtime.loader)
    def factory(name, *, config):
        assert name == "qwen3_vl" and config.backend == "qwen3_vl"
        return backend
    monkeypatch.setattr("visionevidence.cli.create_vlm", factory)
    args = ["vlm", "--image", str(IMAGE), "--backend", "qwen3_vl", "--request-id", "cli-test"]
    if as_json:
        args += ["--json"]
    assert main(args) == 0
    result = capsys.readouterr()
    assert not result.err
    if as_json:
        data = json.loads(result.out)
        assert data["backend"] == "qwen3_vl" and data["is_mock"] is False
        assert data["request_id"] == "cli-test" and data["metadata"]["verified"] is False
    else:
        assert "REAL (unverified model output)" in result.out and "is_mock=False" in result.out


def test_cli_runtime_failure_has_no_success_json(config, runtime, monkeypatch, capsys):
    runtime.model.generate.side_effect = RuntimeError("CUDA out of memory")
    backend = Qwen3VLBackend(config, runtime_loader=runtime.loader)
    monkeypatch.setattr("visionevidence.cli.create_vlm", lambda *a, **k: backend)
    with pytest.raises(SystemExit) as error:
        main(["vlm", "--image", str(IMAGE), "--backend", "qwen3_vl", "--json"])
    assert error.value.code == 2
    result = capsys.readouterr()
    assert result.out == "" and "CUDA out of memory" in result.err and "Traceback" not in result.err

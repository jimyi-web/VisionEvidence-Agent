"""V1.1 contracts and leakage/durability checks, using only synthetic CPU inputs."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from visionevidence.batch import StructuredMock, run_batch
from visionevidence.caption_experiment import run_experiment
from visionevidence.observation import PROMPT, build_query, observe_and_retrieve, parse_observation
from visionevidence.observation_policy import ObservationPolicy, V10_POLICY, load_observation_policy
from visionevidence.retrieval.bm25 import BM25Index
from visionevidence.schemas import Evidence
from visionevidence.vlm.schemas import VLMRequest
from tests.test_batch import setup_run
from tests.test_caption_experiment import bundle

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "examples/vlm_demo_fixture.png"
POLICIES = [ObservationPolicy(f, d) for f in (False, True) for d in (False, True)]


def payload(objects=None, visible_text=None, uncertainties=None):
    return json.dumps({"objects": objects if objects is not None else ["two red cars crossing a bridge"],
                       "visible_text": visible_text or [],
                       "uncertainties": uncertainties if uncertainties is not None else ["[MOCK] synthetic fixture"]})


class Replay(StructuredMock):
    def __init__(self, text, *, real=False, tokens=50):
        self.text, self.real, self.tokens, self.calls = text, real, tokens, 0

    def generate(self, request):
        self.calls += 1
        response = super().generate(request)
        return replace(response, description=self.text, is_mock=not self.real,
                       backend="fixture" if self.real else "mock",
                       metadata={"inference_performed": self.real, "generated_tokens": self.tokens,
                                 "max_new_tokens": 384})


class CountingIndex(BM25Index):
    def __init__(self):
        super().__init__([Evidence("d1", "synthetic://doc", "two red cars crossing bridge 27")])
        self.calls = 0

    def search(self, *args, **kwargs):
        self.calls += 1
        return super().search(*args, **kwargs)


def test_v10_prompt_keeps_frozen_bytes_and_legacy_eight_items():
    assert hashlib.sha256(PROMPT.encode()).hexdigest() == "dde82bfda2d224991529b789abbabeae1cce72d8caf3ea8d2d401c41101cd0e6"
    assert V10_POLICY.prompt(PROMPT) == PROMPT
    observation = parse_observation(payload([f"item {n}" for n in range(8)]))
    V10_POLICY.validate(observation)
    with pytest.raises(ValueError, match="at most 8"):
        parse_observation(payload([f"item {n}" for n in range(9)]))


@pytest.mark.parametrize("name,flags", [("v10_control", (False, False)), ("format_only", (True, False)),
                                       ("details_only", (False, True)), ("format_details", (True, True))])
def test_versioned_configs_have_independent_switches(name, flags):
    policy = load_observation_policy(ROOT / f"configs/v11/{name}.toml")
    assert policy == ObservationPolicy(*flags)
    prompt = policy.prompt(PROMPT)
    assert ("Additional format contract:" in prompt) == flags[0]
    assert ("Additional information contract:" in prompt) == flags[1]


@pytest.mark.parametrize("text", [
    '[observation]\nconfig_version="unknown"\nformat_constraints=true\nretain_visual_details=true',
    '[observation]\nconfig_version="observation_policy_v1"\nformat_constraints=1\nretain_visual_details=false',
    '[observation]\nconfig_version="observation_policy_v1"\nformat_constraints=true',
    '[observation]\nconfig_version="observation_policy_v1"\nformat_constraints=true\nretain_visual_details=false\nextra=true',
])
def test_invalid_policy_configs_fail_closed(tmp_path, text):
    path = tmp_path / "invalid.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid observation policy"):
        load_observation_policy(path)


@pytest.mark.parametrize("text,error", [
    (payload([f"item {n}" for n in range(7)]), "at most 6"),
    (payload(["red car", " RED   CAR "]), "duplicate objects"),
    (payload(["a" * 101]), "100 characters"),
    (payload([" ".join(["word"] * 16) + str(i) for i in range(4)]), "60 words"),
    (payload(visible_text=["27", "28", "29"]), "at most 2"),
    (payload(uncertainties=["[MOCK]", "unknown", "not legible"]), "at most 2"),
    ('{"objects":["[MOCK] car",', "invalid observation JSON"),
    (payload([f"item {n}" for n in range(9)]), "at most 8"),
])
def test_rejected_output_is_not_silently_trimmed_or_repaired(text, error):
    policy, model, index = ObservationPolicy(True, True), Replay(text), CountingIndex()
    request = VLMRequest(IMAGE, policy.prompt(PROMPT), "synthetic")
    with pytest.raises(ValueError, match=error):
        observe_and_retrieve(request, model, index, policy=policy)
    assert model.text == text and model.calls == 1 and index.calls == 0


@pytest.mark.parametrize("policy", POLICIES)
def test_truncation_rejected_even_when_json_would_be_valid(policy):
    index = CountingIndex()
    with pytest.raises(ValueError, match="token limit reached"):
        observe_and_retrieve(VLMRequest(IMAGE, policy.prompt(PROMPT), "synthetic"),
                             Replay(payload(), real=True, tokens=384), index, policy=policy)
    assert index.calls == 0


def test_query_preserves_action_attributes_scene_count_and_digits_without_expansion():
    text = payload(["two red cars crossing a bridge", "wet city road"], ["27"], ["[MOCK] location unknown"])
    policy, index = ObservationPolicy(True, True), CountingIndex()
    result = observe_and_retrieve(VLMRequest(IMAGE, policy.prompt(PROMPT), "synthetic"),
                                  Replay(text), index, policy=policy)
    assert result["query"] == "two red cars crossing a bridge wet city road 27"
    assert "unknown" not in result["query"] and "vehicle" not in result["query"]
    assert result["query_sources"][-1] == {"field": "visible_text", "index": 0, "terms": ["27"]}
    assert result["trace"]["observation_policy"] == policy.identity()
    assert result["vlm_response"]["description"] == text
    assert build_query(parse_observation(text))[0] == result["query"]


@pytest.mark.parametrize("policy", POLICIES)
def test_all_ablations_mock_pair_resume_and_identity(tmp_path, policy):
    _, dataset = bundle(tmp_path)
    requests = []
    class Spy(StructuredMock):
        def generate(self, request):
            requests.append(request.prompt)
            assert "HUMAN_REFERENCE_SECRET" not in request.prompt
            return super().generate(request)
    output = tmp_path / "run"
    result = run_experiment(dataset=dataset, output=output, limit=2,
                            observation_policy=policy, backend_factory=Spy)
    assert result["status"] == "completed" and result["scale"]["real_vlm_image_calls"] == 0
    assert requests[-1] == policy.prompt(PROMPT)
    identity = json.loads((output / "structured/manifest.json").read_text())["identity"]
    assert identity["observation_policy"] == policy.identity()
    run_experiment(dataset=dataset, output=output, limit=2, observation_policy=policy, resume=True,
                   backend_factory=lambda: pytest.fail("cache must not call model"))
    assert len(requests) == 4
    with pytest.raises(ValueError, match="identity changed"):
        run_experiment(dataset=dataset, output=output, limit=2, resume=True,
                       observation_policy=ObservationPolicy(not policy.format_constraints, policy.retain_visual_details))


def test_failure_raw_output_exported_and_denominator_preserved(tmp_path):
    args = setup_run(tmp_path, count=1)
    text = payload(["car", " CAR "])
    result = run_batch(**args, max_attempts=1, backend_factory=lambda: Replay(text),
                       observation_policy=ObservationPolicy(True, False))
    assert result["status"] == "incomplete" and result["sample_status"] == {"failed": 1}
    exported = json.loads((args["output"] / "attempts-0001.jsonl").read_text())
    assert exported["status"] == "failed" and "duplicate objects" in exported["error"]
    assert exported["raw_response"]["description"] == text
    with sqlite3.connect(args["output"] / "state.sqlite3") as db:
        assert db.execute("SELECT result FROM samples").fetchone()[0] is None


def test_details_flag_changes_generation_contract_not_word_injection():
    observation = parse_observation(payload(["object"], uncertainties=[]))
    assert build_query(observation)[0] == "object"
    ObservationPolicy(False, True).validate(observation)
    assert build_query(observation)[0] == "object"  # Do not invent absent actions/counts.


def test_policy_cli_and_failure_exit_code(tmp_path):
    args = setup_run(tmp_path, count=1)
    args["samples"].write_text(args["samples"].read_text().replace("image.png", "missing.png"))
    command = [sys.executable, "-m", "visionevidence.batch", "--mode", "mock", "--max-attempts", "1",
               "--observation-config", str(ROOT / "configs/v11/format_details.toml")]
    for key, value in args.items():
        command.extend([f"--{key}", str(value)])
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1 and json.loads(result.stdout)["status"] == "incomplete"
    assert json.loads((args["output"] / "attempts-0001.jsonl").read_text())["raw_response"] is None

"""Audit invariants: keep syntax, schema, truncation and cohorts distinct."""

import json

import pytest

from scripts.audit_flickr8k_gpu import aggregate, metrics, response_checks, strict_json


def response(text, generated=30):
    return {"description": text, "metadata": {"generated_tokens": generated, "max_new_tokens": 384}}


def observation(objects):
    return json.dumps({"objects": objects, "visible_text": [], "uncertainties": []})


def test_valid_json_can_fail_schema_without_truncation():
    result = response_checks(response(observation([str(i) for i in range(9)])))
    assert result["json_valid"] and not result["schema_valid"]
    assert result["category"] == "schema" and not result["token_limit_reached"]


def test_truncation_is_root_category_and_json_is_independent_symptom():
    result = response_checks(response('{"objects":["dog",', generated=384))
    assert result["category"] == "token_limit"
    assert not result["json_valid"] and "Expecting value" in result["json_error"]


def test_complete_json_at_limit_still_rejected_by_original_policy():
    result = response_checks(response(observation(["dog"]), generated=384))
    assert result["json_valid"] and result["schema_valid"]
    assert result["category"] == "token_limit"


def test_syntax_failure_below_limit_is_separate():
    result = response_checks(response('```json\n' + observation(["dog"]) + '\n```'))
    assert result["category"] == "json_syntax" and not result["token_limit_reached"]


def test_valid_schema_does_not_guarantee_supported_query_terms():
    result = response_checks(response(observation(["小狗"])))
    assert result["json_valid"] and result["schema_valid"]
    assert not result["query_valid"] and result["category"] == "query"


@pytest.mark.parametrize("text", ['{"objects":[],"objects":["dog"]}', '{"value":NaN}'])
def test_strict_json_does_not_silently_accept_duplicate_keys_or_nan(text):
    with pytest.raises(ValueError):
        strict_json(text)


def test_failed_query_keeps_denominator_and_caption_recall_distinct_from_hit():
    relevant = frozenset({"a", "b", "c", "d", "e"})
    success = {"metrics": metrics(["x", "a", "b"], relevant)}
    failure = {"metrics": metrics([], relevant)}
    all_selected = aggregate([success, failure])
    assert all_selected["denominator"] == 2
    assert all_selected["metrics"]["5"] == {"hit_rate": 0.5, "mrr": 0.25, "caption_recall": 0.2}
    assert aggregate([success])["metrics"]["5"]["hit_rate"] == 1.0

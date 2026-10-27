"""Analytic arithmetic tests only; no synthetic experimental results or VLM calls."""

import importlib.util
from pathlib import Path

import pytest

from visionevidence.evaluation.caption_pair import paired_hit_interval
from visionevidence.observation import parse_observation

spec = importlib.util.spec_from_file_location("audit_v11_independent", Path(__file__).parents[1]/"scripts/audit_v11_independent.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_metrics_and_failure_denominator():
    qrels = list("abcde")
    ranked = ["x","a","y","z","b","c","w","v","u","d"]
    assert audit.score(ranked,qrels,"completed") == {"Hit@1":0.,"Hit@5":1.,"Hit@10":1.,"Recall@10":.8,"MRR@10":.5}
    assert all(x == 0 for x in audit.score(ranked,qrels,"failed").values())


def test_rank_ten_and_eleven_boundary():
    qrels = list("abcde")
    assert audit.score([str(x) for x in range(9)]+["a"],qrels,"completed")["MRR@10"] == .1
    assert audit.score([str(x) for x in range(10)]+["a"],qrels,"completed")["Hit@10"] == 0


def test_reject_duplicate_ranks_and_invalid_qrels():
    with pytest.raises(ValueError,match="duplicate ranked"):
        audit.score(["a","a"],list("abcde"),"completed")
    with pytest.raises(ValueError,match="five distinct"):
        audit.score([],list("aabcd"),"failed")
    with pytest.raises(ValueError,match="nonterminal"):
        audit.score([],list("abcde"),"pending")


def test_bootstrap_preserves_frozen_binary_method():
    baseline, structured = [0,1,1,0,1], [1,0,1,1,0]
    pairs = [{"delta":{m:s-b for m in audit.METRICS}} for b,s in zip(baseline,structured)]
    result=audit.bootstrap(pairs)
    frozen=paired_hit_interval(baseline,structured)
    assert {k:result["Hit@10"][k] for k in ("difference","lower","upper")} == {k:frozen[k] for k in ("difference","lower","upper")}
    assert all(result[m] == result["Hit@10"] for m in audit.METRICS)


def test_fractional_constant_difference_and_paired_cancellation():
    pairs=[{"delta":{m:.125 for m in audit.METRICS}} for _ in range(3)]
    ci=audit.bootstrap(pairs,repetitions=100)
    assert all(r["lower"] == r["upper"] == r["difference"] == .125 for r in ci.values())
    zeros=[{"delta":{m:0 for m in audit.METRICS}} for _ in range(3)]
    assert all(r["lower"] == r["upper"] == 0 for r in audit.bootstrap(zeros,100).values())


def test_all_contract_violations_not_just_first_error():
    observation=parse_observation('{"objects":[],"visible_text":["a","b","c"],"uncertainties":["x","y","z"]}')
    assert audit.contract_violations(observation) == ["visible_text: 3 > 2","uncertainties: 3 > 2"]

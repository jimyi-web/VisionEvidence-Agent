"""Paired image uncertainty, independent of any real test observations."""

import pytest

from visionevidence.evaluation.caption_pair import paired_hit_interval, report
from visionevidence.caption_experiment import run_experiment
from visionevidence.datasets.flickr8k import adapt_flickr8k
from tests.test_flickr8k import make_raw


def test_bootstrap_preserves_pairing_and_includes_failure_zeros():
    base, selected = [1, 0, 1, 0], [1, 1, 0, 0]
    result = paired_hit_interval(base, selected, repetitions=1000)
    assert result == paired_hit_interval(base, selected, repetitions=1000)
    assert result["difference"] == 0
    assert result["lower"] < 0 < result["upper"]
    tied = paired_hit_interval(base, base, repetitions=1000)
    assert tied["lower"] == tied["upper"] == 0
    positive = paired_hit_interval([0] * 4, [1] * 4, repetitions=1000)
    assert positive["lower"] == positive["upper"] == positive["difference"] == 1


@pytest.mark.parametrize("base,selected", [([], []), ([1], []), ([2], [1])])
def test_invalid_pairs_rejected(base, selected):
    with pytest.raises(ValueError):
        paired_hit_interval(base, selected)


def test_final_report_rejects_mock_provenance(tmp_path):
    raw = make_raw(tmp_path)
    adapter = tmp_path / "adapter"
    adapt_flickr8k(input_root=raw, output=adapter)
    run = tmp_path / "mock"
    run_experiment(dataset=adapter / "dev", output=run, limit=2)
    with pytest.raises(ValueError, match="real run"):
        report(run, 2)

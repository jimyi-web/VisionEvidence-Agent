"""Complete, explicit human labels must remain distinct from historical qrels."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'experiments/t04_v1/human-review-20261018-01'
FROZEN = ROOT / 'experiments/t04_v1/run-20261017-01'
spec = importlib.util.spec_from_file_location('human_review_validator', ROOT / 'scripts/validate_t04_human_review.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def inputs():
    return (
        validator.read_jsonl(REVIEW / 'judgments.jsonl'),
        json.loads((REVIEW / 'confirmation.json').read_text(encoding='utf-8')),
        validator.read_jsonl(FROZEN / 'qrels.jsonl'),
        validator.read_jsonl(FROZEN / 'corpus.jsonl'),
    )


def test_complete_archive_and_historical_integrity():
    result = validator.validate(ROOT, REVIEW)
    assert (result['judgments'], result['relevant'], result['not_relevant'], result['uncertain']) == (120, 11, 109, 0)
    assert result['matches'] == 120 and result['mismatches'] == []
    rows, confirmation, _, _ = inputs()
    assert confirmation['human_reviewers'] == 1
    assert confirmation['independent_blind_review'] is False
    assert confirmation['multi_annotator_study'] is False
    assert sum(r['sample_id'] == 'S09' and r['label'] == 'not_relevant' for r in rows) == 12
    assert '部分覆盖' in confirmation['samples'][2]['note']


@pytest.mark.parametrize('mutation, message', [
    ('missing', 'missing or unknown'), ('duplicate', 'duplicate'),
    ('unknown', 'missing or unknown'), ('invalid', 'invalid label'),
    ('flipped', 'differs from user'), ('text', 'evidence text mismatch'),
    ('source', 'evidence source mismatch'),
])
def test_reject_damaged_judgments(mutation, message):
    rows, confirmation, qrels, corpus = inputs()
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'duplicate':
        rows.append(deepcopy(rows[0]))
    elif mutation == 'unknown':
        rows[0]['document_id'] = 'D99'
    elif mutation == 'invalid':
        rows[0]['label'] = 'yes'
    elif mutation == 'flipped':
        rows[0]['label'] = 'not_relevant'
    elif mutation == 'text':
        rows[0]['evidence_text'] = 'changed'
    else:
        rows[0]['evidence_source'] = 'changed'
    with pytest.raises(ValueError, match=message):
        validator.compare(rows, confirmation, qrels, corpus)


def test_comparison_reports_disagreement_without_relabeling():
    rows, confirmation, qrels, corpus = inputs()
    before = deepcopy(rows)
    qrels[0]['relevant_ids'].remove('D01')
    result = validator.compare(rows, confirmation, qrels, corpus)
    assert result['matches'] == 119
    assert result['mismatches'] == [dict(sample_id='S01', document_id='D01', human='relevant', frozen='not_relevant')]
    assert rows == before


def test_validator_rejects_hash_tampering(tmp_path):
    (tmp_path / 'review').mkdir()
    (tmp_path / 'target').write_text('changed', encoding='utf-8')
    (tmp_path / 'review/manifest.json').write_text(json.dumps({'sha256': {'target': '0' * 64}}), encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        validator.validate(tmp_path, tmp_path / 'review')

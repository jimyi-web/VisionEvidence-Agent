"""Audit must reject changed archives and preserve human-entered review files."""

import importlib.util
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('t04_audit', ROOT / 'scripts/audit_t04_5.py')
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)
ARCHIVE = ROOT / 'experiments/t04_v1/run-20261017-01'


def test_frozen_replay_and_complete_review(tmp_path):
    output = tmp_path / 'review'
    result = audit_module.audit(ARCHIVE, output)
    assert result['replay_matches'] and result['term_sums_match']
    assert result['summary']['Structured']['evaluated'] == 9
    review = (output / 'human_review.md').read_text(encoding='utf-8')
    assert sum(line.startswith('## S') for line in review.splitlines()) == 10
    assert review.count('| 待填写 |') == 120
    assert 'Platform 7' in review and 'automobile' in review


def test_tampered_archive_rejected_before_export(tmp_path):
    archive = tmp_path / 'archive'
    shutil.copytree(ARCHIVE, archive)
    with (archive / 'qrels.jsonl').open('a', encoding='utf-8') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='hash mismatch: qrels.jsonl'):
        audit_module.audit(archive, tmp_path / 'review')
    assert not (tmp_path / 'review').exists()


def test_existing_human_review_never_overwritten(tmp_path):
    output = tmp_path / 'review'
    output.mkdir()
    review = output / 'human_review.md'
    review.write_text('human notes', encoding='utf-8')
    with pytest.raises(FileExistsError):
        audit_module.audit(ARCHIVE, output)
    assert review.read_text(encoding='utf-8') == 'human notes'

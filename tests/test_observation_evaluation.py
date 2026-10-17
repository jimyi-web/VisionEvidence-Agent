import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from visionevidence.evaluation.observation import (
    construct_query, load_judgments, load_samples, rank_samples, run, score_rows,
)
from visionevidence.retrieval import BM25Index
from visionevidence.schemas import Evidence

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/evaluation/t04_v1'
CONFIG = ROOT / 'configs/t04_evaluation.toml'


def sample(sid='s', description='rain', objects=None):
    return dict(sample_id=sid, original_description=description,
                structured=dict(objects=['rain'] if objects is None else objects,
                                visible_text=[], uncertainties=['secret evidence B']))


def test_label_isolation_and_hand_metrics():
    index = BM25Index([Evidence(i, 'synthetic', 'rain') for i in ('A', 'B', 'C', 'D')])
    ranks = rank_samples(index, [sample(), sample('failed', '中文', []), sample('excluded')])
    judgments = {sid: dict(sample_id=sid, relevant_ids=ids, evidence_status='available' if ids else 'unjudged', rationale='test')
                 for sid, ids in [('s', ['B', 'D']), ('failed', ['A']), ('excluded', [])]}
    result = score_rows(ranks, judgments)
    assert result['Baseline']['end_to_end'] == dict(recall_at_1=0, recall_at_3=.25, recall_at_5=.5, mrr_at_5=.25)
    assert result['Structured']['end_to_end'] == result['Baseline']['end_to_end']
    assert result['Structured']['evaluated'] == 2
    assert result['Structured']['successful_query_count'] == 1
    before = [(r['query'], r['evidence']) for r in ranks]
    judgments['s']['relevant_ids'] = ['A']
    score_rows(ranks, judgments)
    assert before == [(r['query'], r['evidence']) for r in ranks]
    assert construct_query('rain', sample()['structured'], 'Structured')[0] == 'rain'


def test_failures_are_distinct():
    index = BM25Index([Evidence('A', 'synthetic', 'rain')])
    rows = rank_samples(index, [sample('miss', 'snow', ['snow']), sample('bad', '中文', []), sample('absent')])
    judgments = {sid: dict(sample_id=sid, relevant_ids=[] if sid == 'absent' else ['A'],
                           evidence_status='corpus_insufficient' if sid == 'absent' else 'available', rationale='test')
                 for sid in ('miss', 'bad', 'absent')}
    score_rows(rows, judgments)
    assert rows[0]['diagnostics'] == ['retrieval_no_hit']
    assert rows[2]['diagnostics'] == ['query_construction_error']
    assert rows[4]['diagnostics'] == ['evidence_insufficient']
    assert rows[4]['metrics'] is None


def test_fixed_fields_are_extractive():
    for row in load_samples(DATA / 'samples.jsonl'):
        for field in ('objects', 'visible_text'):
            assert all(v.lower() in row['original_description'].lower() for v in row['structured'][field])


@pytest.mark.parametrize('change', ['unknown', 'duplicate', 'missing', 'status', 'all_empty'])
def test_bad_qrels(tmp_path, change):
    rows = [dict(sample_id='s', relevant_ids=['A'], evidence_status='available', rationale='test')]
    if change == 'unknown':
        rows[0]['relevant_ids'] = ['missing']
    elif change == 'duplicate':
        rows *= 2
    elif change == 'missing':
        rows = []
    elif change == 'status':
        rows[0]['evidence_status'] = 'corpus_insufficient'
    else:
        rows[0].update(relevant_ids=[], evidence_status='unjudged')
    path = tmp_path / 'qrels.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):
        load_judgments(path, {'s'}, frozenset({'A'}))


def test_cli_snapshot_replay_and_no_overwrite(tmp_path):
    output = tmp_path / 'run'
    execution = subprocess.run([sys.executable, '-m', 'visionevidence.evaluation.observation',
        '--dataset', str(DATA), '--config', str(CONFIG), '--output', str(output)],
        capture_output=True, text=True, timeout=30)
    assert execution.returncode == 0, execution.stderr
    summary = json.loads(execution.stdout)
    assert summary['Baseline']['total'] == 10
    assert summary['Structured']['evaluated'] == 9
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'completed' and manifest['vlm_calls'] == 0
    assert manifest['random_seed'] == 42 and manifest['code']['commit']
    for name, record in manifest['inputs'].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == record['sha256']
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        run(DATA, CONFIG, output)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    assert run(output, output / 'config.toml', tmp_path / 'replay') == summary
    a = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
    b = [json.loads(line) for line in (tmp_path / 'replay/results.jsonl').read_text().splitlines()]
    for rows in (a, b):
        for row in rows:
            row.pop('query_seconds')
            row.pop('retrieval_seconds')
    assert a == b


def test_failed_run_is_recorded(tmp_path):
    config = tmp_path / 'bad.toml'
    config.write_text('top_k = 3')
    with pytest.raises(ValueError):
        run(DATA, config, tmp_path / 'bad-run')
    assert json.loads((tmp_path / 'bad-run/manifest.json').read_text())['status'] == 'failed'


def test_sample_labels_rejected(tmp_path):
    row = json.loads((DATA / 'samples.jsonl').read_text(encoding='utf-8').splitlines()[0])
    row['relevant_ids'] = ['D01']
    path = tmp_path / 'samples.jsonl'
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='labels'):
        load_samples(path)

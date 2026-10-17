"""Frozen CPU representation comparison. Labels are loaded only after ranking."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import tomllib

from visionevidence.evaluation.retrieval import metrics_at_k
from visionevidence.observation import build_query, parse_observation
from visionevidence.observation_run import code_provenance, write_json
from visionevidence.retrieval import BM25Index, load_corpus
from visionevidence.retrieval.bm25 import TOKENIZER, tokenize
from visionevidence.retrieval.corpus import read_jsonl


def load_samples(path: Path) -> list[dict]:
    samples = [r for _, r in read_jsonl(path)]
    seen = set()
    for row in samples:
        if set(row) != {'sample_id', 'visual_observation', 'original_description', 'structured', 'provenance'}:
            raise ValueError('invalid sample fields; labels must be kept separately')
        for field in ('sample_id', 'visual_observation', 'original_description', 'provenance'):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f'invalid sample {field}')
        if row['sample_id'] in seen:
            raise ValueError('duplicate sample_id')
        seen.add(row['sample_id'])
    if not samples:
        raise ValueError('empty samples')
    return samples


def construct_query(description: str, structured: dict, method: str) -> tuple[str, list]:
    """No corpus, judgments, sample IDs or annotation rationales accepted."""
    if method == 'Baseline':
        terms = tokenize(description)
        if not terms:
            raise ValueError('description has no supported query terms')
        return ' '.join(terms), [{'field': 'original_description', 'terms': terms}]
    if method == 'Structured':
        return build_query(parse_observation(json.dumps(structured, ensure_ascii=False)))
    raise ValueError('unknown method')


def rank_samples(index: BM25Index, samples: list[dict]) -> list[dict]:
    rows = []
    for sample in samples:
        for method in ('Baseline', 'Structured'):
            started = perf_counter()
            row = dict(sample_id=sample['sample_id'], method=method, query=None,
                       query_sources=[], evidence=[], query_error=None, retrieval_seconds=0.0)
            try:
                row['query'], row['query_sources'] = construct_query(
                    sample['original_description'], sample['structured'], method)
            except ValueError as exc:
                row['query_error'] = str(exc)
            row['query_seconds'] = perf_counter() - started
            if row['query_error'] is None:
                started = perf_counter()
                row['evidence'] = [asdict(e) for e in index.search(row['query'], k=5)]
                row['retrieval_seconds'] = perf_counter() - started
            rows.append(row)
    return rows


def load_judgments(path: Path, sample_ids: set[str], document_ids: frozenset[str]) -> dict:
    result = {}
    for _, row in read_jsonl(path):
        if set(row) != {'sample_id', 'relevant_ids', 'evidence_status', 'rationale'}:
            raise ValueError('invalid qrels fields')
        sid, ids = row['sample_id'], row['relevant_ids']
        if not isinstance(sid, str) or sid not in sample_ids or sid in result:
            raise ValueError('unknown or duplicate qrels sample_id')
        if not isinstance(ids, list) or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError('invalid relevant_ids')
        if len(ids) != len(set(ids)) or set(ids) - document_ids:
            raise ValueError('duplicate or absent corpus IDs')
        if row['evidence_status'] not in ('available', 'corpus_insufficient', 'unjudged'):
            raise ValueError('invalid evidence_status')
        if bool(ids) != (row['evidence_status'] == 'available'):
            raise ValueError('evidence_status contradicts qrels')
        if not isinstance(row['rationale'], str) or not row['rationale'].strip():
            raise ValueError('qrels rationale required')
        result[sid] = row
    if set(result) != sample_ids:
        raise ValueError('qrels must cover exactly all samples')
    if not any(r['relevant_ids'] for r in result.values()):
        raise ValueError('no evaluable queries')
    return result


def score_rows(rows: list[dict], judgments: dict) -> dict:
    for row in rows:
        judgment = judgments[row['sample_id']]
        relevant = frozenset(judgment['relevant_ids'])
        ranked = [e['document_id'] for e in row['evidence']]
        row['judgment'] = judgment
        row['metrics'] = None
        flags = []
        if row['query_error'] is not None:
            flags.append('query_construction_error')
        elif not ranked:
            flags.append('retrieval_no_hit')
        if judgment['evidence_status'] == 'corpus_insufficient':
            flags.append('evidence_insufficient')
        if relevant:
            row['metrics'] = {f'recall_at_{k}': metrics_at_k(ranked, relevant, k)[0] for k in (1, 3, 5)}
            row['metrics']['mrr_at_5'] = metrics_at_k(ranked, relevant, 5)[1]
            if ranked and not (set(ranked) & relevant):
                flags.append('relevant_not_retrieved_at_5')
            elif 0 < row['metrics']['recall_at_5'] < 1:
                flags.append('partial_recall_at_5')
        row['diagnostics'] = flags

    def aggregate(subset):
        if not subset:
            return None
        return {key: sum(r['metrics'][key] for r in subset) / len(subset)
                for key in subset[0]['metrics']}

    summary = {}
    for method in ('Baseline', 'Structured'):
        all_rows = [r for r in rows if r['method'] == method]
        evaluated = [r for r in all_rows if r['metrics'] is not None]
        successful = [r for r in evaluated if r['query_error'] is None]
        summary[method] = dict(total=len(all_rows), evaluated=len(evaluated),
                               excluded_ids=[r['sample_id'] for r in all_rows if r['metrics'] is None],
                               end_to_end=aggregate(evaluated), successful_query_count=len(successful),
                               successful_query_subset=aggregate(successful),
                               diagnostics=dict(Counter(f for r in all_rows for f in r['diagnostics'])))
    return summary


def run(dataset: Path, config: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    manifest = dict(schema_version=1, experiment='T04-4', status='started',
                    started_utc=datetime.now(timezone.utc).isoformat(), inputs={},
                    inference_performed=False, vlm_calls=0, model=None, peak_memory_bytes=None,
                    memory_note='not measured; CPU only', data_kind='synthetic_fixture_not_vlm_output')
    try:
        manifest['code'] = code_provenance()
        manifest['environment'] = dict(python=sys.version, executable=sys.executable,
            platform=platform.platform(), processor=platform.processor(), device='cpu',
            dependencies=sorted({f"{d.metadata['Name']}=={d.version}" for d in distributions()}))
        for path in [dataset / n for n in ('samples.jsonl', 'corpus.jsonl', 'qrels.jsonl', 'dataset.json')] + [config]:
            name = 'config.toml' if path == config else path.name
            data = path.read_bytes()
            (output / name).write_bytes(data)
            manifest['inputs'][name] = dict(source=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest())
        settings = tomllib.loads((output / 'config.toml').read_text(encoding='utf-8'))
        if (set(settings) != {'schema_version', 'seed', 'top_k', 'query_policy', 'bm25'}
                or settings['schema_version'] != 1 or type(settings['seed']) is not int
                or type(settings['top_k']) is not int or settings['top_k'] != 5
                or settings['query_policy'] != 'description-v1_vs_objects-visible-text-v1'
                or set(settings['bm25']) != {'k1', 'b'}):
            raise ValueError('unsupported evaluation configuration')
        manifest.update(config=settings, random_seed=settings['seed'],
                        randomness='none; no sampling', tokenizer=TOKENIZER)
        manifest['dataset'] = json.loads((output / 'dataset.json').read_text(encoding='utf-8'))
        samples = load_samples(output / 'samples.jsonl')
        index_start = perf_counter()
        index = BM25Index(load_corpus(output / 'corpus.jsonl'), **settings['bm25'])
        manifest.update(index_build_seconds=perf_counter() - index_start,
                        documents=len(index.document_ids), samples=len(samples))
        rows = rank_samples(index, samples)
        judgments = load_judgments(output / 'qrels.jsonl', {s['sample_id'] for s in samples}, index.document_ids)
        summary = score_rows(rows, judgments)
        with (output / 'results.jsonl').open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + '\n')
        write_json(output / 'summary.json', summary)
        manifest['artifacts'] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                 for name in ('results.jsonl', 'summary.json')}
        manifest['status'] = 'completed'
        return summary
    except Exception as exc:
        manifest.update(status='failed', error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        manifest['total_seconds'] = perf_counter() - started
        write_json(output / 'manifest.json', manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.dataset, args.config, args.output), indent=2))
    except (ValueError, OSError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

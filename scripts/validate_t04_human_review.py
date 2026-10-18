"""Validate the complete human-confirmed Cartesian product against frozen qrels."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def compare(rows: list[dict], confirmation: dict, qrels: list[dict], corpus: list[dict]) -> dict:
    """Human labels come from explicit confirmation, never from the frozen qrels."""
    samples = confirmation['samples']
    ids = confirmation['document_ids']
    check(len(ids) == len(set(ids)) == 12, 'expected 12 unique documents')
    sample_ids = [s['sample_id'] for s in samples]
    check(len(sample_ids) == len(set(sample_ids)) == 10, 'expected 10 unique samples')
    check(confirmation['unlisted_documents_label'] == 'not_relevant', 'exhaustive negatives required')
    documents = {d['evidence_id']: d for d in corpus}
    check(len(corpus) == len(documents) == 12 and set(documents) == set(ids), 'corpus IDs mismatch')
    frozen = {r['sample_id']: set(r['relevant_ids']) for r in qrels}
    check(len(qrels) == len(frozen) == 10 and set(frozen) == set(sample_ids), 'qrels coverage mismatch')
    check(all(v <= set(ids) for v in frozen.values()), 'unknown frozen document')
    expected = set()
    for sample in samples:
        relevant = sample['relevant_ids']
        check(len(relevant) == len(set(relevant)) and set(relevant) <= set(ids), 'invalid confirmed IDs')
        expected.update((sample['sample_id'], doc) for doc in ids)
    pairs = [(r['sample_id'], r['document_id']) for r in rows]
    check(len(pairs) == len(set(pairs)), 'duplicate judgment')
    check(set(pairs) == expected, 'missing or unknown judgment')
    by_sample = {s['sample_id']: s for s in samples}
    counts = Counter()
    mismatches = []
    for row in rows:
        sid, doc = row['sample_id'], row['document_id']
        check(row['label'] in {'relevant', 'not_relevant', 'uncertain'}, 'invalid label')
        human_label = 'relevant' if doc in by_sample[sid]['relevant_ids'] else 'not_relevant'
        check(row['label'] == human_label, 'label differs from user confirmation')
        check(row['review_id'] == confirmation['review_id'], 'review ID mismatch')
        check(row['review_date'] == confirmation['confirmation_date'], 'review date mismatch')
        check(row['sample_note'] == by_sample[sid]['note'], 'note mismatch')
        check(row['evidence_text'] == documents[doc]['text'], 'evidence text mismatch')
        check(row['evidence_source'] == documents[doc]['source'], 'evidence source mismatch')
        frozen_label = 'relevant' if doc in frozen[sid] else 'not_relevant'
        if row['label'] != frozen_label:
            mismatches.append(dict(sample_id=sid, document_id=doc, human=row['label'], frozen=frozen_label))
        counts[row['label']] += 1
    return dict(samples=10, documents=12, judgments=len(rows), relevant=counts['relevant'],
                not_relevant=counts['not_relevant'], uncertain=counts['uncertain'],
                matches=len(rows) - len(mismatches), mismatches=mismatches,
                comparison='human confirmation versus frozen synthetic qrels; not inter-annotator agreement')


def validate(root: Path, review: Path) -> dict:
    manifest = json.loads((review / 'manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['sha256'].items():
        path = (root / relative).resolve()
        check(path.is_relative_to(root.resolve()), 'hash path outside repository')
        check(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'hash mismatch: {relative}')
    frozen = root / 'experiments/t04_v1/run-20261017-01'
    result = compare(read_jsonl(review / 'judgments.jsonl'),
                     json.loads((review / 'confirmation.json').read_text(encoding='utf-8')),
                     read_jsonl(frozen / 'qrels.jsonl'), read_jsonl(frozen / 'corpus.jsonl'))
    check(result == json.loads((review / 'statistics.json').read_text(encoding='utf-8')), 'statistics mismatch')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--review', type=Path, default=Path('experiments/t04_v1/human-review-20261018-01'))
    args = parser.parse_args()
    print(json.dumps(validate(args.root, args.root / args.review), ensure_ascii=False, indent=2))

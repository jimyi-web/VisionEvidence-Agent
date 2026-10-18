"""Read-only T04-4 replay and human-review export; never changes frozen inputs."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import tomllib

from visionevidence.evaluation.observation import (
    load_judgments, load_samples, rank_samples, score_rows,
)
from visionevidence.retrieval import BM25Index, load_corpus
from visionevidence.retrieval.bm25 import tokenize


NOTES = {
    'S01': ('对象、红色、Route 12 保留；beside 关系未保留。两组 Top-5 均召回两篇；首位互换不改变指标。', '确认 D01/D02 均相关及近重复计数是否符合既定粒度；核对 D12 的颜色/线路冲突。'),
    'S02': ('bicycle、river bridge、closed 保留；beside 关系未保留。两组首位命中；D04 含 no bicycle/open，词袋不能理解否定。', '确认 D03 相关；D04/D06 应否作为主题相关但场景不匹配的负例。'),
    'S03': ('library entrance、Open Sunday 保留；empty 状态遗漏。两组首位命中；证据 D05 本来也未描述 empty。', '确认允许证据只覆盖部分场景；核对 D05 与 Closed Sunday 的 D06。'),
    'S04': ('Platform 7 可见文字丢失；train platform/suitcase 保留。D07/D08 同分，按 ID 排序命中，不能证明识别站台号。', '重点：D07 相关、D08 不相关的编号判据；现有 rationale 描述提取失败而非充分解释相关性。'),
    'S05': ('dog、fountain、Drinking water 保留；beside 关系遗漏。D10 含 No drinking，仍有正词重合。', '核对 D09 与 D10 的饮用/禁止饮用语义；相关不等于已验证安全。'),
    'S06': ('两组查询均为 automobile；D11 只有 car，无词面交集。有效查询无候选；未发生结构化信息损失。', '重点：单词 automobile 与 car 是否足以定义同一合成场景的相关性。'),
    'S07': ('中文字段保留，但两组均无 ASCII 词项，查询构造失败；不是提取遗漏，也不是成功检索后漏召回。', '重点：中文含义与 D05 是否对应，D06 是否不相关；不能当作真实 OCR 失败。'),
    'S08': ('objects/visible_text 全空，bus stop、Route 12 全部遗漏；Structured 查询失败，Baseline 召回两篇。', '重点：描述未提及红色巴士，D01/D02 添加该属性是否仍相关；rationale 只解释查询失败。'),
    'S09': ('telescope/dome/Observatory 保留；under 关系遗漏。Baseline 与语料的唯一公共词为 a；Structured 无候选。空 qrels 排除。', '重点：遍查全部 12 篇确认 corpus_insufficient，而不是尚未标注；空候选不证明现实中无证据。'),
    'S10': ('river bridge/Bridge open 保留，无关键状态丢失；Baseline 的 with 对 D03 有贡献，Structured 首位变为 D04。', '重点：D04 额外写 no bicycle 是否允许；D03 的 closed 是否足以排除。'),
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(archive: Path, output: Path) -> dict:
    manifest = read_json(archive / 'manifest.json')
    hashes = {name: item['sha256'] for name, item in manifest['inputs'].items()}
    hashes.update(manifest['artifacts'])
    for name, expected in hashes.items():
        require(digest(archive / name) == expected, f'hash mismatch: {name}')
    samples = load_samples(archive / 'samples.jsonl')
    require([s['sample_id'] for s in samples] == list(NOTES), 'expected frozen S01-S10')
    corpus = load_corpus(archive / 'corpus.jsonl')
    settings = tomllib.loads((archive / 'config.toml').read_text(encoding='utf-8'))
    index = BM25Index(corpus, **settings['bm25'])
    replay = rank_samples(index, samples)
    judgments = load_judgments(archive / 'qrels.jsonl', set(NOTES), index.document_ids)
    summary = score_rows(replay, judgments)
    archived = [json.loads(line) for line in (archive / 'results.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(archived) == len(replay) == 20, 'expected 20 result rows')
    for old, new in zip(archived, replay, strict=True):
        for key in ('sample_id', 'method', 'query', 'query_sources', 'query_error', 'judgment', 'metrics', 'diagnostics'):
            require(old[key] == new[key], f'replay mismatch: {old["sample_id"]}/{key}')
        require(len(old['evidence']) == len(new['evidence']), 'candidate count mismatch')
        for a, b in zip(old['evidence'], new['evidence'], strict=True):
            require(all(a[k] == b[k] for k in ('document_id', 'source', 'text', 'metadata')), 'evidence mismatch')
            require(math.isclose(a['score'], b['score'], rel_tol=1e-12, abs_tol=1e-12), 'score mismatch')
    require(summary == read_json(archive / 'summary.json'), 'summary mismatch')
    # Single-term scores are additive in this implementation; no alternate query policy.
    contributions = []
    for row in archived:
        terms = sorted(set(tokenize(row['query'] or '')))
        term_scores = {term: {e.document_id: e.score for e in index.search(term, k=len(corpus))} for term in terms}
        for evidence in row['evidence']:
            parts = {t: scores[evidence['document_id']] for t, scores in term_scores.items() if evidence['document_id'] in scores}
            require(math.isclose(sum(parts.values()), evidence['score'], rel_tol=1e-12, abs_tol=1e-12), 'term sum mismatch')
            contributions.append(dict(sample_id=row['sample_id'], method=row['method'], document_id=evidence['document_id'], score=evidence['score'], terms=parts))
    lines = ['# T04-5 人工复核表', '',
             '全部为合成文字场景，无图片、真实 VLM 输出或人工完成标注。机器命中均相对冻结 qrels，不代表人工结论。', '',
             '填写方式：逐例阅读原描述和全部证据；记录相关/不相关/不确定及理由。排名可能造成锚定偏差，建议先遮住排名独立判断。', '',
             '保持原 qrels 不动；修订建议另存，批准后另建版本。未填写代表待复核，不代表认可。', '']
    by_key = {(r['sample_id'], r['method']): r for r in archived}
    vocabulary = set(t for e in corpus for t in tokenize(e.text))
    for sample in samples:
        sid = sample['sample_id']
        judgment = judgments[sid]
        lines += [f'## {sid}', '', f'原始描述：{sample["original_description"]}', '',
                  f'场景规格（非像素观察）：{sample["visual_observation"]}', '',
                  '结构化视觉观察：', '```json', json.dumps(sample['structured'], ensure_ascii=False, indent=2), '```', '',
                  f'冻结 qrels：{", ".join(judgment["relevant_ids"]) or "空"}；状态：{judgment["evidence_status"]}', '',
                  f'原标注理由：{judgment["rationale"]}', '',
                  f'已验证事实：{NOTES[sid][0]}', '', f'待人工判断：{NOTES[sid][1]}', '']
        bterms = set(tokenize(sample['original_description']))
        sterms = set(tokenize(' '.join(sample['structured']['objects'] + sample['structured']['visible_text'])))
        lines += [f'ASCII 去重词项损失：{", ".join(sorted(bterms - sterms)) or "无"}（中文丢失需结合原字段判断）。', '']
        for method in ('Baseline', 'Structured'):
            row = by_key[sid, method]
            terms = tokenize(row['query'] or '')
            ranked = [e['document_id'] for e in row['evidence']]
            relevant = set(judgment['relevant_ids'])
            lines += [f'### {method}', '', f'实际查询：`{row["query"]}`' if row['query'] is not None else '实际查询：null（构造失败）', '',
                      f'错误：{row["query_error"] or "无"}；词项数/去重数：{len(terms)}/{len(set(terms))}；语料外词项：{", ".join(sorted(set(terms) - vocabulary)) or "无"}', '',
                      '查询来源：`' + json.dumps(row['query_sources'], ensure_ascii=False) + '`', '',
                      '| 排名 | ID | 原始分数 | 冻结标签 | 证据原文 |', '| --- | --- | --- | --- | --- |']
            for rank, e in enumerate(row['evidence'], 1):
                lines.append(f'| {rank} | {e["document_id"]} | {e["score"]!r} | {"相关" if e["document_id"] in relevant else "不相关"} | {e["text"]} |')
            if not ranked:
                lines.append('| — | — | — | — | 无候选，不补齐零分文档 |')
            lines += ['', f'Top-5 命中：{", ".join(sorted(relevant & set(ranked))) or "无"}；遗漏：{", ".join(sorted(relevant - set(ranked))) or "无"}。' if relevant else '空 qrels：命中/遗漏及检索指标不适用；不计零。', '',
                      '逐例指标：`' + json.dumps(row['metrics']) + '`', '']
        lines += ['### 人工填写（覆盖全部闭集，不只核对已检出文档）', '',
                  '| 证据 | 原文 | 冻结标签 | 人工：相关/不相关/不确定 | 理由 |', '| --- | --- | --- | --- | --- |']
        for e in corpus:
            lines.append(f'| {e.document_id} | {e.text} | {"相关" if e.document_id in judgment["relevant_ids"] else "不相关"} | 待填写 | |')
        lines += ['', '- [ ] 观察中的对象、文字、动作/关系、时间、地点已核对',
                  '- [ ] qrels 正例与全部负例已核对', '- [ ] 失败解释已复核（同意/修订/不确定）：____',
                  '- 审核人 / 日期：____', '- 建议相关 ID（不自动写回）：____', '- 证据充分性与备注：____', '']
    lines += ['## 证据来源', '', '所有文档均为虚构合成证据，human_reviewed=false。', '']
    lines += [f'- {e.document_id}: `{e.source}`' for e in corpus]
    validation = dict(archive=str(archive.as_posix()), input_and_artifact_sha256=hashes,
                      manifest_sha256=digest(archive / 'manifest.json'), samples=10, result_rows=20,
                      replay_matches=True, term_sums_match=True, score_tolerance=1e-12,
                      excluded_ids=['S09'], summary=summary, term_contributions=contributions)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'human_review.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (output / 'audit.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return validation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('experiments/t04_v1/run-20261017-01'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.archive, args.output)
    print('PASS: hashes, 20 replay rows, summary, additive term scores; review exported')

"""Offline independent-test audit. Reads saved responses; never creates a VLM.

Keep original transport and extracted files unchanged. Output must be new.
This script extends the frozen Hit@10 bootstrap to other requested metrics;
it does not change the experiment or select another strategy.
"""

import argparse
from collections import Counter
from contextlib import closing
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from visionevidence.description import PROMPT as BASELINE_PROMPT
from visionevidence.observation import PROMPT, build_query, parse_observation
from visionevidence.observation_policy import ObservationPolicy
from visionevidence.retrieval import BM25Index, load_corpus

COMMIT = "8428c626a7ba55bb723a0e1506889a04595c1307"
FREEZE = "experiments/v11/frozen-format-details-20261020-01"
RUN = "outputs/flickr8k-v11-independent100-01"
METRICS = ("Hit@1", "Hit@5", "Hit@10", "Recall@10", "MRR@10")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(path.read_bytes())


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]


def inventory(path):
    return {p.relative_to(path).as_posix(): sha(p) for p in sorted(path.rglob("*")) if p.is_file()}


def score(ranked, relevant, status):
    """Independent metric implementation; failure dominates any cached ranking."""
    require(len(relevant) == 5 and len(set(relevant)) == 5, "expected five distinct qrels")
    require(len(ranked) == len(set(ranked)), "duplicate ranked document")
    require(status in ("completed", "failed"), "nonterminal query")
    ranks = [i for i, doc in enumerate(ranked[:10], 1) if doc in relevant] if status == "completed" else []
    return {"Hit@1": float(any(i <= 1 for i in ranks)),
            "Hit@5": float(any(i <= 5 for i in ranks)), "Hit@10": float(bool(ranks)),
            "Recall@10": len(ranks) / 5, "MRR@10": 1 / ranks[0] if ranks else 0.0}


def bootstrap(pairs, repetitions=20000, seed=0):
    require(bool(pairs) and repetitions >= 100, "empty pairs or too few replicates")
    differences = {m: [p["delta"][m] for p in pairs] for m in METRICS}
    draws = {m: [] for m in METRICS}
    rng = random.Random(seed)
    for _ in range(repetitions):
        indices = rng.choices(range(len(pairs)), k=len(pairs))
        for m in METRICS:
            draws[m].append(sum(differences[m][i] for i in indices) / len(pairs))
    output = {}
    for m in METRICS:
        values = sorted(draws[m])
        def quantile(p):
            position = (repetitions - 1) * p
            lo = int(position)
            return values[lo] + (values[min(lo + 1, repetitions - 1)] - values[lo]) * (position - lo)
        d = differences[m]
        output[m] = {"difference": statistics.mean(d), "lower": quantile(.025), "upper": quantile(.975),
                     "wins": sum(x > 0 for x in d), "ties": sum(x == 0 for x in d), "losses": sum(x < 0 for x in d)}
    return output


def contract_violations(observation):
    """Diagnostic enumeration, including violations hidden by fail-fast validation."""
    violations, words = [], 0
    for field, maximum in (("objects", 6), ("visible_text", 2), ("uncertainties", 2)):
        values = getattr(observation, field)
        if len(values) > maximum:
            violations.append(f"{field}: {len(values)} > {maximum}")
        normalized = [" ".join(x.casefold().split()) for x in values]
        if len(normalized) != len(set(normalized)):
            violations.append(f"{field}: duplicate entries")
        if any(len(x) > 100 for x in values):
            violations.append(f"{field}: string exceeds 100 characters")
        words += sum(len(x.split()) for x in values)
    if words > 60:
        violations.append(f"total words: {words} > 60")
    return violations


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def audit(source, transfer, image_archive):
    before, transfer_before = inventory(source), inventory(transfer)
    for line in (transfer / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        require((transfer / name).resolve().is_relative_to(transfer.resolve()), "unsafe checksum path")
        require(sha(transfer / name) == expected, f"transport checksum: {name}")
    require((transfer / "git-head.txt").read_text().strip() == COMMIT, "transfer commit mismatch")
    require(not (transfer / "git-status.txt").read_text().strip(), "dirty transfer Git")
    require(not (transfer / "tracked-working-tree.diff").read_bytes(), "transfer tracked diff")
    archive_files, omitted_links = {}, []
    with tarfile.open(transfer / "raw-artifacts.tar.gz") as archive:
        for member in archive.getmembers():
            if member.isfile():
                require(member.name not in archive_files, "duplicate archive member")
                archive_files[member.name] = digest(archive.extractfile(member).read())
            elif member.issym():
                omitted_links.append({"path": member.name, "target": member.linkname})
            else:
                require(member.isdir(), "unsupported archive member")
    require(archive_files == before, "extracted files differ from archive")
    frozen = read(source / FREEZE / "freeze-manifest.json")
    require(frozen["status"] == "frozen" and frozen["selected_strategy"] == "format_details", "wrong freeze")
    git = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "show"]
    for p in (source / FREEZE).iterdir():
        blob = subprocess.check_output([*git, f"{COMMIT}:{FREEZE}/{p.name}"], cwd=ROOT)
        require(blob.replace(b"\r\n", b"\n") == p.read_bytes().replace(b"\r\n", b"\n"), f"freeze drift: {p.name}")
    for name, expected in frozen["code_and_config_sha256"].items():
        blob = subprocess.check_output([*git, f"{COMMIT}:{name}"], cwd=ROOT)
        require(digest(blob.replace(b"\r\n", b"\n")) == expected, f"frozen Git blob: {name}")
        require(digest((ROOT / name).read_bytes().replace(b"\r\n", b"\n")) == expected, f"local frozen code: {name}")
    cohort = source / frozen["dataset"]["cohort"]
    require(sha(cohort / "manifest.json") == frozen["dataset"]["adapter_manifest_sha256"], "adapter hash")
    for label, expected in frozen["dataset"]["output_sha256"].items():
        require(sha(cohort / "test" / f"{label}.jsonl") == expected, f"cohort {label} hash")
    expected_ids = frozen["dataset"]["sample_ids"]
    require(len(expected_ids) == len(set(expected_ids)) == 100, "frozen ID count")
    require(not set(expected_ids) & set(frozen["dataset"]["excluded_diagnostic_sample_ids"]), "diagnostic overlap")
    image_archive_hash = sha(image_archive)
    with tarfile.open(image_archive) as archive:
        members = archive.getmembers()
        require(len({m.name for m in members}) == len(members), "duplicate image archive member")
        require(all(m.isfile() or m.isdir() for m in members), "image archive links")
        raw = {m.name.removeprefix("flickr8k-v11-test-01/"): archive.extractfile(m).read()
               for m in members if m.isfile()}
    for name, expected in frozen["dataset"]["metadata_sha256"].items():
        require(digest(raw[name]) == expected, f"raw metadata hash: {name}")
    raw_source = json.loads(raw["source.json"])
    require(raw_source["repository"] == frozen["dataset"]["repository"] and raw_source["revision"] == frozen["dataset"]["revision"], "raw source revision")
    rows = sorted(csv.DictReader(io.StringIO(raw["test/metadata.csv"].decode("utf-8"))), key=lambda r: r["file_name"])
    require(["flickr8k:" + r["file_name"] for r in rows[20:120]] == expected_ids, "test offset/order")
    image_hashes = {sid: digest(raw["test/" + sid.removeprefix("flickr8k:")]) for sid in expected_ids}
    image_inventory = json.loads(raw["image-inventory.json"])
    for sid, hashed in image_hashes.items():
        require(image_inventory["test/" + sid.removeprefix("flickr8k:")]["sha256"] == hashed, "image inventory hash")
    require(len(set(image_hashes.values())) == 100, "duplicate image content")
    expected_corpus = {f"flickr8k:{r['file_name']}#{i}": r[f"caption_{i}"] for r in rows for i in range(5)}
    run = source / RUN
    require(not list(run.rglob("*-wal")) and not list(run.rglob("*-journal")), "SQLite sidecar needs recovery on a separate copy")
    experiment = read(run / "experiment.json")
    policy = ObservationPolicy(True, True)
    require(experiment["observation_policy"] == policy.identity() and experiment["split"] == "test", "experiment policy/split")
    require(experiment["adapter_manifest_sha256"] == frozen["dataset"]["adapter_manifest_sha256"], "experiment adapter")
    require(experiment["cutoffs"] == [1,5,10], "experiment cutoffs")
    require(sorted(p.name for p in run.glob("comparison-*.json")) == ["comparison-0001.json","comparison-0002.json"], "unexpected comparison snapshots")
    branches, identities, failures = {}, {}, []
    per_arm, sessions_out, snapshot_scores = {}, {}, {}
    for arm in ("baseline", "structured"):
        path = run / arm
        for prefix, suffix in (("attempts","jsonl"),("results","jsonl"),("summary","json"),("runtime","json")):
            require(sorted(p.name for p in path.glob(f"{prefix}-*.{suffix}")) == [f"{prefix}-{n:04d}.{suffix}" for n in (1,2)], "missing/extra snapshots")
        manifest = read(path / "manifest.json")
        identity = manifest["identity"]
        identities[arm] = identity
        require(manifest["git"]["commit"] == COMMIT and manifest["git"]["dirty"] is False, f"{arm} git")
        require(manifest["environment"] == frozen["measured_dev_environment"], f"{arm} environment drift")
        require(identity["vlm_config"] == frozen["vlm_config"], f"{arm} VLM config")
        require(identity["model_file_sha256"] == frozen["model_file_sha256"], f"{arm} recorded model hashes")
        require(identity["retrieval"] == {"top_k": 10, "k1": 1.5, "b": .75} and identity["seed"] == 0, "retrieval/seed")
        require(identity["task"] == "image_to_caption_retrieval_proxy" and identity["query_mode"] == arm, "task/query mode")
        expected_prompt = BASELINE_PROMPT if arm == "baseline" else policy.prompt(PROMPT)
        require(identity["effective_prompt"] == expected_prompt, f"{arm} effective prompt")
        prompt_label = "baseline" if arm == "baseline" else "format_details"
        require(digest(expected_prompt.encode()) == frozen["prompt_sha256"][prompt_label], "prompt digest")
        require(identity["observation_policy"] == (None if arm == "baseline" else policy.identity()), "observation policy")
        for name, expected in identity["source_sha256"].items():
            require(expected == frozen["code_and_config_sha256"]["src/" + name], f"runtime source hash: {name}")
        require(identity["source_sha256"] == manifest["git"]["source_sha256"], "git source hashes")
        for label in ("samples", "corpus", "qrels"):
            require(sha(path / f"input-{label}.jsonl") == identity[f"{label}_sha256"] == frozen["dataset"]["output_sha256"][label], f"{arm}/{label} snapshot")
        sample_inputs = lines(path / "input-samples.jsonl")
        qrel_inputs = lines(path / "input-qrels.jsonl")
        with closing(sqlite3.connect((path / "state.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            require([r[0] for r in db.execute("PRAGMA integrity_check")] == ["ok"], "SQLite integrity")
            samples = [dict(r) for r in db.execute("SELECT * FROM samples ORDER BY position")]
            attempts = [dict(r) for r in db.execute("SELECT * FROM attempts ORDER BY id")]
            sessions = [dict(r) for r in db.execute("SELECT * FROM sessions ORDER BY id")]
            qrels_rows = [dict(r) for r in db.execute("SELECT * FROM qrels ORDER BY sample_id")]
        for a in attempts:
            a["raw_response"] = json.loads(a["raw_response"]) if a["raw_response"] else None
        require([s["id"] for s in samples] == expected_ids and [s["position"] for s in samples] == list(range(1, 101)), "query slots/order")
        require([json.loads(s["payload"]) for s in samples] == sample_inputs, "sample payload export")
        require(all(s["split"] == "test" and s["dataset_version"] == frozen["dataset"]["revision"] for s in sample_inputs), "sample split/version")
        require(len(attempts) == 100 and Counter(a["sample_id"] for a in attempts) == Counter(expected_ids), "attempt count or duplicate")
        require(all(s["attempts"] == 1 for s in samples), "sample retry")
        qrels = {r["sample_id"]: json.loads(r["relevant_ids"]) for r in qrels_rows}
        require(len(qrels_rows) == 100 and set(qrels) == set(expected_ids), "DB qrels IDs")
        require(sorted([{**r, "relevant_ids": json.loads(r["relevant_ids"])} for r in qrels_rows], key=lambda x:x["sample_id"]) == sorted(qrel_inputs, key=lambda x:x["sample_id"]), "qrels DB/export")
        documents = load_corpus(path / "input-corpus.jsonl")
        require(len(documents) == 5000 and {d.document_id:d.text for d in documents} == expected_corpus, "candidate corpus vs raw captions")
        index = BM25Index(documents, k1=1.5, b=.75)
        by_id = {a["sample_id"]: a for a in attempts}
        per_query, raw_responses, costs = [], [], []
        schema = contract = replays = 0
        for sample in samples:
            sid = sample["id"]
            a = by_id[sid]
            response = a["raw_response"]
            require(a["backend_called"] == a["real_response"] == 1 and response is not None, "missing real response")
            meta = response["metadata"]
            require(type(meta["generated_tokens"]) is int and meta["generated_tokens"] > 0, "invalid generated token count")
            require(response["backend"] == "qwen3_vl" and response["is_mock"] is False and meta["inference_performed"] is True, "mock/non-inference record")
            require(response["request_id"] == sid and response["model_name"] == frozen["vlm_config"]["model_name"], "response identity")
            for field in ("max_new_tokens", "do_sample", "dtype", "device", "max_image_pixels", "max_image_side", "model_path"):
                require(meta[field] == frozen["vlm_config"][field], f"response config {field}")
            require(a["image_sha256"] == sample["image_sha256"] == image_hashes[sid], "actual JPEG hash")
            require(set(qrels[sid]) == {f"{sid}#{i}" for i in range(5)}, "same-image qrels")
            require(a["status"] == sample["status"] and a["finished"] is not None, "attempt terminal state")
            error, category, violations, query = None, "accepted", [], response["description"]
            if arm == "structured":
                try:
                    observation = parse_observation(query)
                    schema += 1
                    violations = contract_violations(observation)
                    policy.validate(observation)
                    contract += 1
                    query = build_query(observation)[0]
                except ValueError as exc:
                    error = str(exc)
                    category = "format_contract" if error.startswith("v11 format") else "schema_or_json_or_query"
            elif not query.strip() or len(query) > 8000:
                error, category = "baseline requires nonempty description of at most 8000 characters", "query"
            if meta["generated_tokens"] >= 384:
                error, category = ("token limit reached; baseline description rejected" if arm == "baseline" else "token limit reached; structured observation rejected"), "token_limit"
            result = json.loads(sample["result"]) if sample["result"] else None
            require((error is None) == (sample["status"] == "completed"), "failure acceptance mismatch")
            if result is not None:
                require(sample["status"] == "completed" and a["error"] is None and a["error_type"] is None, "completed/error state")
                require(result["request_id"] == sid and result["query"] == query and result["vlm_response"] == response, "saved query/response")
                require(result["input"] == json.loads(sample["payload"]), "result input drift")
                require(result["trace"]["image_sha256"] == image_hashes[sid] and result["trace"]["prompt_sha256"] == frozen["prompt_sha256"][prompt_label], "result trace hash")
                require(result["trace"]["k"] == 10 and result["trace"]["bm25"] == {"k1":1.5,"b":.75,"tokenizer":"ascii-alphanumeric-lower-v1"}, "result retrieval trace")
                replay = index.search(query, k=10)
                require([d.document_id for d in replay] == [d["document_id"] for d in result["evidence"]], "CPU BM25 ranking mismatch")
                require(all(close(x.score, y["score"]) for x,y in zip(replay,result["evidence"])), "CPU BM25 score mismatch")
                require(all({k:v for k,v in asdict(x).items() if k!='score'} == {k:v for k,v in y.items() if k!='score'} for x,y in zip(replay,result["evidence"])), "evidence content/source mismatch")
                ranked = [d["document_id"] for d in result["evidence"]]
                replays += 1
            else:
                require(sample["status"] == "failed" and a["error_type"] == "ValueError" and a["error"] == error, "failure error mismatch")
                ranked = []
                failures.append({"arm":arm,"sample_id":sid,"attempt_id":a["id"],"error_type":a["error_type"],
                                 "error":a["error"],"category":category,"all_contract_violations":violations,
                                 "raw_response":response,"raw_description_sha256":digest(response["description"].encode())})
            per_query.append({"sample_id":sid,"status":sample["status"],"ranked_document_ids":ranked,
                              "relevant_ids":qrels[sid],"metrics":score(ranked,qrels[sid],sample["status"])})
            raw_responses.append(response)
            costs.append((datetime.fromisoformat(a["finished"])-datetime.fromisoformat(a["started"])).total_seconds()*1000)
        require([s["id"] for s in sessions] == [1,2], "unexpected session IDs")
        sessions_out[arm] = []
        snapshot_scores[arm] = {}
        for session in sessions:
            settings = json.loads(session["settings"])
            n, number = settings["limit"], session["id"]
            require(n == (2 if number == 1 else 100) and settings["max_attempts"] == 1 and settings["batch_size"] == 4, "session budget/limit")
            require(settings["git"] == manifest["git"], "session git drift")
            require(session["finished"] is not None, "unfinished session")
            prefix_attempts = [a for a in attempts if a["sample_id"] in expected_ids[:n]]
            require(lines(path/f"attempts-{number:04d}.jsonl") == prefix_attempts, "attempt snapshot mismatch")
            require(lines(path/f"results-{number:04d}.jsonl") == [json.loads(s["result"]) for s in samples[:n] if s["result"]], "results snapshot mismatch")
            summary = read(path/f"summary-{number:04d}.json")
            require(summary == json.loads(session["summary"]), "summary DB/export")
            require(summary["attempts"] == n and summary["requested_limit"] == n and summary["is_mock"] is False, "summary count/provenance")
            require(summary["sample_status"] == dict(Counter(s["status"] for s in samples[:n])), "summary statuses")
            require(summary["error_counts"] == dict(Counter(a["error_type"] for a in prefix_attempts if a["error_type"])), "summary errors")
            require(summary["status"] == ("completed" if all(s["status"]=='completed' for s in samples[:n]) else "incomplete"), "summary completion status")
            require(summary["qrels_counts"] == {"empty_excluded":0,"evaluated":n,"unannotated":0} and summary["retrieval_metrics"]["evaluated_queries"] == n, "summary metric denominator")
            require(summary["scale"]["real_vlm_image_calls"] == summary["scale"]["real_vlm_responses"] == n, "summary calls")
            metrics = {m:statistics.mean(q["metrics"][m] for q in per_query[:n]) for m in METRICS}
            require(close(metrics["Recall@10"],summary["retrieval_metrics"]["recall_at_k"]) and close(metrics["MRR@10"],summary["retrieval_metrics"]["mrr_at_k"]), "summary metrics")
            runtime = read(path/f"runtime-{number:04d}.json")
            require(runtime == frozen["measured_dev_runtime"], "runtime drift")
            new = [a for a in attempts if session["started"] <= a["started"] <= a["finished"] <= session["finished"]]
            require(len(new) == (2 if number == 1 else 98), "session attempt timing/count")
            sessions_out[arm].append({"id":number,"limit":n,"new_calls":len(new),"started":session["started"],"finished":session["finished"],"status":summary["status"]})
            snapshot_scores[arm][number] = metrics
        values = lambda key: [r["metadata"][key] for r in raw_responses]
        branches[arm] = {"metrics":snapshot_scores[arm][2],"completed":replays,"failed":100-replays,"real_calls":100,
                         "original_schema_successes":schema if arm=="structured" else None,
                         "format_contract_successes":contract if arm=="structured" else None,
                         "generated_tokens_sum":sum(values("generated_tokens")),
                         "response_latency_ms_mean_all":statistics.mean(r["latency_ms"] for r in raw_responses),
                         "attempt_wall_ms_mean_all":statistics.mean(costs),
                         "retrieval_ms_mean_successful":statistics.mean(json.loads(s["result"])["trace"]["retrieval_ms"] for s in samples if s["result"]),
                         "cold_model_load_ms":[r["metadata"]["model_load_ms"] for r in raw_responses if r["metadata"]["model_loaded_this_call"]],
                         "peak_torch_allocated_MiB":max(values("peak_gpu_memory_mb")),"runtime":runtime,
                         "environment":manifest["environment"]}
        per_arm[arm] = per_query
    for field in ("samples_sha256","corpus_sha256","qrels_sha256","vlm_config","model_file_sha256","source_sha256","seed","retrieval"):
        require(identities["baseline"][field] == identities["structured"][field], f"cross-arm control {field}")
    for number, n in ((1,2),(2,100)):
        comparison = read(run/f"comparison-{number:04d}.json")
        require(comparison["scale"] == {"candidate_caption_documents":5000,"evaluation_images":n,"retrieval_query_slots":2*n,"real_vlm_image_calls":2*n,"unique_real_vlm_images":n}, "comparison scale")
        require(comparison["is_mock"] is False and comparison["status"] == "incomplete", "comparison status")
        for arm in branches:
            require(comparison["branches"][arm] == read(run/arm/f"summary-{number:04d}.json"), "comparison summary")
            scores = comparison["scores"][arm]
            require(scores["denominator"] == n and len(scores["per_query"]) == n, "comparison denominator")
            for k in (1,5,10):
                totals = {"hit_rate":[],"caption_recall":[],"mrr":[]}
                for computed, recorded in zip(per_arm[arm][:n],scores["per_query"]):
                    require(computed["sample_id"] == recorded["sample_id"] and computed["status"] == recorded["status"], "comparison ID/status")
                    ranks = [i for i,d in enumerate(computed["ranked_document_ids"][:k],1) if d in computed["relevant_ids"]]
                    values = {"hit_rate":float(bool(ranks)),"caption_recall":len(ranks)/5,"mrr":1/ranks[0] if ranks else 0}
                    for m,v in values.items():
                        require(close(v,recorded["metrics"][str(k)][m]), "comparison per-query metric")
                        totals[m].append(v)
                require(all(close(statistics.mean(v),scores["metrics"][str(k)][m]) for m,v in totals.items()), "comparison aggregate")
    paired = [{"sample_id":b["sample_id"],"baseline_status":b["status"],"format_details_status":s["status"],
               "baseline":b["metrics"],"format_details":s["metrics"],"delta":{m:s["metrics"][m]-b["metrics"][m] for m in METRICS}}
              for b,s in zip(per_arm["baseline"],per_arm["structured"])]
    intervals = bootstrap(paired)
    require(inventory(source) == before and inventory(transfer) == transfer_before and sha(image_archive) == image_archive_hash, "audit modified input")
    return {"status":"audited_with_provenance_limits","task":"image_to_caption_retrieval_proxy","is_fact_verification":False,
            "commit":COMMIT,"gpu_calls_during_audit":0,"branches":branches,"failures":failures,"paired":paired,
            "per_arm":per_arm,"ci95":intervals,"bootstrap":{**frozen["hit10_ci95"],"generator":"Python random.Random(0).choices",
            "shared_resamples_across_metrics":True,"primary_preregistered":"Hit@10","additional_requested_descriptive_intervals":[m for m in METRICS if m!='Hit@10'],"multiplicity_adjusted":False},
            "sessions":sessions_out,"transport_sha256":sha(transfer/"raw-artifacts.tar.gz"),
            "image_archive_sha256":image_archive_hash,"source_file_sha256":before,"transfer_file_sha256":transfer_before,
            "image_sha256":image_hashes,"omitted_external_symlinks":omitted_links,
            "console_logs_included":bool((transfer/"included-logs.txt").read_text().strip()),
            "limitations":["Transport raw directory is only an external symlink; images and metadata verified against separate existing image archive.",
                           "No console logs supplied; exact shell invocation, wrapper preflight and absence of out-of-directory runs are not independently proven.",
                           "200 internally consistent recorded real-backend calls; not independent hardware attestation. Model hashes are recorded, weights not rehashed offline.",
                           "Fixed ASCII-order cohort, correlated captions, assumed negatives, possible pretraining exposure; no population-wide or factual-verification claim."]}


def report_markdown(result):
    """Private full report, including unchanged raw failure descriptions."""
    text = ["# V1.1 独立 test 100 图离线审计", "", "审计日期：2026-10-27。任务：image-to-caption retrieval proxy；不是事实核验。",
            "", f"冻结 Commit：`{COMMIT}`。审计状态：原始记录一致，存在备份来源限制。新增 VLM/GPU 调用：0。",
            "", "## 结果与统计方法", "", "两路各 100 个查询槽位；失败计零，全部指标分母 100。主要预注册指标 Hit@10 下降，未获得独立测试提升证据。区间包含零不等于证明等效。",
            "", "| 指标 | Baseline | format_details | 配对差 | 95% CI | 胜/平/负 |", "| --- | ---: | ---: | ---: | --- | --- |"]
    for m, ci in result["ci95"].items():
        b=result["branches"]["baseline"]["metrics"][m]; s=result["branches"]["structured"]["metrics"][m]
        text.append(f"| {m} | {b:.10f} | {s:.10f} | {ci['difference']:+.10f} | [{ci['lower']:.10f}, {ci['upper']:.10f}] | {ci['wins']}/{ci['ties']}/{ci['losses']} |")
    text += ["", "Hit@K 是前 K 至少命中一个冻结同图 caption；Recall@10 是前十命中的五个相关 caption 比例；MRR@10 是首个相关排名倒数。失败和未命中为零。",
             "", "逐图片差 = format_details − Baseline。以冻结顺序排列 100 对图片，Python random.Random(0).choices 每轮有放回抽 100 对，共 20,000 轮；所有指标共享抽样索引。取均值差分布的 2.5% 和 97.5% 线性插值分位点。Hit@10 区间为预注册主分析；其余按用户要求补充，属描述性区间，未作多重比较调整。不能将 500 个相关 caption 当成独立样本，也不能将固定顺序队列区间解释为总体泛化保证。",
             "", "## 完整性及协议", "", "- 两路 SQLite integrity_check=ok，100 个唯一 sample_id 及其顺序完全匹配冻结 Manifest，排除此前诊断 20 图。候选文档 5,000；每图五个 qrels。",
             "- 每路 100 attempts、100 返回响应，所有样本 attempts=1；共 200 条真实 backend 标记响应、100 个唯一图片哈希。Smoke 每路新增 2 次，完整会话每路新增 98 次。累计快照不相加。",
             "- 全部 0001/0002 attempts、成功 results、summary、comparison 与 SQLite 相符。Baseline 完成 100；format_details 完成 94、失败 6，保留 incomplete。100 个槽位均终结，incomplete 表示存在失败，并非漏跑六图。",
             "- 194 个成功查询从保存响应重建查询，再用 CPU BM25 重放，排名、分数、证据文本及来源一致。失败未构造替代排名。",
             "- 47 个解包普通文件逐字节匹配传输包；原件审计前后哈希未变。冻结文件与 Commit 比对，源码/配置按冻结 CRLF→LF 约定检查。输入、qrels、模型记录哈希、Prompt、生成及检索参数均一致。",
             "- 100 张图片字节哈希与两路 attempts/samples、单独图片包 inventory 一致；镜像元数据哈希匹配冻结记录，样本为原 test ASCII 排序第 21–120 张。",
             "- 未发现所提供记录内部的测试协议偏差、重复调用、缺失响应或哈希冲突。六次合约失败是被测方法的不利结果，不是协议修改。",
             "", "## 失败记录", "", "100/100 structured 响应通过严格 JSON 与原 Schema；94/100 通过附加格式合约。六条均 ValueError、无 token-limit 截断；主错误 visible_text 超限四条、objects 超限两条。首图还同时 uncertainties 超限；以下列出全部可机械核验的违规，避免 fail-fast 掩盖第二项。原始描述保留原有换行。完整响应元数据在 failures.json。"]
    for f in result["failures"]:
        text += ["", f"### {f['sample_id']}", "", f"`{f['error_type']}: {f['error']}`", "",
                 f"分类：{f['category']}；generated_tokens={f['raw_response']['metadata']['generated_tokens']}；违规：{'；'.join(f['all_contract_violations'])}。",
                 "", "```json", f["raw_response"]["description"], "```"]
    text += ["", "## 成本记录", "", "| 指标 | Baseline | format_details |", "| --- | ---: | ---: |"]
    for label,key in (("总 generated_tokens","generated_tokens_sum"),("响应平均延迟 ms（各100，含失败）","response_latency_ms_mean_all"),("attempt 平均 wall ms（各100）","attempt_wall_ms_mean_all"),("成功检索平均 ms（100/94）","retrieval_ms_mean_successful"),("最大 torch allocated MiB","peak_torch_allocated_MiB")):
        text.append(f"| {label} | {result['branches']['baseline'][key]:.4f} | {result['branches']['structured'][key]:.4f} |")
    text += ["", "Baseline 含两次冷加载，format_details 复用模型；固定顺序先 Baseline 后 Structured，未随机交叉。响应 latency 排除加载；torch allocated MiB 排除加载峰值，也不等于设备总显存。因此不据此声称普遍推理加速或硬件容量保证。",
             "", "## 来源限制", "", "传输包 raw 目录是指向 AutoDL 的外部软链接，未解包该链接；用本地已有 flickr8k-v11-test-100.tar.gz 补足图片与原始元数据核验，未下载或重选样本。included-logs.txt 为空，没有终端日志；无法独立验证实际 shell 命令、wrapper preflight 输出或目录外运行。SQLite sessions 和逐条响应证据仍完整。模型权重未传输，校验的是冻结与运行记录中的 16 个模型文件哈希，未在本地重哈希权重。真实调用结论限于保存记录与代码的一致性，不是硬件独立认证。",
             "", "其他图 caption 属假定负例；语义相近 caption 可能被误计负例。每图五个描述相关；模型预训练暴露未知；队列是固定排序窗口。统计区间不消除上述限制，亦不证明真实事实核验能力。",
             "", "format_details 保持冻结，不因该结果重选策略、修复失败后替换结果或修改协议。后续若研究新方法，应登记新版本并使用另外的未用评测集。",
             "", "## 原件标识与复现", "", f"结果包 SHA256：`{result['transport_sha256']}`", "", f"图片包 SHA256：`{result['image_archive_sha256']}`",
             "", "详见 audit.json（逐文件哈希、运行环境、会话与完整排名）、paired.csv / paired.json（100 对指标及差值）、failures.json（六条原响应）。脚本命令及验证记录见仓库 docs/v11_independent100_audit.md。"]
    return "\n".join(text)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists")
    require(not args.output.resolve().is_relative_to(args.source.resolve()) and not args.output.resolve().is_relative_to(args.transfer.resolve()), "output within source")
    result = audit(args.source,args.transfer,args.images)
    result["audit_script_sha256"] = sha(Path(__file__))
    result["audit_python"] = sys.version
    args.output.mkdir(parents=True,exist_ok=False)
    for name, value in (("audit.json",result),("failures.json",result["failures"]),("paired.json",result["paired"])):
        with (args.output/name).open("x",encoding="utf-8") as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write("\n")
    with (args.output/"report.md").open("x",encoding="utf-8") as stream:
        stream.write(report_markdown(result))
    with (args.output/"paired.csv").open("x",encoding="utf-8",newline="") as stream:
        fieldnames = ["sample_id","baseline_status","format_details_status"] + [f"{arm}_{m}" for arm in ("baseline","format_details","delta") for m in METRICS]
        writer=csv.DictWriter(stream,fieldnames=fieldnames);writer.writeheader()
        for row in result["paired"]:
            writer.writerow({**{k:row[k] for k in fieldnames[:3]},**{f"{arm}_{m}":row[arm][m] for arm in ("baseline","format_details","delta") for m in METRICS}})
    print(json.dumps({"status":result["status"],"metrics":{k:v["metrics"] for k,v in result["branches"].items()},"ci95":result["ci95"],"failures":len(result["failures"])}))


if __name__ == "__main__":
    main()

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORPUS = str(ROOT / "examples/evidence_demo.jsonl")
QRELS = str(ROOT / "examples/qrels_demo.jsonl")
CONFIG = str(ROOT / "configs/retrieval_bm25.toml")


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "visionevidence.cli", *args],
                          capture_output=True, text=True, timeout=15, check=False)


def test_retrieve_cli() -> None:
    result = run_cli("retrieve", "--query", "flood road closures", "--corpus", CORPUS, "--top-k", "3")
    assert result.returncode == 0, result.stderr
    for value in ("E001", "E002", "E004", "synthetic/demo", "score", "Heavy rainfall"):
        assert value in result.stdout
    assert "predicted_label" not in result.stdout


def test_retrieve_json_config_and_metadata() -> None:
    result = run_cli("retrieve", "--query", "FLOOD road closures!", "--corpus", CORPUS,
                     "--config", CONFIG, "--top-k", "2", "--json")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["config"]["k1"] == 1.5
    assert [row["evidence_id"] for row in data["results"]] == ["E001", "E002"]
    assert all(row["score"] > 0 for row in data["results"])
    assert data["results"][0]["metadata"]["synthetic"] is True
    assert data["results"][0]["metadata"]["note"] == "虚构演示文本，不是真实新闻"


def test_cli_no_matches() -> None:
    result = run_cli("retrieve", "--query", "zzzzzz", "--corpus", CORPUS, "--json")
    assert result.returncode == 0
    assert json.loads(result.stdout)["results"] == []
    result = run_cli("retrieve", "--query", "zzzzzz", "--corpus", CORPUS)
    assert result.returncode == 0
    assert result.stdout.strip() == "No matching evidence."


@pytest.mark.parametrize("args, message", [
    (["retrieve", "--query", "", "--corpus", CORPUS], "English letter"),
    (["retrieve", "--query", "rain", "--corpus", CORPUS, "--top-k", "0"], "positive integer"),
    (["retrieve", "--query", "rain", "--corpus", CORPUS, "--top-k", "-1"], "positive integer"),
    (["retrieve", "--query", "rain", "--corpus", CORPUS, "--top-k", "abc"], "invalid int"),
    (["retrieve", "--query", "rain", "--corpus", "missing-corpus.jsonl"], "No such file"),
    (["--text", "rain", "retrieve", "--query", "rain", "--corpus", CORPUS], "cannot be combined"),
])
def test_cli_errors(args: list[str], message: str) -> None:
    result = run_cli(*args)
    assert result.returncode == 2
    assert message in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("content", ["[broken", "unexpected = 1", "[bm25]\nk1=1.5", "[bm25]\nk1=1.5\nb=-1"])
def test_invalid_config(tmp_path: Path, content: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    result = run_cli("retrieve", "--query", "rain", "--corpus", CORPUS, "--config", str(path))
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_evaluation_json_and_no_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "evaluation.json"
    args = ("evaluate", "--corpus", CORPUS, "--qrels", QRELS, "--config", CONFIG,
            "--top-k", "3", "--dataset-name", "synthetic/demo", "--dataset-version", "1",
            "--split", "demo", "--output", str(output))
    result = run_cli(*args)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data == json.loads(output.read_text(encoding="utf-8"))
    assert data["metrics"] == {"recall_at_k": 0.75, "mrr_at_k": 0.75}
    assert data["counts"]["queries_evaluated"] == 4
    assert data["counts"]["queries_skipped"] == 1
    assert len(data["inputs"]["corpus"]["sha256"]) == 64
    assert data["dataset"] == {"name": "synthetic/demo", "version": "1", "split": "demo"}
    assert data["environment"]["device"] == "cpu"
    assert data["timing"]["index_build_seconds"] >= 0
    original = output.read_bytes()
    again = run_cli(*args)
    assert again.returncode == 2
    assert "File exists" in again.stderr
    assert output.read_bytes() == original


def test_evaluate_invalid_qrels_reports_error(tmp_path: Path) -> None:
    path = tmp_path / "qrels.jsonl"
    path.write_text('{"query_id":"Q","query":"rain","relevant_ids":["missing"]}', encoding="utf-8")
    result = run_cli("evaluate", "--corpus", CORPUS, "--qrels", str(path))
    assert result.returncode == 2
    assert "absent from corpus" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("command", ["retrieve", "evaluate"])
def test_subcommand_help(command: str) -> None:
    result = run_cli(command, "--help")
    assert result.returncode == 0
    assert "--top-k" in result.stdout

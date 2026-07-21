import json

import pytest

from nanoscale.config import Config
from nanoscale.experiments import (
    RunRecord,
    hash_bytes,
    read_manifest,
    read_summary,
)


def _make(tmp_path, **over):
    cfg = Config(**over) if over else Config()
    return RunRecord.create(cfg, base_dir=tmp_path, repo_dir=tmp_path)


def test_run_directory_scaffold(tmp_path):
    rec = _make(tmp_path, name="scaffold")
    d = rec.dir
    assert (d / "resolved_config.yaml").exists()
    assert (d / "environment.json").exists()
    assert (d / "metrics.jsonl").exists()
    summary = read_summary(d)
    assert summary["status"] == "running"
    assert summary["n_params"] == rec.config.n_params()
    assert summary["flops_per_token"] == rec.config.flops_per_token()


def test_log_metrics_appends(tmp_path):
    rec = _make(tmp_path)
    rec.log_metrics({"step": 1, "train_loss": 5.0})
    rec.log_metrics({"step": 2, "train_loss": 4.5})
    lines = (rec.dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["train_loss"] == 4.5


def test_finish_writes_manifest(tmp_path):
    rec = _make(tmp_path)
    rec.finish(final_val_loss=3.2, tokens_seen=1000)
    summary = read_summary(rec.dir)
    assert summary["status"] == "completed"
    assert summary["final_val_loss"] == 3.2
    assert summary["wall_clock_s"] is not None
    rows = read_manifest(tmp_path)
    assert len(rows) == 1 and rows[0]["status"] == "completed"


def test_fail_records_reason(tmp_path):
    rec = _make(tmp_path)
    rec.fail("NaN loss at step 10")
    summary = read_summary(rec.dir)
    assert summary["status"] == "failed"
    assert "NaN" in summary["failure_reason"]
    rows = read_manifest(tmp_path)
    assert rows[0]["status"] == "failed"


def test_context_manager_marks_failure_and_reraises(tmp_path):
    cfg = Config()
    with pytest.raises(RuntimeError, match="boom"):
        with RunRecord.create(cfg, base_dir=tmp_path, repo_dir=tmp_path) as rec:
            run_dir = rec.dir
            raise RuntimeError("boom")
    summary = read_summary(run_dir)
    assert summary["status"] == "failed"
    assert "boom" in summary["failure_reason"]


def test_context_manager_completes_on_success(tmp_path):
    cfg = Config()
    with RunRecord.create(cfg, base_dir=tmp_path, repo_dir=tmp_path) as rec:
        rec.log_metrics({"step": 1})
    summary = read_summary(rec.dir)
    assert summary["status"] == "completed"


def test_hash_bytes_deterministic():
    assert hash_bytes(b"hello") == hash_bytes(b"hello")
    assert hash_bytes(b"hello") != hash_bytes(b"world")

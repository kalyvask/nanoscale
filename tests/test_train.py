import json

import pytest

torch = pytest.importorskip("torch")

from nanoscale.config import Config
from nanoscale.data import prepare
from nanoscale.experiments import read_manifest, read_summary
from nanoscale.tokenizer import Tokenizer
from nanoscale.train import lr_at, train

DOCS = [f"line {i}: the quick brown fox number {i % 7} jumps" for i in range(60)]


@pytest.fixture
def prepared(tmp_path):
    tok = Tokenizer.bytes_tokenizer()
    data_dir = tmp_path / "data"
    prepare(_write(tmp_path, DOCS), tok, data_dir, val_frac=0.2, seed=1)
    return data_dir, tok


def _write(tmp_path, docs):
    p = tmp_path / "corpus.txt"
    p.write_text("\n\n".join(docs), encoding="utf-8")
    return p


def _cfg(**over):
    base = dict(
        name="traintest", vocab_size=257, block_size=16, n_layer=2, n_head=4,
        n_embd=32, batch_size=8, max_steps=120, eval_interval=20, eval_iters=10,
        warmup_frac=0.1, lr=5e-3, z_loss=0.0, attention_backend="reference",
        dataset="corpus", device="cpu",
    )
    base.update(over)
    return Config(**base)


def test_short_run_decreases_loss_and_writes_records(tmp_path, prepared):
    data_dir, _ = prepared
    exp = tmp_path / "experiments"
    summary = train(_cfg(), base_dir=exp, data_dir=data_dir)

    assert summary["status"] == "completed"
    assert summary["final_val_loss"] is not None and summary["final_val_loss"] < 6.0
    assert summary["bits_per_byte"] is not None
    assert summary["tokens_seen"] > 0

    run_dir = exp / "runs" / summary["run_id"]
    for f in ["resolved_config.yaml", "environment.json", "metrics.jsonl", "summary.json"]:
        assert (run_dir / f).exists()

    metrics = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["train_loss"] < metrics[0]["train_loss"]

    rows = read_manifest(exp)
    assert any(r["run_id"] == summary["run_id"] and r["status"] == "completed" for r in rows)


def test_resolved_config_is_recorded(tmp_path, prepared):
    import yaml

    data_dir, _ = prepared
    exp = tmp_path / "experiments"
    cfg = _cfg(n_layer=3)
    summary = train(cfg, base_dir=exp, data_dir=data_dir)
    saved = yaml.safe_load((exp / "runs" / summary["run_id"] / "resolved_config.yaml").read_text())
    assert saved["n_layer"] == 3
    assert saved == cfg.to_dict()


def test_non_finite_loss_is_recorded_as_failed(tmp_path, prepared, monkeypatch):
    import nanoscale.model as model_mod

    data_dir, _ = prepared
    exp = tmp_path / "experiments"

    def bad_loss(self, logits, targets):
        return torch.tensor(float("inf"))

    monkeypatch.setattr(model_mod.GPT, "loss_fn", bad_loss)
    with pytest.raises(ValueError, match="non-finite"):
        train(_cfg(), base_dir=exp, data_dir=data_dir)

    rows = read_manifest(exp)
    assert rows and rows[-1]["status"] == "failed"
    summary = read_summary(exp / "runs" / rows[-1]["run_id"])
    assert "non-finite" in summary["failure_reason"]


def test_lr_schedule_warmup_and_decay():
    cfg = _cfg(lr=1e-2, min_lr_frac=0.1, warmup_frac=0.1, max_steps=100)
    assert lr_at(0, cfg, 100) < cfg.lr           # warmup starts low
    assert lr_at(9, cfg, 100) == pytest.approx(cfg.lr, rel=0.2)  # ~peak after warmup
    assert lr_at(99, cfg, 100) == pytest.approx(cfg.lr * cfg.min_lr_frac, abs=1e-4)

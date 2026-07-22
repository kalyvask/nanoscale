import json

import pytest

pytest.importorskip("torch")

from nanoscale.config import Config
from nanoscale.experiments import RunRecord, completed_identities, run_identity
from nanoscale.planning import build_grid, protocol_hash, variance_pilot
from scripts import run_study


def test_dry_run_prints_full_matrix(capsys):
    run_study.main(["--dry-run", "--gpu", "h100", "--base-dir", "experiments"])
    out = capsys.readouterr().out
    assert "63 runs total" in out
    assert "DRY RUN: nothing was executed" in out
    assert "protocol_hash:" in out
    for scale in ("S", "M", "L"):
        assert f" {scale} " in out


def test_dry_run_pilot_is_ten_runs(capsys):
    run_study.main(["--pilot", "--dry-run", "--base-dir", "experiments"])
    out = capsys.readouterr().out
    assert "10 runs total" in out


def test_run_identity_round_trips(tmp_path):
    cfg = Config(study_id="s", scale_id="S", recipe_id="baseline",
                 init_seed=1, data_seed=2)
    rec = RunRecord.create(cfg, base_dir=tmp_path, repo_dir=tmp_path)
    rec.finish(final_val_loss=1.0)
    ids = completed_identities(tmp_path)
    assert ("s", "S", "baseline", 1, 2) in ids


def test_failed_runs_are_not_treated_as_done(tmp_path):
    cfg = Config(study_id="s", scale_id="S", recipe_id="baseline",
                 init_seed=1, data_seed=2)
    rec = RunRecord.create(cfg, base_dir=tmp_path, repo_dir=tmp_path)
    rec.fail("boom")
    # a failed run must be retried, not skipped
    assert completed_identities(tmp_path) == set()


def test_identity_distinguishes_seeds_and_recipes(tmp_path):
    for recipe in ("baseline", "no_rope"):
        for seed in (1, 2):
            cfg = Config(study_id="s", scale_id="S", recipe_id=recipe,
                         init_seed=seed, data_seed=seed)
            rec = RunRecord.create(cfg, base_dir=tmp_path, repo_dir=tmp_path)
            rec.finish(final_val_loss=1.0)
    assert len(completed_identities(tmp_path)) == 4


def test_every_grid_run_has_a_unique_identity():
    runs = build_grid(Config(vocab_size=16384, block_size=512))
    ids = {
        (r.config.study_id, r.config.scale_id, r.config.recipe_id,
         r.config.resolved_init_seed, r.config.resolved_data_seed)
        for r in runs
    }
    assert len(ids) == len(runs) == 63


def test_protocol_hash_is_stable_and_sensitive():
    base = Config(vocab_size=16384, block_size=512)
    assert protocol_hash(base) == protocol_hash(base)
    # changing a protocol-defining field changes the hash
    assert protocol_hash(base.override(lr=1e-3)) != protocol_hash(base)
    # changing per-run identity does not
    assert protocol_hash(base.override(seed=99)) == protocol_hash(base)


def test_atomic_summary_leaves_no_partial_file(tmp_path):
    cfg = Config(study_id="s", scale_id="S", recipe_id="baseline")
    rec = RunRecord.create(cfg, base_dir=tmp_path, repo_dir=tmp_path)
    rec.finish(final_val_loss=2.0)
    # a valid JSON document, and no stray temp file left behind
    json.loads((rec.dir / "summary.json").read_text())
    assert not list(rec.dir.glob("*.tmp"))

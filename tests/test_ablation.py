import pytest

pytest.importorskip("torch")

from nanoscale.data import prepare
from nanoscale.experiments import read_manifest
from nanoscale.tokenizer import Tokenizer
from scripts import make_table, run_ablation

DOCS = [f"scene {i}: to be or not to be number {i % 5}" for i in range(80)]


@pytest.fixture
def prepared(tmp_path):
    tok = Tokenizer.bytes_tokenizer()
    data_dir = tmp_path / "data"
    p = tmp_path / "corpus.txt"
    p.write_text("\n\n".join(DOCS), encoding="utf-8")
    prepare(p, tok, data_dir, val_frac=0.2, seed=1)
    return data_dir


def test_seven_config_grid_runs_on_cpu(tmp_path, prepared):
    exp = tmp_path / "experiments"
    run_ablation.main([
        "--config", "configs/cpu_smoke.yaml",
        "--data-dir", str(prepared),
        "--base-dir", str(exp),
        # tiny + fast overrides applied to every variant
        "--vocab_size", "257", "--block_size", "16", "--n_embd", "32",
        "--n_layer", "2", "--n_head", "4", "--batch_size", "4",
        "--max_steps", "5", "--eval_interval", "5", "--eval_iters", "3",
    ])

    rows = [r for r in read_manifest(exp) if r.get("group") == "ablation"]
    assert len(rows) == 7
    assert all(r["status"] == "completed" for r in rows)


def test_make_table_classifies_groups(tmp_path, prepared):
    exp = tmp_path / "experiments"
    run_ablation.main([
        "--config", "configs/cpu_smoke.yaml",
        "--data-dir", str(prepared), "--base-dir", str(exp),
        "--vocab_size", "257", "--block_size", "16", "--n_embd", "32",
        "--n_layer", "2", "--n_head", "4", "--batch_size", "4",
        "--max_steps", "5", "--eval_interval", "5", "--eval_iters", "3",
    ])

    summaries = make_table.latest_ablation_runs(str(exp))
    assert len(summaries) == 7
    groups = sorted(make_table.classify(s["config"])[0] for s in summaries)
    assert groups == ["baseline", "efficiency", "quality", "quality",
                      "quality", "stability", "stability"]


def test_classify_labels():
    assert make_table.classify({"pos": "learned", "norm": "rms", "activation": "swiglu",
                                "qk_norm": True, "tie_weights": True, "z_loss": 1e-4}) \
        == ("quality", "RoPE -> learned pos")
    assert make_table.classify({"pos": "rope", "norm": "rms", "activation": "swiglu",
                                "qk_norm": True, "tie_weights": True, "z_loss": 0.0}) \
        == ("stability", "z-loss off")
    assert make_table.classify({"pos": "rope", "norm": "rms", "activation": "swiglu",
                                "qk_norm": True, "tie_weights": True, "z_loss": 1e-4})[0] \
        == "baseline"

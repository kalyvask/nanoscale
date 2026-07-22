import pytest

pytest.importorskip("torch")

from nanoscale.tokenizer import Tokenizer
from scripts import tokenizer_study as ts

TEXT = ("the quick brown fox jumps over the lazy dog. " * 300
        + "internationalization tokenization compression. " * 300)


def test_utilization_and_segment():
    tok = Tokenizer.train(TEXT, vocab_size=400)
    util = tok.utilization("the quick brown fox")
    assert 0.0 < util <= 1.0
    seg = tok.segment("the fox")
    assert "".join(seg) == "the fox"  # pieces reconstruct the text
    assert isinstance(seg, list)


def test_piece_repr_hex_for_non_utf8():
    tok = Tokenizer.bytes_tokenizer()
    # byte 0xff alone is not valid UTF-8 -> hex-escaped
    assert tok.piece_repr(0xFF) == "\\xff"


def test_build_tokenizers_orders_and_bounds():
    toks = ts.build_tokenizers(TEXT, [256 + 20, 256 + 50], sample_bytes=None)
    assert list(toks)[0] == "bytes"
    assert toks["bytes"][1] == 0.0  # no training time for bytes
    for label in ["bpe_276", "bpe_306"]:
        tok, train_time = toks[label]
        assert train_time >= 0.0
        assert tok.vocab_size <= int(label.split("_")[1])


def test_intrinsic_metrics_fields():
    tok = Tokenizer.train(TEXT, vocab_size=400)
    m = ts.intrinsic_metrics(tok, "the quick brown fox jumps")
    assert set(m) >= {"vocab_size", "compression", "fertility", "utilization",
                      "encode_bytes_per_sec", "n_tokens_eval"}
    # BPE should compress better than raw bytes on in-distribution text
    bytes_m = ts.intrinsic_metrics(Tokenizer.bytes_tokenizer(), "the quick brown fox jumps")
    assert m["compression"] > bytes_m["compression"]


def test_split_train_eval_disjoint():
    text = "abcdefghij" * 1000
    train_text, eval_text = ts.split_train_eval(text, sample_bytes=2000, eval_bytes=1000)
    assert len(train_text.encode()) <= 2000
    assert len(eval_text.encode()) <= 1000


def test_eval_slice_nonempty_when_sample_exceeds_corpus():
    """Regression: a sample larger than the corpus must not starve the eval slice."""
    text = "abcdefghij" * 100  # 1000 bytes
    train_text, eval_text = ts.split_train_eval(text, sample_bytes=10_000, eval_bytes=300)
    assert len(eval_text.encode()) == 300
    assert len(train_text.encode()) == 700
    # and the two halves do not overlap
    assert text.encode().endswith(eval_text.encode())


def test_eval_slice_capped_at_half_corpus():
    text = "abcdefghij" * 100  # 1000 bytes
    _, eval_text = ts.split_train_eval(text, sample_bytes=None, eval_bytes=10_000)
    assert len(eval_text.encode()) == 500


def test_render_report_labels_smoke():
    rows = [{"label": "bytes", "vocab_size": 257, "compression": 1.0, "fertility": 5.0,
             "utilization": 0.5, "train_time": 0.0, "encode_bytes_per_sec": 1e6}]
    out = ts.render_report(rows, {"bytes": {}}, is_smoke=True, with_model=False)
    assert "SMOKE-TEST PLUMBING" in out
    assert "NOT A FINDING" in out


def test_script_runs_standalone(tmp_path):
    """Regression: running the file directly must not break `scripts` imports."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    corpus = tmp_path / "c.txt"
    corpus.write_text("\n\n".join(f"doc {i} the quick brown fox" for i in range(40)),
                      encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "tokenizer_study.py"),
         "--input", str(corpus), "--vocab-sizes", "300",
         "--sample-mb", "0.02", "--eval-mb", "0.01", "--out", str(tmp_path / "out")],
        capture_output=True, text=True, cwd=repo, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "TOKENIZER STUDY" in proc.stdout


def test_model_bits_per_byte_path(tmp_path):
    # exercise the equal-budget model path once (bytes tokenizer, tiny budget)
    docs = "\n\n".join(f"scene {i}: the quick brown fox {i % 4}" for i in range(60))
    corpus = tmp_path / "c.txt"
    corpus.write_text(docs, encoding="utf-8")
    tok = Tokenizer.bytes_tokenizer()
    res = ts.model_bits_per_byte(tok, corpus, tmp_path, steps=8, base_dir=tmp_path / "exp")
    assert res["bits_per_byte"] is not None and res["bits_per_byte"] > 0
    assert res["tokens_per_sec"] > 0

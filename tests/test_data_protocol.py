"""Tests for the protocol-validity properties of the data layer."""

import numpy as np
import pytest

from nanoscale.corpora import CorpusSpec, iter_corpus
from nanoscale.data import (
    FrozenEvalSet,
    PackedStream,
    doc_split,
    iter_documents,
    prepare_streaming,
)
from nanoscale.tokenizer import Tokenizer

DOCS = [f"document {i} with words {i % 11} and more text here" for i in range(400)]


# ---------------------------------------------------------------- ingestion
def test_iter_documents_is_lazy(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("\n\n".join(DOCS), encoding="utf-8")
    it = iter_documents(p)
    first = next(it)
    assert first.startswith("document 0")  # produced without reading everything
    assert sum(1 for _ in it) == len(DOCS) - 1


def test_jsonl_ingestion(tmp_path):
    import json

    p = tmp_path / "c.jsonl"
    p.write_text("\n".join(json.dumps({"text": d}) for d in DOCS), encoding="utf-8")
    assert list(iter_documents(p)) == DOCS


# ------------------------------------------------------------- hash split
def test_split_is_content_stable_and_order_independent():
    a = [doc_split(d, val_permille=100) for d in DOCS]
    b = [doc_split(d, val_permille=100) for d in reversed(DOCS)]
    assert a == list(reversed(b))  # depends on content only, not position


def test_split_assignment_survives_corpus_growth():
    """Adding shards must never move an existing document across the split."""
    before = {d: doc_split(d, 100) for d in DOCS[:100]}
    grown = DOCS + [f"new doc {i}" for i in range(500)]
    after = {d: doc_split(d, 100) for d in grown if d in before}
    assert before == after


def test_val_permille_controls_proportion():
    frac = sum(doc_split(d, 100) == "val" for d in DOCS) / len(DOCS)
    assert 0.03 < frac < 0.20  # ~10% with sampling slack


# --------------------------------------------------- incremental writing
def test_prepare_streaming_writes_counts_and_hashes(tmp_path):
    tok = Tokenizer.bytes_tokenizer()
    meta = prepare_streaming(iter(DOCS), tok, tmp_path / "out", val_permille=100)
    assert meta["n_tokens_train"] > 0 and meta["n_tokens_val"] > 0
    assert meta["n_docs_train"] + meta["n_docs_val"] == len(DOCS)
    assert meta["tokenizer_hash"] == tok.content_hash()
    assert len(meta["dataset_hash"]) == 16


def test_prepare_streaming_is_reproducible(tmp_path):
    tok = Tokenizer.bytes_tokenizer()
    m1 = prepare_streaming(iter(DOCS), tok, tmp_path / "a", val_permille=100)
    m2 = prepare_streaming(iter(DOCS), tok, tmp_path / "b", val_permille=100)
    assert m1["dataset_hash"] == m2["dataset_hash"]


def test_empty_split_is_rejected(tmp_path):
    tok = Tokenizer.bytes_tokenizer()
    with pytest.raises(ValueError, match="empty split"):
        prepare_streaming(iter(DOCS[:3]), tok, tmp_path / "out", val_permille=1)


# ------------------------------------------------------- packed stream
def _stream(n_tokens=10_000, block=16, seed=0):
    data = np.arange(n_tokens, dtype=np.uint16)
    return PackedStream(data, block, seed)


def test_packed_stream_is_deterministic():
    a, b = _stream(seed=7), _stream(seed=7)
    assert np.array_equal(a.block_order, b.block_order)
    x1, y1 = a.batch(0, 4)
    x2, y2 = b.batch(0, 4)
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)


def test_packed_stream_targets_are_shifted():
    s = _stream()
    x, y = s.batch(0, 4)
    assert np.array_equal(y[:, :-1], x[:, 1:])


def test_packed_stream_visits_blocks_without_replacement_within_an_epoch():
    s = _stream(n_tokens=1000, block=10)  # 99 blocks
    seen = [int(s.block_order[i]) for i in range(s.n_blocks)]
    assert len(set(seen)) == s.n_blocks  # every block exactly once per epoch


def test_nested_prefixes_across_scales():
    """S, M and L must consume nested prefixes of the same stream for a data_seed."""
    s = _stream(n_tokens=100_000, block=16, seed=42)
    small = set(s.prefix_block_ids(1_000).tolist())
    medium = set(s.prefix_block_ids(4_000).tolist())
    large = set(s.prefix_block_ids(16_000).tolist())
    assert small <= medium <= large
    assert len(small) < len(medium) < len(large)


def test_different_data_seeds_give_different_orders():
    a, b = _stream(seed=1), _stream(seed=2)
    assert not np.array_equal(a.block_order, b.block_order)


def test_epochs_reported_for_budget():
    s = _stream(n_tokens=1600, block=16)  # 99 blocks
    assert s.epochs_for_tokens(1600) == pytest.approx(100 / 99, rel=0.02)


# --------------------------------------------------------- frozen eval
def test_frozen_eval_independent_of_training_seed():
    data = np.arange(50_000, dtype=np.uint16)
    a = FrozenEvalSet(data, 16, n_batches=3, batch_size=4, eval_seed=999)
    b = FrozenEvalSet(data, 16, n_batches=3, batch_size=4, eval_seed=999)
    assert np.array_equal(a.block_ids, b.block_ids)
    assert a.content_hash() == b.content_hash()
    # a different eval_seed gives a different (but still frozen) set
    c = FrozenEvalSet(data, 16, n_batches=3, batch_size=4, eval_seed=1000)
    assert c.content_hash() != a.content_hash()


def test_frozen_eval_batches_are_stable_and_shifted():
    data = np.arange(50_000, dtype=np.uint16)
    es = FrozenEvalSet(data, 16, n_batches=2, batch_size=4, eval_seed=5)
    first = [ (x.copy(), y.copy()) for x, y in es.batches() ]
    second = [ (x.copy(), y.copy()) for x, y in es.batches() ]
    for (x1, y1), (x2, y2) in zip(first, second):
        assert np.array_equal(x1, x2) and np.array_equal(y1, y2)
        assert np.array_equal(y1[:, :-1], x1[:, 1:])


def test_frozen_eval_no_duplicate_blocks_when_data_is_ample():
    data = np.arange(50_000, dtype=np.uint16)
    es = FrozenEvalSet(data, 16, n_batches=4, batch_size=4, eval_seed=3)
    assert len(set(es.block_ids.tolist())) == len(es.block_ids)


# ------------------------------------------------------------- corpora
def test_hf_corpus_refuses_unpinned_revision():
    with pytest.raises(ValueError, match="immutable revision"):
        CorpusSpec(name="x", kind="hf", repo_id="a/b", revision="main")
    with pytest.raises(ValueError, match="immutable revision"):
        CorpusSpec(name="x", kind="hf", repo_id="a/b", revision=None)


def test_shards_are_ordered_deterministically():
    spec = CorpusSpec(name="x", kind="hf", repo_id="a/b", revision="abc123",
                      shards=("c.parquet", "a.parquet", "b.parquet"))
    assert spec.ordered_shards == ("a.parquet", "b.parquet", "c.parquet")
    assert spec.metadata()["n_shards"] == 3


def test_local_corpus_iterates(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("\n\n".join(DOCS[:10]), encoding="utf-8")
    spec = CorpusSpec(name="local", kind="local", path=str(p), max_documents=5)
    assert len(list(iter_corpus(spec))) == 5


def test_fineweb_config_is_pinned_to_an_immutable_revision():
    """The shipped config must name a commit sha, never a moving branch."""
    spec = CorpusSpec.from_yaml("configs/corpora/fineweb_edu.yaml")
    assert spec.revision and len(spec.revision) == 40
    assert all(c in "0123456789abcdef" for c in spec.revision)
    assert spec.revision not in ("main", "master")


def test_fineweb_shards_are_explicit_and_ordered():
    """An implicit shard set would make the token stream irreproducible."""
    spec = CorpusSpec.from_yaml("configs/corpora/fineweb_edu.yaml")
    assert len(spec.shards) >= 1
    assert spec.ordered_shards == tuple(sorted(spec.shards))
    assert all(s.endswith(".parquet") for s in spec.ordered_shards)
    # the recorded metadata carries the pin, so runs can be audited later
    meta = spec.metadata()
    assert meta["revision"] == spec.revision
    assert meta["n_shards"] == len(spec.shards)

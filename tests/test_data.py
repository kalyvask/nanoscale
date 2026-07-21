import json

import numpy as np
import pytest

from nanoscale.data import (
    encode_documents,
    get_batch,
    load_meta,
    load_split,
    prepare,
    read_documents,
    sample_batch,
    split_documents,
)
from nanoscale.tokenizer import Tokenizer

DOCS = [f"document number {i} with some words {i} {i}" for i in range(40)]


@pytest.fixture(scope="module")
def tok():
    return Tokenizer.bytes_tokenizer()


def _write_txt(tmp_path, docs):
    p = tmp_path / "corpus.txt"
    p.write_text("\n\n".join(docs), encoding="utf-8")
    return p


def test_read_txt_documents(tmp_path):
    p = _write_txt(tmp_path, DOCS)
    docs = read_documents(p, fmt="txt")
    assert len(docs) == len(DOCS)


def test_read_jsonl_documents(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps({"text": d}) for d in DOCS), encoding="utf-8")
    docs = read_documents(p, text_field="text")
    assert docs == DOCS


def test_split_is_deterministic_and_disjoint():
    train, val = split_documents(DOCS, val_frac=0.25, seed=7)
    train2, val2 = split_documents(DOCS, val_frac=0.25, seed=7)
    assert train == train2 and val == val2
    assert set(train).isdisjoint(set(val))
    assert len(train) + len(val) == len(DOCS)
    # different seed can produce a different split
    train3, _ = split_documents(DOCS, val_frac=0.25, seed=8)
    assert train3 != train


def test_no_document_leakage_after_prepare(tmp_path, tok):
    p = _write_txt(tmp_path, DOCS)
    out = tmp_path / "prepared"
    meta = prepare(p, tok, out, val_frac=0.25, seed=7)
    # reconstruct the split and confirm disjoint document sets
    train_docs, val_docs = split_documents(DOCS, val_frac=0.25, seed=7)
    assert set(train_docs).isdisjoint(set(val_docs))
    assert meta["n_docs_train"] == len(train_docs)
    assert meta["n_docs_val"] == len(val_docs)


def test_prepare_writes_bins_and_meta(tmp_path, tok):
    p = _write_txt(tmp_path, DOCS)
    out = tmp_path / "prepared"
    meta = prepare(p, tok, out, val_frac=0.2, seed=1)
    assert (out / "train.bin").exists() and (out / "val.bin").exists()
    saved = load_meta(out)
    assert saved["tokenizer_hash"] == tok.content_hash()
    assert saved["dtype"] == "uint16"
    assert saved["n_tokens_train"] > 0 and saved["n_tokens_val"] > 0
    train = load_split(out, "train")
    assert train.dtype == np.uint16


def test_prepare_requires_multiple_documents(tmp_path, tok):
    p = tmp_path / "single.txt"
    p.write_text("just one document with no blank lines", encoding="utf-8")
    with pytest.raises(ValueError, match=">= 2 documents"):
        prepare(p, tok, tmp_path / "out")


def test_eot_between_documents(tok):
    ids = encode_documents(["ab", "cd"], tok)
    eot = tok.special_tokens["<|endoftext|>"]
    assert eot in ids
    assert ids[-1] == eot  # trailing separator


def test_sample_batch_shifted_targets_and_shape():
    data = np.arange(100, dtype=np.uint16)
    rng = np.random.default_rng(0)
    x, y = sample_batch(data, block_size=8, batch_size=4, rng=rng)
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert x.dtype == np.int64
    # y is x shifted by one position in the source stream
    assert np.array_equal(y[:, :-1], x[:, 1:])


def test_sample_batch_reproducible_under_seed():
    data = np.arange(200, dtype=np.uint16)
    x1, y1 = sample_batch(data, 8, 4, np.random.default_rng(123))
    x2, y2 = sample_batch(data, 8, 4, np.random.default_rng(123))
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)


def test_get_batch_returns_torch_tensors():
    torch = pytest.importorskip("torch")
    data = np.arange(200, dtype=np.uint16)
    x, y = get_batch(data, block_size=8, batch_size=4, device="cpu",
                     rng=np.random.default_rng(0))
    assert isinstance(x, torch.Tensor) and x.dtype == torch.int64
    assert x.shape == (4, 8)


def test_sample_batch_rejects_short_data():
    with pytest.raises(ValueError, match="too short"):
        sample_batch(np.arange(4, dtype=np.uint16), block_size=8, batch_size=2,
                     rng=np.random.default_rng(0))

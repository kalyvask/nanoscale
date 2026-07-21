import pytest

from nanoscale.tokenizer import BYTE_VOCAB, Tokenizer

TRAIN_TEXT = ("the quick brown fox jumps over the lazy dog. " * 200
              + "hello world, hello nanoscale. " * 200)


@pytest.fixture(scope="module")
def bpe():
    return Tokenizer.train(TRAIN_TEXT, vocab_size=400, max_bytes=None)


def test_bytes_fallback_round_trip():
    tok = Tokenizer.bytes_tokenizer()
    assert tok.vocab_size == BYTE_VOCAB + 1  # + endoftext
    for s in ["hello", "", "a\nb\tc", "unicode: café ☕ 日本語"]:
        assert tok.decode(tok.encode_ordinary(s)) == s


def test_unicode_round_trip(bpe):
    for s in ["hello world", "café ☕ 日本語 🚀", "mixed 123 !@#\n\ttabs", ""]:
        assert bpe.decode(bpe.encode_ordinary(s)) == s


def test_arbitrary_byte_round_trip(bpe):
    for raw in [b"", b"\x00\x01\x02\xff\xfe", bytes(range(256)), b"normal text"]:
        assert bpe.decode_bytes(bpe.encode_bytes(raw)) == raw


def test_empty_input(bpe):
    assert bpe.encode_ordinary("") == []
    assert bpe.decode([]) == ""
    assert bpe.encode("", allowed_special="all") == []


def test_compression_from_merges(bpe):
    # BPE should encode training-like text in fewer tokens than raw bytes
    text = "the quick brown fox"
    n_tokens = len(bpe.encode_ordinary(text))
    assert n_tokens < len(text.encode("utf-8"))
    st = bpe.stats(text)
    assert st["compression"] > 1.0
    assert st["fertility"] > 0


def test_special_tokens(bpe):
    text = "hello<|endoftext|>world"
    # not allowed: the literal is encoded as ordinary bytes and round-trips
    ordinary = bpe.encode(text, allowed_special="none")
    assert bpe.decode(ordinary) == text
    # allowed: the special becomes a single id above the BPE range
    with_special = bpe.encode(text, allowed_special="all")
    eot = bpe.special_tokens["<|endoftext|>"]
    assert eot in with_special
    assert eot >= BYTE_VOCAB + len(bpe.merges)
    assert bpe.decode(with_special) == text


def test_special_token_not_produced_by_merges(bpe):
    # ordinary encoding of arbitrary text never emits a special id
    ids = bpe.encode_ordinary(TRAIN_TEXT[:500])
    assert all(i not in bpe._special_inv for i in ids)


def test_training_is_deterministic():
    a = Tokenizer.train(TRAIN_TEXT, vocab_size=350)
    b = Tokenizer.train(TRAIN_TEXT, vocab_size=350)
    assert a.merges == b.merges
    assert a.special_tokens == b.special_tokens
    assert a.content_hash() == b.content_hash()


def test_max_bytes_bounds_training():
    full = Tokenizer.train(TRAIN_TEXT, vocab_size=350)
    bounded = Tokenizer.train(TRAIN_TEXT, vocab_size=350, max_bytes=500)
    # different training sample -> different learned merges
    assert full.merges != bounded.merges


def test_save_load_equivalence(bpe, tmp_path):
    path = tmp_path / "tok.json"
    bpe.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.merges == bpe.merges
    assert loaded.special_tokens == bpe.special_tokens
    assert loaded.content_hash() == bpe.content_hash()
    text = "the quick brown fox jumps"
    assert loaded.encode_ordinary(text) == bpe.encode_ordinary(text)


def test_vocab_size_constraint():
    # the tokenizer never exceeds the requested vocabulary
    tok = Tokenizer.train(TRAIN_TEXT, vocab_size=320)
    assert tok.vocab_size <= 320
    # asking for far more than the (repetitive) data supports caps out, never exceeds
    capped = Tokenizer.train(TRAIN_TEXT, vocab_size=5000)
    assert capped.vocab_size <= 5000
    # more vocabulary budget yields at least as many merges
    assert capped.vocab_size >= tok.vocab_size
    with pytest.raises(ValueError, match="vocab_size"):
        Tokenizer.train(TRAIN_TEXT, vocab_size=256)  # no room for special token

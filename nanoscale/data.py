"""Data pipeline: iterable document ingestion -> hash-split -> packed token stream.

Protocol properties this module is responsible for:

* **Streaming ingestion.** Documents are consumed from an iterator and tokens are
  written incrementally. Nothing builds a corpus-wide Python list, so a multi-billion
  token corpus is bounded by disk, not RAM.
* **Stable hash-based split.** A document's split is decided by a hash of its content
  and a fixed salt, so it does not depend on corpus order, corpus size, or how many
  shards happened to be present. Re-running on more shards never moves an existing
  document from train to val.
* **Deterministic packed stream.** Training consumes whole, non-overlapping blocks in
  a permuted order fixed by ``data_seed``. Sampling random offsets with replacement
  (the old behaviour) meant two runs saw different amounts of unique data and some
  tokens several times, which is not a controlled budget.
* **Nested prefixes.** Because the permutation depends only on ``data_seed``, the
  blocks that S trains on are a prefix of M's, which are a prefix of L's. Scale is
  then the only thing that differs across tiers, not which data was seen.
* **Frozen evaluation.** The eval batches are chosen by a dedicated ``eval_seed`` that
  is independent of the training seed, so every run in the study is scored on exactly
  the same examples.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from nanoscale.tokenizer import Tokenizer

STORAGE_DTYPE = np.uint16
DEFAULT_DELIMITER = "\n\n"
DEFAULT_SPLIT_SALT = "nanoscale-split-v1"
HASH_BUCKETS = 10_000
WRITE_CHUNK_TOKENS = 1 << 20  # flush to disk about every million tokens


# ---------------------------------------------------------------------- #
# ingestion (iterable, never corpus-wide)
# ---------------------------------------------------------------------- #
def iter_documents(
    path: str | Path,
    fmt: str | None = None,
    text_field: str = "text",
    delimiter: str = DEFAULT_DELIMITER,
) -> Iterator[str]:
    """Yield documents one at a time from a TXT or JSONL file."""
    path = Path(path)
    fmt = fmt or ("jsonl" if path.suffix.lower() in (".jsonl", ".ndjson") else "txt")
    if fmt == "jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get(text_field)
                if text:
                    yield text
        return
    if fmt == "txt":
        # stream on the delimiter rather than reading the whole file into memory
        buf = ""
        with open(path, "r", encoding="utf-8") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                buf += chunk
                parts = buf.split(delimiter)
                buf = parts.pop()
                for p in parts:
                    if p.strip():
                        yield p
        if buf.strip():
            yield buf
        return
    raise ValueError(f"unknown format: {fmt!r}")


def read_documents(path, fmt=None, text_field="text", delimiter=DEFAULT_DELIMITER) -> list[str]:
    """Eager convenience wrapper. Only for small corpora (tests, smoke runs)."""
    return list(iter_documents(path, fmt=fmt, text_field=text_field, delimiter=delimiter))


# ---------------------------------------------------------------------- #
# stable hash-based split
# ---------------------------------------------------------------------- #
def doc_bucket(doc: str, salt: str = DEFAULT_SPLIT_SALT) -> int:
    h = hashlib.sha1(salt.encode("utf-8") + doc.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % HASH_BUCKETS


def doc_split(doc: str, val_permille: int, salt: str = DEFAULT_SPLIT_SALT) -> str:
    """Assign a document to train or val by content hash.

    Deterministic and order-independent: adding shards never reassigns a document.
    """
    return "val" if doc_bucket(doc, salt) < val_permille * (HASH_BUCKETS // 1000) else "train"


def split_documents(docs: list[str], val_frac: float, seed: int = 0,
                    salt: str = DEFAULT_SPLIT_SALT) -> tuple[list[str], list[str]]:
    """Back-compat helper: hash-split an in-memory list (``seed`` folded into the salt)."""
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0, 1); got {val_frac}")
    val_permille = max(1, int(round(val_frac * 1000)))
    effective_salt = f"{salt}:{seed}"
    train, val = [], []
    for d in docs:
        (val if doc_split(d, val_permille, effective_salt) == "val" else train).append(d)
    return train, val


# ---------------------------------------------------------------------- #
# incremental writing
# ---------------------------------------------------------------------- #
class _SplitWriter:
    """Appends tokens to a .bin while maintaining running counts and a content hash."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh = open(path, "wb")
        self._buf: list[int] = []
        self.n_tokens = 0
        self.n_docs = 0
        self.n_bytes = 0
        self._hash = hashlib.sha256()

    def add(self, ids: list[int], raw_bytes: int, doc_hash: bytes) -> None:
        self._buf.extend(ids)
        self.n_tokens += len(ids)
        self.n_docs += 1
        self.n_bytes += raw_bytes
        self._hash.update(doc_hash)
        if len(self._buf) >= WRITE_CHUNK_TOKENS:
            self.flush()

    def flush(self) -> None:
        if self._buf:
            np.asarray(self._buf, dtype=STORAGE_DTYPE).tofile(self._fh)
            self._buf.clear()

    def close(self) -> str:
        self.flush()
        self._fh.close()
        return self._hash.hexdigest()[:16]


def prepare_streaming(
    docs: Iterable[str],
    tokenizer: Tokenizer,
    out_dir: str | Path,
    val_permille: int = 100,
    split_salt: str = DEFAULT_SPLIT_SALT,
    dataset_name: str = "corpus",
    corpus_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tokenize an iterable of documents into train/val memmaps, incrementally."""
    if tokenizer.vocab_size > np.iinfo(STORAGE_DTYPE).max + 1:
        raise ValueError(f"tokenizer vocab ({tokenizer.vocab_size}) exceeds uint16 storage")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eot = tokenizer.special_tokens.get("<|endoftext|>")
    writers = {s: _SplitWriter(out_dir / f"{s}.bin") for s in ("train", "val")}
    for doc in docs:
        split = doc_split(doc, val_permille, split_salt)
        ids = tokenizer.encode_ordinary(doc)
        if eot is not None:
            ids = ids + [eot]
        writers[split].add(ids, len(doc.encode("utf-8")),
                           hashlib.sha1(doc.encode("utf-8")).digest())

    hashes = {s: w.close() for s, w in writers.items()}
    if writers["train"].n_tokens == 0 or writers["val"].n_tokens == 0:
        raise ValueError(
            f"empty split after hashing (train={writers['train'].n_tokens} tokens, "
            f"val={writers['val'].n_tokens}); corpus too small for val_permille={val_permille}"
        )

    meta = {
        "dataset": dataset_name,
        "dataset_hash": hashlib.sha256(
            (hashes["train"] + hashes["val"]).encode()
        ).hexdigest()[:16],
        "tokenizer_hash": tokenizer.content_hash(),
        "vocab_size": tokenizer.vocab_size,
        "dtype": np.dtype(STORAGE_DTYPE).name,
        "split_salt": split_salt,
        "val_permille": val_permille,
        "n_docs_train": writers["train"].n_docs,
        "n_docs_val": writers["val"].n_docs,
        "n_tokens_train": writers["train"].n_tokens,
        "n_tokens_val": writers["val"].n_tokens,
        "n_bytes_train": writers["train"].n_bytes,
        "n_bytes_val": writers["val"].n_bytes,
        "compression_val": writers["val"].n_bytes / max(1, writers["val"].n_tokens),
        "corpus": corpus_meta or {},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def prepare(
    input_path: str | Path,
    tokenizer: Tokenizer,
    out_dir: str | Path,
    val_frac: float = 0.1,
    seed: int = 1337,
    fmt: str | None = None,
    text_field: str = "text",
    delimiter: str = DEFAULT_DELIMITER,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Prepare a local file. Thin wrapper over the streaming path."""
    docs = iter_documents(input_path, fmt=fmt, text_field=text_field, delimiter=delimiter)
    # peek so we can fail loudly on a corpus with too few documents to split
    docs = list(docs)
    if len(docs) < 2:
        raise ValueError(
            f"need >= 2 documents to split without leakage; got {len(docs)}. "
            "Adjust the delimiter or provide more documents."
        )
    return prepare_streaming(
        docs, tokenizer, out_dir,
        val_permille=max(1, int(round(val_frac * 1000))),
        split_salt=f"{DEFAULT_SPLIT_SALT}:{seed}",
        dataset_name=dataset_name or Path(input_path).stem,
    )


def encode_documents(docs: list[str], tokenizer: Tokenizer, add_eot: bool = True) -> np.ndarray:
    eot = tokenizer.special_tokens.get("<|endoftext|>")
    ids: list[int] = []
    for doc in docs:
        ids.extend(tokenizer.encode_ordinary(doc))
        if add_eot and eot is not None:
            ids.append(eot)
    return np.asarray(ids, dtype=STORAGE_DTYPE)


def load_split(out_dir: str | Path, split: str) -> np.ndarray:
    return np.memmap(Path(out_dir) / f"{split}.bin", dtype=STORAGE_DTYPE, mode="r")


def load_meta(out_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(out_dir) / "meta.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------- #
# deterministic packed training stream
# ---------------------------------------------------------------------- #
class PackedStream:
    """Non-overlapping blocks consumed in a permutation fixed by ``data_seed``.

    ``block_order`` depends only on the data and the seed, never on the model, so all
    scales sharing a ``data_seed`` walk the same order and shorter budgets are strict
    prefixes of longer ones.
    """

    def __init__(self, data: np.ndarray, block_size: int, data_seed: int) -> None:
        self.data = data
        self.block_size = block_size
        self.data_seed = data_seed
        # need one extra token for the shifted target of the final block
        self.n_blocks = max(0, (len(data) - 1) // block_size)
        if self.n_blocks < 1:
            raise ValueError(
                f"data too short ({len(data)} tokens) for block_size {block_size}"
            )
        rng = np.random.default_rng(data_seed)
        self.block_order = rng.permutation(self.n_blocks)

    def blocks_for_tokens(self, total_tokens: int) -> int:
        return int(np.ceil(total_tokens / self.block_size))

    def epochs_for_tokens(self, total_tokens: int) -> float:
        return self.blocks_for_tokens(total_tokens) / self.n_blocks

    def take(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Block at position ``index`` in the permuted order (wraps for extra epochs)."""
        b = int(self.block_order[index % self.n_blocks])
        start = b * self.block_size
        x = self.data[start : start + self.block_size].astype(np.int64)
        y = self.data[start + 1 : start + 1 + self.block_size].astype(np.int64)
        return x, y

    def batch(self, step: int, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for i in range(batch_size):
            x, y = self.take(step * batch_size + i)
            xs.append(x)
            ys.append(y)
        return np.stack(xs), np.stack(ys)

    def prefix_block_ids(self, total_tokens: int) -> np.ndarray:
        """The set of source blocks a budget of ``total_tokens`` will consume."""
        n = min(self.blocks_for_tokens(total_tokens), self.n_blocks)
        return self.block_order[:n]


# ---------------------------------------------------------------------- #
# frozen evaluation set
# ---------------------------------------------------------------------- #
class FrozenEvalSet:
    """A fixed set of eval blocks, chosen independently of any training seed."""

    def __init__(self, data: np.ndarray, block_size: int, n_batches: int,
                 batch_size: int, eval_seed: int = 12345) -> None:
        self.block_size = block_size
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.eval_seed = eval_seed
        n_blocks = max(0, (len(data) - 1) // block_size)
        if n_blocks < 1:
            raise ValueError(f"val data too short ({len(data)}) for block_size {block_size}")
        needed = n_batches * batch_size
        rng = np.random.default_rng(eval_seed)
        if needed <= n_blocks:
            chosen = rng.choice(n_blocks, size=needed, replace=False)
        else:  # small val split: allow repeats but keep it deterministic
            chosen = rng.choice(n_blocks, size=needed, replace=True)
        self.block_ids = np.sort(chosen)
        self.data = data

    def batches(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for b in range(self.n_batches):
            ids = self.block_ids[b * self.batch_size : (b + 1) * self.batch_size]
            xs, ys = [], []
            for blk in ids:
                start = int(blk) * self.block_size
                xs.append(self.data[start : start + self.block_size].astype(np.int64))
                ys.append(self.data[start + 1 : start + 1 + self.block_size].astype(np.int64))
            yield np.stack(xs), np.stack(ys)

    def content_hash(self) -> str:
        payload = json.dumps({
            "block_ids": self.block_ids.tolist(),
            "block_size": self.block_size,
            "batch_size": self.batch_size,
            "n_batches": self.n_batches,
            "eval_seed": self.eval_seed,
        }, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


# ---------------------------------------------------------------------- #
# legacy random sampling (smoke/tests only)
# ---------------------------------------------------------------------- #
def sample_batch(data: np.ndarray, block_size: int, batch_size: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    high = len(data) - block_size - 1
    if high < 1:
        raise ValueError(f"data too short ({len(data)}) for block_size {block_size}")
    ix = rng.integers(0, high, size=batch_size)
    x = np.stack([data[i : i + block_size].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size].astype(np.int64) for i in ix])
    return x, y


def get_batch(data, block_size, batch_size, device="cpu", rng=None):
    import torch

    if rng is None:
        rng = np.random.default_rng()
    x, y = sample_batch(data, block_size, batch_size, rng)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    if device != "cpu":
        xt = xt.pin_memory().to(device, non_blocking=True)
        yt = yt.pin_memory().to(device, non_blocking=True)
    return xt, yt


def to_torch(x: np.ndarray, y: np.ndarray, device: str = "cpu"):
    import torch

    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    if device != "cpu":
        xt = xt.pin_memory().to(device, non_blocking=True)
        yt = yt.pin_memory().to(device, non_blocking=True)
    return xt, yt

"""Data pipeline: local text/JSONL -> document-split -> memmapped token stream.

The split is done at the **document** level *before* concatenation, so no document
straddles the train/val boundary (no leakage). The split is deterministic under a
recorded seed. Tokens are stored as ``uint16`` memmaps; a ``meta.json`` records the
dataset hash, tokenizer hash, document/token counts, and the split seed so a run is
fully reconstructible.

``get_batch`` returns torch tensors; the underlying sampling is pure numpy and seeded,
so batches are reproducible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from nanoscale.tokenizer import Tokenizer

STORAGE_DTYPE = np.uint16
DEFAULT_DELIMITER = "\n\n"


# ---------------------------------------------------------------------- #
# ingestion
# ---------------------------------------------------------------------- #
def read_documents(
    path: str | Path,
    fmt: str | None = None,
    text_field: str = "text",
    delimiter: str = DEFAULT_DELIMITER,
) -> list[str]:
    """Read a corpus into a list of documents.

    TXT is split on ``delimiter`` (default blank lines). JSONL yields one document
    per line, reading ``text_field``.
    """
    path = Path(path)
    fmt = fmt or ("jsonl" if path.suffix.lower() in (".jsonl", ".ndjson") else "txt")
    if fmt == "txt":
        raw = path.read_text(encoding="utf-8")
        docs = [d for d in raw.split(delimiter) if d.strip()]
        return docs
    if fmt == "jsonl":
        docs = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get(text_field)
            if text:
                docs.append(text)
        return docs
    raise ValueError(f"unknown format: {fmt!r}")


def split_documents(
    docs: list[str], val_frac: float, seed: int
) -> tuple[list[str], list[str]]:
    """Deterministic document-level split. No document appears in both sides."""
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0, 1); got {val_frac}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(docs))
    n_val = max(1, int(round(len(docs) * val_frac)))
    val_idx = set(order[:n_val].tolist())
    train, val = [], []
    for i, doc in enumerate(docs):
        (val if i in val_idx else train).append(doc)
    return train, val


def encode_documents(
    docs: list[str], tokenizer: Tokenizer, add_eot: bool = True
) -> np.ndarray:
    """Encode and concatenate documents, inserting an end-of-text id between them."""
    eot = tokenizer.special_tokens.get("<|endoftext|>")
    ids: list[int] = []
    for doc in docs:
        ids.extend(tokenizer.encode_ordinary(doc))
        if add_eot and eot is not None:
            ids.append(eot)
    return np.asarray(ids, dtype=STORAGE_DTYPE)


def _hash_docs(docs: list[str]) -> str:
    h = hashlib.sha256()
    for d in docs:
        h.update(d.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------- #
# prepare / load
# ---------------------------------------------------------------------- #
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
    if tokenizer.vocab_size > np.iinfo(STORAGE_DTYPE).max + 1:
        raise ValueError(
            f"tokenizer vocab ({tokenizer.vocab_size}) exceeds uint16 storage"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = read_documents(input_path, fmt=fmt, text_field=text_field, delimiter=delimiter)
    if len(docs) < 2:
        raise ValueError(
            f"need >= 2 documents to split without leakage; got {len(docs)}. "
            "Adjust the delimiter or provide more documents."
        )
    train_docs, val_docs = split_documents(docs, val_frac=val_frac, seed=seed)
    train_ids = encode_documents(train_docs, tokenizer)
    val_ids = encode_documents(val_docs, tokenizer)

    n_bytes_train = sum(len(d.encode("utf-8")) for d in train_docs)
    n_bytes_val = sum(len(d.encode("utf-8")) for d in val_docs)

    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")

    meta = {
        "dataset": dataset_name or Path(input_path).stem,
        "dataset_hash": _hash_docs(docs),
        "tokenizer_hash": tokenizer.content_hash(),
        "vocab_size": tokenizer.vocab_size,
        "dtype": np.dtype(STORAGE_DTYPE).name,
        "split_seed": seed,
        "val_frac": val_frac,
        "n_docs_total": len(docs),
        "n_docs_train": len(train_docs),
        "n_docs_val": len(val_docs),
        "n_tokens_train": int(train_ids.size),
        "n_tokens_val": int(val_ids.size),
        "n_bytes_train": n_bytes_train,
        "n_bytes_val": n_bytes_val,
        "compression_val": n_bytes_val / max(1, int(val_ids.size)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_split(out_dir: str | Path, split: str) -> np.ndarray:
    path = Path(out_dir) / f"{split}.bin"
    return np.memmap(path, dtype=STORAGE_DTYPE, mode="r")


def load_meta(out_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(out_dir) / "meta.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------- #
# batching
# ---------------------------------------------------------------------- #
def sample_batch(
    data: np.ndarray, block_size: int, batch_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample x and one-shifted y as int64 numpy arrays. Seeded via ``rng``."""
    high = len(data) - block_size - 1
    if high < 1:
        raise ValueError(
            f"data too short ({len(data)}) for block_size {block_size}"
        )
    ix = rng.integers(0, high, size=batch_size)
    x = np.stack([data[i : i + block_size].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size].astype(np.int64) for i in ix])
    return x, y


def get_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
    rng: np.random.Generator | None = None,
):
    """Return (x, y) torch LongTensors on ``device``. Reproducible when ``rng`` is seeded."""
    import torch

    if rng is None:
        rng = np.random.default_rng()
    x, y = sample_batch(data, block_size, batch_size, rng)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    if device != "cpu":
        xt = xt.pin_memory().to(device, non_blocking=True)
        yt = yt.pin_memory().to(device, non_blocking=True)
    return xt, yt

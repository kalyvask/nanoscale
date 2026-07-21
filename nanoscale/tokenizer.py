"""Byte-level BPE tokenizer, written from scratch (CS336 Lecture 1 / Assignment 1).

Design choices that matter for the study:

* **Byte-level.** The base vocabulary is the 256 bytes, so any input round-trips
  exactly at the byte level and there are no out-of-vocabulary characters.
* **Deterministic training.** On equal pair counts we merge the lexicographically
  smallest pair, so training the same text twice yields identical merges.
* **Bounded training sample.** ``train(..., max_bytes=...)`` caps how much text the
  merges are learned from, so we never blindly scan an entire corpus.
* **Special tokens** (e.g. ``<|endoftext|>``) live above the BPE ids, are never
  produced by merges, and are only emitted when explicitly allowed.

The tokenizer is frozen for the real study; :meth:`save`/:meth:`load` round-trip it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

BYTE_VOCAB = 256
DEFAULT_SPECIALS = ("<|endoftext|>",)


def _count_pairs(ids: list[int], counts: dict[tuple[int, int], int] | None = None):
    counts = {} if counts is None else counts
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(
        self,
        merges: dict[tuple[int, int], int],
        special_tokens: dict[str, int] | None = None,
        mode: str = "bpe",
    ) -> None:
        self.mode = mode
        # merges preserved in creation order (dict preserves insertion order)
        self.merges = dict(merges)
        self.special_tokens = dict(special_tokens or {})
        self._build_vocab()

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def _build_vocab(self) -> None:
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(BYTE_VOCAB)}
        for (a, b), new_id in self.merges.items():
            vocab[new_id] = vocab[a] + vocab[b]
        self.vocab = vocab
        # rank = merge priority (lower merges first) for encoding
        self._ranks = {pair: i for i, pair in enumerate(self.merges.keys())}
        self._special_inv = {i: s for s, i in self.special_tokens.items()}
        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
            self._special_re = re.compile(pattern)
        else:
            self._special_re = None

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        max_bytes: int | None = None,
        special_tokens: Iterable[str] | None = DEFAULT_SPECIALS,
    ) -> "Tokenizer":
        specials = list(special_tokens or [])
        reserved = len(specials)
        if vocab_size < BYTE_VOCAB + reserved:
            raise ValueError(
                f"vocab_size ({vocab_size}) must be >= {BYTE_VOCAB + reserved} "
                f"(256 bytes + {reserved} special tokens)"
            )
        num_merges = vocab_size - BYTE_VOCAB - reserved

        sample = text.encode("utf-8")
        if max_bytes is not None:
            sample = sample[:max_bytes]
        ids = list(sample)

        merges: dict[tuple[int, int], int] = {}
        for i in range(num_merges):
            counts = _count_pairs(ids)
            if not counts:
                break
            max_count = max(counts.values())
            # deterministic tie-break: smallest pair among the most frequent
            best = min(p for p, c in counts.items() if c == max_count)
            new_id = BYTE_VOCAB + i
            merges[best] = new_id
            ids = _merge(ids, best, new_id)

        tok = cls(merges, mode="bpe")
        # assign special ids above the learned BPE vocabulary
        base = BYTE_VOCAB + len(merges)
        tok.special_tokens = {s: base + j for j, s in enumerate(specials)}
        tok._build_vocab()
        return tok

    @classmethod
    def bytes_tokenizer(
        cls, special_tokens: Iterable[str] | None = DEFAULT_SPECIALS
    ) -> "Tokenizer":
        """Vocab-256 fallback (no merges); used for the CPU smoke test."""
        tok = cls({}, mode="bytes")
        specials = list(special_tokens or [])
        tok.special_tokens = {s: BYTE_VOCAB + j for j, s in enumerate(specials)}
        tok._build_vocab()
        return tok

    # ------------------------------------------------------------------ #
    # size
    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return BYTE_VOCAB + len(self.merges) + len(self.special_tokens)

    # ------------------------------------------------------------------ #
    # encode / decode
    # ------------------------------------------------------------------ #
    def encode_bytes(self, data: bytes) -> list[int]:
        ids = list(data)
        if not self._ranks:
            return ids
        while len(ids) >= 2:
            # find the mergeable pair with the lowest rank present
            best_pair = None
            best_rank = None
            for pair in zip(ids, ids[1:]):
                r = self._ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pair = pair
            if best_pair is None:
                break
            ids = _merge(ids, best_pair, self.merges[best_pair])
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text with no special-token handling."""
        return self.encode_bytes(text.encode("utf-8"))

    def encode(self, text: str, allowed_special: str | set[str] = "none") -> list[int]:
        if allowed_special == "all":
            allowed = set(self.special_tokens)
        elif allowed_special == "none":
            allowed = set()
        elif isinstance(allowed_special, set):
            allowed = allowed_special
        else:
            raise ValueError("allowed_special must be 'all', 'none', or a set of names")

        if not allowed or self._special_re is None:
            return self.encode_ordinary(text)

        out: list[int] = []
        for chunk in self._special_re.split(text):
            if chunk in allowed:
                out.append(self.special_tokens[chunk])
            elif chunk:
                out.extend(self.encode_ordinary(chunk))
        return out

    def decode_bytes(self, ids: list[int]) -> bytes:
        pieces: list[bytes] = []
        for i in ids:
            if i in self._special_inv:
                pieces.append(self._special_inv[i].encode("utf-8"))
            else:
                pieces.append(self.vocab[i])
        return b"".join(pieces)

    def decode(self, ids: list[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # stats
    # ------------------------------------------------------------------ #
    def stats(self, text: str) -> dict[str, float]:
        n_bytes = len(text.encode("utf-8"))
        n_words = max(1, len(text.split()))
        n_tokens = len(self.encode_ordinary(text))
        return {
            "n_bytes": n_bytes,
            "n_tokens": n_tokens,
            "fertility": n_tokens / n_words,           # tokens per word
            "compression": n_bytes / max(1, n_tokens), # bytes per token
        }

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        data = {
            "mode": self.mode,
            "merges": [[a, b, i] for (a, b), i in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = {(a, b): i for a, b, i in data["merges"]}
        tok = cls(merges, special_tokens=data.get("special_tokens"), mode=data["mode"])
        return tok

    def content_hash(self) -> str:
        import hashlib

        payload = json.dumps(
            {
                "merges": [[a, b, i] for (a, b), i in self.merges.items()],
                "special_tokens": self.special_tokens,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

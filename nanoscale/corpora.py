"""Corpus adapters, with the revision and shard order pinned.

An unpinned corpus silently invalidates the study: HuggingFace datasets are mutable,
so "FineWeb-Edu" today and "FineWeb-Edu" in three weeks can differ, and two runs that
should be comparable would not be. Every corpus therefore carries an explicit
``revision`` and an explicit, sorted shard list, both recorded in the run metadata.

The adapter deliberately refuses to run against an unpinned revision rather than
defaulting to ``main``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml


@dataclass(frozen=True)
class CorpusSpec:
    """Everything needed to reproduce exactly which bytes were trained on."""

    name: str
    kind: str                      # "local" | "hf"
    revision: str | None = None    # HF commit sha; required for kind == "hf"
    repo_id: str | None = None
    config_name: str | None = None
    split: str = "train"
    text_field: str = "text"
    shards: tuple[str, ...] = field(default_factory=tuple)
    path: str | None = None        # local corpora
    max_documents: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("local", "hf"):
            raise ValueError(f"kind must be 'local' or 'hf'; got {self.kind!r}")
        if self.kind == "hf":
            if not self.repo_id:
                raise ValueError("hf corpus requires repo_id")
            if not self.revision or self.revision in ("main", "master", ""):
                raise ValueError(
                    f"corpus '{self.name}' must pin an immutable revision (a commit sha); "
                    f"got {self.revision!r}. An unpinned dataset makes runs incomparable."
                )
        if self.kind == "local" and not self.path:
            raise ValueError("local corpus requires path")

    @property
    def ordered_shards(self) -> tuple[str, ...]:
        """Shards in a deterministic order, so the token stream is reproducible."""
        return tuple(sorted(self.shards))

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "config_name": self.config_name,
            "split": self.split,
            "n_shards": len(self.ordered_shards),
            "shards": list(self.ordered_shards),
            "max_documents": self.max_documents,
        }

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CorpusSpec":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if "shards" in data and data["shards"] is not None:
            data["shards"] = tuple(data["shards"])
        return cls(**data)


def iter_corpus(spec: CorpusSpec) -> Iterator[str]:
    """Yield documents for a spec, one at a time."""
    if spec.kind == "local":
        from nanoscale.data import iter_documents

        yielded = 0
        for doc in iter_documents(spec.path, text_field=spec.text_field):
            yield doc
            yielded += 1
            if spec.max_documents and yielded >= spec.max_documents:
                return
        return

    # hf: stream shards in pinned order at the pinned revision
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "streaming an HF corpus needs the 'datasets' package: pip install datasets"
        ) from exc

    kwargs: dict[str, Any] = {
        "path": spec.repo_id,
        "split": spec.split,
        "revision": spec.revision,
        "streaming": True,
    }
    if spec.config_name:
        kwargs["name"] = spec.config_name
    if spec.ordered_shards:
        kwargs["data_files"] = list(spec.ordered_shards)

    ds = load_dataset(**kwargs)
    yielded = 0
    for row in ds:
        text = row.get(spec.text_field)
        if not text:
            continue
        yield text
        yielded += 1
        if spec.max_documents and yielded >= spec.max_documents:
            return


def write_shard_manifest(spec: CorpusSpec, out_dir: str | Path) -> Path:
    """Record the exact shard list used, next to the prepared data."""
    out = Path(out_dir) / "corpus_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec.metadata(), indent=2), encoding="utf-8")
    return out

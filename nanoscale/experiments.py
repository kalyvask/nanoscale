"""Experiment records.

Each run owns a directory ``experiments/runs/<run_id>/`` that is the authoritative
record:

    resolved_config.yaml   the exact config used
    environment.json       git sha + dirty, torch/CUDA/device, platform, deps
    metrics.jsonl          streamed per-step metrics
    summary.json           final record (status, losses, params, flops, timing)
    checkpoint.pt          optional

``experiments/manifest.jsonl`` holds one index line per completed/failed run pointing
at its directory. The manifest is a convenience index; the run directory is the truth.

Use :class:`RunRecord` as a context manager so an exception marks the run ``failed``
with its reason instead of leaving a dangling ``running`` record.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanoscale.config import Config


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git_info(repo_dir: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode != 0:
                return None
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    sha = run(["rev-parse", "HEAD"])
    status = run(["status", "--porcelain"])
    dirty = bool(status) if status is not None else None
    return {"git_sha": sha, "git_dirty": dirty}


def capture_environment(repo_dir: Path) -> dict[str, Any]:
    env: dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    env.update(_git_info(repo_dir))
    try:  # torch is a dependency but keep env capture robust in minimal installs
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - env capture must never crash a run
        env["torch"] = None
    return env


@dataclass
class RunRecord:
    """A live run directory. Prefer the ``with RunRecord.create(...) as run:`` form."""

    run_id: str
    dir: Path
    config: Config
    _summary: dict[str, Any]
    _manifest_path: Path
    _start_time: float

    # ------------------------------------------------------------------ #
    @classmethod
    def create(
        cls,
        config: Config,
        base_dir: str | Path = "experiments",
        repo_dir: str | Path | None = None,
        extra_summary: dict[str, Any] | None = None,
    ) -> "RunRecord":
        base = Path(base_dir)
        runs = base / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = f"{config.name}_{stamp}_{uuid.uuid4().hex[:6]}"
        run_dir = runs / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        repo = Path(repo_dir) if repo_dir else base.resolve().parent
        config.save_yaml(run_dir / "resolved_config.yaml")
        env = capture_environment(repo)
        (run_dir / "environment.json").write_text(
            json.dumps(env, indent=2), encoding="utf-8"
        )
        # start metrics stream
        (run_dir / "metrics.jsonl").write_text("", encoding="utf-8")

        summary: dict[str, Any] = {
            "run_id": run_id,
            "status": "running",
            "timestamp": env["timestamp"],
            "group": config.group,
            "config": config.to_dict(),
            "git_sha": env.get("git_sha"),
            "git_dirty": env.get("git_dirty"),
            "seed": config.seed,
            "device": None,
            "torch": env.get("torch"),
            "cuda": env.get("cuda"),
            "n_params": config.n_params(),
            "n_params_non_embedding": config.n_params_non_embedding(),
            "flops_per_token": config.flops_per_token(),
            "planned_tokens": config.total_tokens(),
            "planned_steps": config.derived_max_steps(),
            "dataset_hash": None,
            "tokenizer_hash": None,
            "tokens_seen": None,
            "final_val_loss": None,
            "bits_per_byte": None,
            "wall_clock_s": None,
            "tokens_per_sec": None,
            "peak_memory_bytes": None,
            "failure_reason": None,
        }
        if extra_summary:
            summary.update(extra_summary)

        rec = cls(
            run_id=run_id,
            dir=run_dir,
            config=config,
            _summary=summary,
            _manifest_path=base / "manifest.jsonl",
            _start_time=time.time(),
        )
        rec._write_summary()
        return rec

    # ------------------------------------------------------------------ #
    def log_metrics(self, metrics: dict[str, Any]) -> None:
        line = json.dumps(metrics)
        with open(self.dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def update_summary(self, **fields: Any) -> None:
        self._summary.update(fields)
        self._write_summary()

    def _write_summary(self) -> None:
        (self.dir / "summary.json").write_text(
            json.dumps(self._summary, indent=2), encoding="utf-8"
        )

    def _append_manifest(self) -> None:
        index = {
            "run_id": self.run_id,
            "dir": str(self.dir.as_posix()),
            "status": self._summary["status"],
            "group": self._summary.get("group"),
            "name": self.config.name,
            "n_params": self._summary.get("n_params"),
            "final_val_loss": self._summary.get("final_val_loss"),
            "bits_per_byte": self._summary.get("bits_per_byte"),
            "timestamp": self._summary.get("timestamp"),
        }
        with open(self._manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(index) + "\n")

    def finish(self, **fields: Any) -> None:
        self._summary.update(fields)
        self._summary["status"] = "completed"
        self._summary["wall_clock_s"] = round(time.time() - self._start_time, 3)
        self._write_summary()
        self._append_manifest()

    def fail(self, reason: str, **fields: Any) -> None:
        self._summary.update(fields)
        self._summary["status"] = "failed"
        self._summary["failure_reason"] = reason
        self._summary["wall_clock_s"] = round(time.time() - self._start_time, 3)
        self._write_summary()
        self._append_manifest()

    # ------------------------------------------------------------------ #
    # context manager: a leaking exception fails the run
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "RunRecord":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            if self._summary["status"] == "running":
                self.fail(f"{exc_type.__name__}: {exc}")
            return False  # re-raise
        if self._summary["status"] == "running":
            self.finish()
        return False


def read_manifest(base_dir: str | Path = "experiments") -> list[dict[str, Any]]:
    path = Path(base_dir) / "manifest.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def read_summary(run_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))

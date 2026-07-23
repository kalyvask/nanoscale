"""Run nanoscale on Modal.

Two jobs live here:

* ``prepare`` (CPU): stream the pinned corpus, tokenize it with the frozen tokenizer,
  and write the memmaps into a persistent Volume. Done once, sized for the LARGEST
  tier that will ever run, because PackedStream permutes over the block count and
  appending data later would break the nested-prefix property.
* ``train_batch`` (GPU): run several configs inside a single container. Batching
  matters here: an S-tier run is a couple of minutes, so one container per run would
  spend a large fraction of the budget on cold starts.

Everything is dry-run by default. ``--execute`` is required to spend anything.

    modal run scripts/modal_run.py --action prepare --max-tokens 2000000000
    python scripts/modal_run.py --action plan            # local dry run, no Modal
    modal run scripts/modal_run.py --action pilot --execute
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import modal
except ImportError:  # allows --action plan to work without modal installed
    modal = None

APP_NAME = "nanoscale"
VOLUME_NAME = "nanoscale-vol"
VOL_MOUNT = "/vol"
GPU_TYPE = os.environ.get("NANOSCALE_GPU", "H100")

IGNORE = [".venv", "__pycache__", ".git", "experiments", "data", "analysis",
          "node_modules", ".pytest_cache"]


def _ignore(path) -> bool:
    return any(p in IGNORE for p in str(path).split("/"))


if modal is not None:
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11"
        )
        .pip_install("torch", "numpy", "pyyaml", "requests", "datasets")
        .add_local_dir(".", "/root/nanoscale", ignore=_ignore, copy=True)
        .run_commands("pip install -e /root/nanoscale")
    )
    app = modal.App(APP_NAME)

    @app.function(image=image, volumes={VOL_MOUNT: volume}, timeout=60 * 60 * 8,
                  cpu=16.0)
    def prepare_remote(corpus_yaml: str, tokenizer_json: str | None,
                       max_tokens: int, dataset_name: str = "fineweb_edu") -> dict:
        """Tokenize the pinned corpus into the Volume. CPU only."""
        os.chdir("/root/nanoscale")
        from nanoscale.corpora import CorpusSpec, iter_corpus, write_shard_manifest
        from nanoscale.data import prepare_streaming
        from nanoscale.tokenizer import Tokenizer

        spec = CorpusSpec.from_yaml(corpus_yaml)
        out_dir = Path(VOL_MOUNT) / "data" / dataset_name
        tok = (Tokenizer.load(tokenizer_json) if tokenizer_json
               else Tokenizer.bytes_tokenizer())

        # bound ingestion by an approximate token budget
        approx_bytes_per_token = 4
        budget_bytes = max_tokens * approx_bytes_per_token

        def bounded():
            total = 0
            for doc in iter_corpus(spec):
                yield doc
                total += len(doc.encode("utf-8"))
                if total >= budget_bytes:
                    return

        meta = prepare_streaming(bounded(), tok, out_dir,
                                 dataset_name=dataset_name,
                                 corpus_meta=spec.metadata())
        write_shard_manifest(spec, out_dir)
        volume.commit()
        return meta

    @app.function(image=image, gpu=GPU_TYPE, volumes={VOL_MOUNT: volume},
                  timeout=60 * 60 * 6)
    def train_batch(config_dicts: list[dict], dataset_name: str,
                    protocol_hash: str | None = None) -> list[dict]:
        """Train several configs in one container, so cold start is amortized."""
        os.chdir("/root/nanoscale")
        from nanoscale.config import Config
        from nanoscale.train import train

        data_dir = Path(VOL_MOUNT) / "data" / dataset_name
        base_dir = Path(VOL_MOUNT) / "experiments"
        out = []
        for d in config_dicts:
            cfg = Config.from_dict(d)
            print(f"=== {cfg.name} ===", flush=True)
            summary = train(cfg, base_dir=base_dir, data_dir=data_dir,
                            protocol_hash=protocol_hash)
            out.append({k: summary.get(k) for k in
                        ("run_id", "status", "scale_id", "recipe_id", "init_seed",
                         "final_val_loss", "bits_per_byte", "tokens_per_sec",
                         "measured_tflops", "peak_memory_bytes")})
            volume.commit()
        return out

    @app.function(image=image, volumes={VOL_MOUNT: volume}, timeout=60 * 30)
    def fetch_results() -> list[dict]:
        os.chdir("/root/nanoscale")
        from nanoscale.analysis import load_runs

        return load_runs(Path(VOL_MOUNT) / "experiments")


# ---------------------------------------------------------------------- #
# planning (works without modal installed)
# ---------------------------------------------------------------------- #
def build_pending(pilot: bool, base_config: str, batch_size: int):
    from nanoscale.config import load_config
    from nanoscale.planning import build_grid, protocol_hash, variance_pilot

    base = load_config(base_config)
    runs = variance_pilot(base) if pilot else build_grid(base)
    phash = protocol_hash(base)
    batches = [runs[i:i + batch_size] for i in range(0, len(runs), batch_size)]
    return runs, batches, phash


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--action", default="plan",
                    choices=["plan", "prepare", "pilot", "study", "results"])
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--corpus", default="configs/corpora/fineweb_edu.yaml")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--dataset-name", default="fineweb_edu")
    ap.add_argument("--max-tokens", type=int, default=2_000_000_000)
    ap.add_argument("--batch-size", type=int, default=7,
                    help="runs per container; amortizes cold start")
    ap.add_argument("--execute", action="store_true",
                    help="actually spend compute (otherwise plan only)")
    args = ap.parse_args()

    if args.action in ("plan", "pilot", "study"):
        pilot = args.action == "pilot"
        runs, batches, phash = build_pending(pilot, args.config, args.batch_size)
        print(f"protocol_hash: {phash}")
        print(f"{len(runs)} runs in {len(batches)} container batches "
              f"(batch size {args.batch_size}, gpu {GPU_TYPE})")
        for i, b in enumerate(batches, 1):
            names = ", ".join(r.config.name for r in b)
            print(f"  batch {i}: {names}")
        if not args.execute or args.action == "plan":
            print("\nPLAN ONLY: nothing dispatched. Re-run with "
                  "`modal run scripts/modal_run.py --action pilot --execute`.")
            return
        if modal is None:
            raise SystemExit("modal is not installed")
        results = []
        for b in batches:
            results.extend(train_batch.remote(
                [r.config.to_dict() for r in b], args.dataset_name, phash))
        print(json.dumps(results, indent=2))
        return

    if modal is None:
        raise SystemExit("modal is not installed; `pip install modal`")

    if args.action == "prepare":
        if not args.execute:
            print(f"PLAN ONLY: would tokenize up to {args.max_tokens:,} tokens from "
                  f"{args.corpus} into volume '{VOLUME_NAME}'. Pass --execute.")
            return
        meta = prepare_remote.remote(args.corpus, args.tokenizer,
                                     args.max_tokens, args.dataset_name)
        print(json.dumps(meta, indent=2))
        return

    if args.action == "results":
        rows = fetch_results.remote()
        print(f"{len(rows)} completed runs in the volume")
        for r in rows[:20]:
            print(f"  {r.get('scale_id')}/{r.get('recipe_id')}/"
                  f"seed{r.get('init_seed')}: {r.get('final_val_loss')}")


if __name__ == "__main__":
    main()

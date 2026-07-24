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
    # normalize Windows backslashes so .git/.venv/etc are actually excluded; otherwise
    # the whole .git and .venv trees get uploaded and a concurrent git op can corrupt
    # the build snapshot ("modified during build process")
    parts = str(path).replace("\\", "/").split("/")
    return any(p in IGNORE for p in parts)


if modal is not None:
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.11"
        )
        .pip_install("torch", "numpy", "pyyaml", "requests", "datasets")
        .add_local_dir(".", "/root/repo", ignore=_ignore, copy=True)
        .run_commands("pip install -e /root/repo")
    )
    app = modal.App(APP_NAME)

    @app.function(image=image, volumes={VOL_MOUNT: volume}, timeout=60 * 60 * 8,
                  cpu=16.0)
    def prepare_remote(corpus_yaml: str, tokenizer_json: str | None,
                       max_tokens: int, dataset_name: str = "fineweb_edu") -> dict:
        """Tokenize the pinned corpus into the Volume. CPU only."""
        os.chdir("/root/repo")
        from nanoscale.corpora import CorpusSpec, iter_corpus, write_shard_manifest
        from nanoscale.data import prepare_streaming
        from nanoscale.tokenizer import Tokenizer

        spec = CorpusSpec.from_yaml(corpus_yaml)
        out_dir = Path(VOL_MOUNT) / "data" / dataset_name
        tok = (Tokenizer.load(tokenizer_json) if tokenizer_json
               else Tokenizer.bytes_tokenizer())

        # bound ingestion by an approximate token budget (compression ~4.25 bytes/token)
        approx_bytes_per_token = 4.25
        budget_bytes = int(max_tokens * approx_bytes_per_token)

        import time as _time

        def bounded():
            total = docs = 0
            next_report = 250_000_000
            t0 = _time.time()
            for doc in iter_corpus(spec):
                yield doc
                total += len(doc.encode("utf-8"))
                docs += 1
                if total >= next_report:
                    rate = total / max(_time.time() - t0, 1e-9) / 1e6
                    print(f"  ingested {total/1e9:.2f} GB, {docs:,} docs, "
                          f"{rate:.2f} MB/s", flush=True)
                    next_report += 250_000_000
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
        os.chdir("/root/repo")
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
        os.chdir("/root/repo")
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


CALLS_DIR = Path("experiments/.modal_calls")


def _deployed(fn_name: str):
    """Look up a function from the DEPLOYED app, so calls survive client disconnects.

    Deploy first with: modal deploy scripts/modal_run.py
    """
    return modal.Function.from_name(APP_NAME, fn_name)


def _save_calls(tag: str, ids: list[str]) -> Path:
    CALLS_DIR.mkdir(parents=True, exist_ok=True)
    p = CALLS_DIR / f"{tag}.json"
    p.write_text(json.dumps(ids), encoding="utf-8")
    return p


def _load_calls(tag: str) -> list[str]:
    p = CALLS_DIR / f"{tag}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def _poll(tag: str) -> None:
    ids = _load_calls(tag)
    if not ids:
        print(f"no saved calls for '{tag}'")
        return
    done, running, results = 0, 0, []
    for cid in ids:
        fc = modal.FunctionCall.from_id(cid)
        try:
            res = fc.get(timeout=0)
            done += 1
            results.extend(res if isinstance(res, list) else [res])
        except TimeoutError:
            running += 1
        except Exception as exc:  # noqa: BLE001 - surface a failed call, keep polling
            print(f"  call {cid[:12]} failed: {exc}")
            done += 1
    print(f"'{tag}': {done}/{len(ids)} calls finished, {running} still running")
    if results:
        print(json.dumps(results, indent=2))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--action", default="plan",
                    choices=["plan", "prepare", "pilot", "study", "results",
                             "gpu-smoke", "poll", "probe"])
    ap.add_argument("--probe-steps", type=int, default=500,
                    help="steps per scale for the throughput probe")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--corpus", default="configs/corpora/fineweb_edu.yaml")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--dataset-name", default="fineweb_edu")
    ap.add_argument("--max-tokens", type=int, default=2_000_000_000)
    ap.add_argument("--batch-size", type=int, default=7,
                    help="runs per container; amortizes cold start")
    ap.add_argument("--poll-tag", default=None, help="which spawned job to poll")
    ap.add_argument("--execute", action="store_true",
                    help="actually spend compute (otherwise plan only)")
    args = ap.parse_args()

    # --- planning (no modal needed) ---
    if args.action in ("plan", "pilot", "study") and not args.execute:
        pilot = args.action == "pilot"
        runs, batches, phash = build_pending(pilot, args.config, args.batch_size)
        print(f"protocol_hash: {phash}")
        print(f"{len(runs)} runs in {len(batches)} container batches "
              f"(batch size {args.batch_size}, gpu {GPU_TYPE})")
        for i, b in enumerate(batches, 1):
            print(f"  batch {i}: " + ", ".join(r.config.name for r in b))
        print("\nPLAN ONLY: nothing dispatched. Re-run with --execute.")
        return

    if modal is None:
        raise SystemExit("modal is not installed; `pip install modal`")

    # --- detached execution against the DEPLOYED app ---
    if args.action in ("pilot", "study"):
        pilot = args.action == "pilot"
        runs, batches, phash = build_pending(pilot, args.config, args.batch_size)
        fn = _deployed("train_batch")
        ids = []
        for b in batches:
            call = fn.spawn([r.config.to_dict() for r in b], args.dataset_name, phash)
            ids.append(call.object_id)
            print(f"  spawned {call.object_id}: " + ", ".join(r.config.name for r in b))
        tag = "pilot" if pilot else "study"
        path = _save_calls(tag, ids)
        print(f"\n{len(ids)} batches spawned detached. Poll with:\n"
              f"  python scripts/modal_run.py --action poll --poll-tag {tag}\n"
              f"(call ids saved to {path})")
        return

    if args.action == "prepare":
        call = _deployed("prepare_remote").spawn(
            args.corpus, args.tokenizer, args.max_tokens, args.dataset_name)
        _save_calls("prepare", [call.object_id])
        print(f"prepare spawned detached: {call.object_id}\n"
              f"runs on Modal independent of this machine. Poll with:\n"
              f"  python scripts/modal_run.py --action poll --poll-tag prepare")
        return

    if args.action == "gpu-smoke":
        from nanoscale.config import load_config

        cfg = load_config(args.config).override(
            name="gpu_smoke", study_id="gpu_smoke", scale_id="S", recipe_id="baseline",
            n_layer=2, n_embd=128, n_head=4, batch_size=16, max_steps=20,
            target_train_tokens=None, eval_interval=10, eval_iters=5, save_checkpoint=False,
        )
        call = _deployed("train_batch").spawn([cfg.to_dict()], args.dataset_name,
                                              "gpu_smoke")
        _save_calls("gpu_smoke", [call.object_id])
        print(f"gpu-smoke spawned: {call.object_id}. Poll with --action poll "
              f"--poll-tag gpu_smoke")
        return

    if args.action == "probe":
        # Measure real per-scale throughput on a few hundred steps, so the study can be
        # priced from measured tokens/sec instead of an optimistic FLOP estimate. Small
        # models underutilize a big GPU, and this is where that shows up.
        from nanoscale.config import load_config
        from nanoscale.planning import SIZES

        base = load_config(args.config)
        cfgs = []
        for scale, geo in SIZES.items():
            cfgs.append(base.override(
                **geo, name=f"probe-{scale}", study_id="probe", scale_id=scale,
                recipe_id="baseline", target_train_tokens=None,
                max_steps=args.probe_steps, eval_interval=max(1, args.probe_steps // 2),
                eval_iters=20, save_checkpoint=False,
            ).to_dict())
        call = _deployed("train_batch").spawn(cfgs, args.dataset_name, "probe")
        _save_calls("probe", [call.object_id])
        print(f"probe spawned: {call.object_id}. Poll with --action poll --poll-tag probe")
        return

    if args.action == "poll":
        _poll(args.poll_tag or "prepare")
        return

    if args.action == "results":
        call = _deployed("fetch_results").spawn()
        rows = call.get()
        print(f"{len(rows)} completed runs in the volume")
        for r in rows[:30]:
            print(f"  {r.get('scale_id')}/{r.get('recipe_id')}/"
                  f"seed{r.get('init_seed')}: {r.get('final_val_loss')}")


if __name__ == "__main__":
    main()

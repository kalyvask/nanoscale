"""Run the transfer study: 7 recipes x 3 scales x 3 seeds = 63 runs.

Resumable by construction. A run is identified by
(study_id, scale_id, recipe_id, init_seed, data_seed); any identity already marked
completed in the manifest is skipped, so re-invoking after a crash, a preemption or a
budget pause continues where it stopped rather than repeating work.

Dry run is the default. It prints the full matrix and the estimated cost and spends
nothing; ``--execute`` is required to train.

    python scripts/run_study.py --dry-run --gpu h100
    python scripts/run_study.py --pilot --dry-run
    python scripts/run_study.py --execute --data-dir data/fineweb_edu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nanoscale.config import Config, load_config
from nanoscale.experiments import completed_identities, run_identity
from nanoscale.planning import (
    GPU_PRESETS,
    RECIPES,
    SEEDS,
    SIZES,
    build_grid,
    protocol_hash,
    summarize,
    variance_pilot,
)


def print_matrix(runs, effective_tflops: float, usd_per_hour: float,
                 done: set[tuple], phash: str) -> dict:
    print(f"protocol_hash: {phash}")
    print(f"{'#':>3} {'scale':>5} {'recipe':>11} {'seed':>4} {'params':>12} "
          f"{'tokens':>14} {'steps':>8} {'gpu_h':>7} {'status':>9}")
    print("-" * 92)
    pending = 0
    for i, r in enumerate(runs, 1):
        c = r.config
        ident = run_identity({
            "study_id": c.study_id, "scale_id": c.scale_id, "recipe_id": c.recipe_id,
            "init_seed": c.resolved_init_seed, "data_seed": c.resolved_data_seed,
        })
        state = "done" if ident in done else "pending"
        pending += state == "pending"
        print(f"{i:>3} {r.size:>5} {r.recipe:>11} {r.seed_index:>4} "
              f"{c.n_params():>12,} {c.total_tokens():>14,} "
              f"{c.derived_max_steps():>8,} "
              f"{r.estimated_gpu_hours(effective_tflops):>7.2f} {state:>9}")
    summary = summarize(runs, effective_tflops, usd_per_hour)
    print("-" * 92)
    print(f"{len(runs)} runs total, {pending} pending, {len(runs) - pending} already done")
    for scale, e in summary["by_size"].items():
        print(f"  {scale}: {e['runs']:>3} runs  {e['gpu_hours']:>7.2f} gpu_h  "
              f"${e['usd']:>8.2f}")
    print(f"  TOTAL estimated: {summary['total_gpu_hours']:.2f} gpu_h  "
          f"${summary['total_usd']:.2f}")
    print("  (estimate from FLOP accounting; small scales will underperform it)")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--base-dir", default="experiments")
    ap.add_argument("--gpu", default="h100", choices=sorted(GPU_PRESETS))
    ap.add_argument("--tflops", type=float, default=None,
                    help="measured effective TFLOP/s; overrides the preset")
    ap.add_argument("--usd-per-hour", type=float, default=None)
    ap.add_argument("--pilot", action="store_true",
                    help="the five-seed S power pilot instead of the full grid")
    ap.add_argument("--pilot-seeds", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--execute", dest="dry_run", action="store_false",
                    help="actually train (spends compute)")
    ap.add_argument("--limit", type=int, default=None,
                    help="execute at most N pending runs then stop")
    args = ap.parse_args(argv)

    preset_usd, preset_tflops = GPU_PRESETS[args.gpu]
    usd_per_hour = args.usd_per_hour if args.usd_per_hour is not None else preset_usd
    tflops = args.tflops if args.tflops is not None else preset_tflops

    base = load_config(args.config) if args.config else Config()
    runs = (variance_pilot(base, n_seeds=args.pilot_seeds) if args.pilot
            else build_grid(base))
    phash = protocol_hash(base)

    title = "POWER PILOT" if args.pilot else "TRANSFER STUDY"
    print(f"=== {title} ===")
    if not args.pilot:
        print(f"grid: {len(RECIPES)} recipes x {len(SIZES)} scales x seeds {SEEDS}")
    done = completed_identities(args.base_dir)
    print_matrix(runs, tflops, usd_per_hour, done, phash)

    if args.dry_run:
        print("\nDRY RUN: nothing was executed. Pass --execute to train.")
        return

    from nanoscale.train import train

    executed = 0
    for r in runs:
        c = r.config
        ident = (c.study_id, c.scale_id, c.recipe_id,
                 c.resolved_init_seed, c.resolved_data_seed)
        if ident in done:
            print(f"skip (done): {c.name}")
            continue
        if args.limit is not None and executed >= args.limit:
            print(f"reached --limit {args.limit}; stopping")
            break
        print(f"\n=== {c.name} ({r.size}/{r.recipe}/seed {r.seed_index}) ===")
        summary = train(c, base_dir=args.base_dir, data_dir=args.data_dir,
                        protocol_hash=phash)
        executed += 1
        print(f"    val_loss={summary['final_val_loss']:.4f} "
              f"bpb={summary['bits_per_byte']:.4f} "
              f"tok/s={summary['tokens_per_sec']:,.0f}")
    print(f"\nexecuted {executed} run(s).")


if __name__ == "__main__":
    main()

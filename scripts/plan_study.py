"""Price the transfer study before spending anything.

    python scripts/plan_study.py --gpu a100 --budget 30
    python scripts/plan_study.py --gpu h100 --budget 30 --tflops 350

Cost comes from the FLOP accounting in config.py, so it is an estimate, not a quote.
It is optimistic for the small tiers: a 15M model at 512 context will not reach the
assumed utilization on a large GPU. Run the calibration pass first and feed the
measured throughput back in with --tflops.
"""

from __future__ import annotations

import argparse

from nanoscale.config import Config, load_config
from nanoscale.planning import (
    GPU_PRESETS,
    SEEDS,
    SIZES,
    build_grid,
    largest_affordable_plan,
    summarize,
    variance_pilot,
)


def _print_summary(title: str, summary: dict) -> None:
    print(f"\n{title}")
    print(f"{'size':5} {'runs':>5} {'params':>12} {'tokens/run':>15} {'gpu_h':>9} {'usd':>9}")
    print("-" * 60)
    for size, e in summary["by_size"].items():
        print(f"{size:5} {e['runs']:>5} {e['params']:>12,} {e['tokens_per_run']:>15,} "
              f"{e['gpu_hours']:>9.2f} {e['usd']:>9.2f}")
    print("-" * 60)
    print(f"{'TOTAL':5} {summary['total_runs']:>5} {'':>12} {'':>15} "
          f"{summary['total_gpu_hours']:>9.2f} {summary['total_usd']:>9.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--gpu", default="a100", choices=sorted(GPU_PRESETS))
    ap.add_argument("--usd-per-hour", type=float, default=None)
    ap.add_argument("--tflops", type=float, default=None,
                    help="measured effective bf16 TFLOP/s (overrides the preset)")
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--reserve-frac", type=float, default=0.25,
                    help="fraction of budget held back for reruns and calibration")
    args = ap.parse_args()

    preset_usd, preset_tflops = GPU_PRESETS[args.gpu]
    usd_per_hour = args.usd_per_hour if args.usd_per_hour is not None else preset_usd
    tflops = args.tflops if args.tflops is not None else preset_tflops

    base = load_config(args.config) if args.config else Config()
    print(f"GPU preset '{args.gpu}': ${usd_per_hour:.2f}/h, assumed {tflops:.0f} "
          f"effective TFLOP/s  (estimates, not a quote)")
    print(f"Seed plan: {SEEDS};  sizes: {list(SIZES)}")

    pilot = variance_pilot(base)
    _print_summary("VARIANCE PILOT (baseline only, 3 seeds at S) -- run this first",
                   summarize(pilot, tflops, usd_per_hour))

    full = build_grid(base)
    full_summary = summarize(full, tflops, usd_per_hour)
    _print_summary("FULL GRID (7 recipes x sizes x seeds)", full_summary)

    if args.budget is not None:
        print(f"\nBudget: ${args.budget:.2f}")
        if full_summary["total_usd"] <= args.budget:
            print("  the full grid fits.")
        else:
            print(f"  the full grid does NOT fit "
                  f"(${full_summary['total_usd']:.2f} > ${args.budget:.2f}).")
            chosen, sub = largest_affordable_plan(
                base, args.budget, tflops, usd_per_hour, args.reserve_frac
            )
            if not chosen:
                print("  no whole size tier fits; reduce seeds, recipes, or D/N.")
            else:
                print(f"  largest affordable tier set (holding back "
                      f"{args.reserve_frac:.0%} for reruns): {'+'.join(chosen)}")
                _print_summary(f"AFFORDABLE PLAN ({'+'.join(chosen)})", sub)
                missing = [s for s in SIZES if s not in chosen]
                if missing:
                    print(f"\n  Dropped: {', '.join(missing)}. Without the largest tier "
                          f"the study measures transfer across a smaller scale gap, "
                          f"which is a weaker claim, not the headline one.")


if __name__ == "__main__":
    main()

"""Run the baseline + one-flip ablation grid.

The baseline is the full-stack recipe (all modern components on). Each variant flips
exactly one component off, so its result is that component's *conditional* effect given
the rest of the baseline (not a universal independent contribution). Variants are
grouped by what they change: quality, stability, or efficiency.

On CPU this is plumbing only. TinyShakespeare numbers are smoke-test output, not
research findings; that is stamped on the table by ``make_table.py``.

Example::

    python scripts/prepare_data.py --dataset tinyshakespeare
    python scripts/run_ablation.py --config configs/cpu_smoke.yaml --max_steps 60
"""

from __future__ import annotations

import argparse
import sys

from nanoscale.config import load_config
from nanoscale.planning import RECIPES
from nanoscale.train import train

# Single source of truth for the recipe grid lives in nanoscale.planning, so the local
# runner, the Modal runner, and the cost estimator cannot drift apart.
VARIANTS = RECIPES


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="Run the CPU ablation plumbing grid.")
    ap.add_argument("--config", default="configs/cpu_smoke.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--base-dir", default="experiments")
    args, overrides = ap.parse_known_args(argv)

    base = load_config(args.config, overrides)
    print(f"Ablation grid from baseline '{base.name}' "
          f"({base.n_params():,} params, {base.derived_max_steps()} steps/run)")

    results = []
    for variant, (group, flip) in VARIANTS.items():
        cfg = base.override(name=f"{base.name}-{variant}", group="ablation", **flip)
        print(f"\n=== {variant} [{group}] ===")
        summary = train(cfg, base_dir=args.base_dir, data_dir=args.data_dir)
        results.append((variant, group, summary))
        print(f"    val_loss={summary['final_val_loss']:.4f} "
              f"bpb={summary['bits_per_byte']:.4f} "
              f"tok/s={summary['tokens_per_sec']:,.0f}")

    print(f"\nDone: {len(results)} runs. Render with:  python scripts/make_table.py")


if __name__ == "__main__":
    main()

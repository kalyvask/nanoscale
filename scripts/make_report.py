"""Run the transfer analysis over completed runs and write the report.

    python scripts/make_report.py --study transfer --out analysis/transfer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nanoscale.analysis import DEFAULT_EQUIVALENCE_MARGIN, analyze, load_runs, save_json
from nanoscale.report import write_report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", default="experiments")
    ap.add_argument("--study", default=None, help="study_id filter (e.g. transfer, pilot)")
    ap.add_argument("--metric", default="final_val_loss")
    ap.add_argument("--margin", type=float, default=DEFAULT_EQUIVALENCE_MARGIN,
                    help="equivalence margin in nats/token")
    ap.add_argument("--out", default="analysis/transfer")
    args = ap.parse_args(argv)

    runs = load_runs(args.base_dir, study_id=args.study)
    if not runs:
        print(f"No completed runs found in {args.base_dir}"
              + (f" for study '{args.study}'" if args.study else ""))
        return

    result = analyze(runs, metric=args.metric, margin=args.margin)
    if not result["protocol"]["consistent"]:
        print("WARNING: runs span multiple protocols/eval sets; pooling is invalid.")

    paths = write_report(result, args.out)
    save_json(result, Path(args.out) / "analysis.json")
    print(f"{len(runs)} runs analyzed across scales {result['scales']}")
    for r in result["regret"]:
        print(f"  regret {r['small']}->{r['large']}: {r.get('regret')}")
    print(f"wrote {paths['markdown']} and {paths['html']}")


if __name__ == "__main__":
    main()

"""Render the ablation grid as a grouped table.

Reads the run records indexed in ``experiments/manifest.jsonl``, keeps the latest
run per name in the ``ablation`` group, classifies each by which single component it
flipped, and prints results grouped into quality / stability / efficiency with the
delta versus the baseline.

Every table is stamped as smoke-test plumbing when the runs are CPU smoke runs. These
numbers are NOT research findings; the transfer study (M7) produces those, with seeds
and uncertainty.
"""

from __future__ import annotations

import argparse

from nanoscale.experiments import read_manifest, read_summary

# baseline "on" value for each component; a run differs from baseline on exactly one.
BASELINE = {"pos": "rope", "norm": "rms", "activation": "swiglu",
            "qk_norm": True, "tie_weights": True}

LABELS = {
    ("pos", "learned"): ("quality", "RoPE -> learned pos"),
    ("norm", "layer"): ("quality", "RMSNorm -> LayerNorm"),
    ("activation", "gelu"): ("quality", "SwiGLU -> GeLU"),
    ("qk_norm", False): ("stability", "QK-norm off"),
    ("z_loss", 0.0): ("stability", "z-loss off"),
    ("tie_weights", False): ("efficiency", "untie embeddings"),
}
GROUP_ORDER = ["quality", "stability", "efficiency"]


def classify(config: dict) -> tuple[str, str]:
    """Return (group, label) for a run based on the one component it flipped."""
    if config.get("z_loss", 1) == 0:
        return LABELS[("z_loss", 0.0)]
    for key, on_value in BASELINE.items():
        if config.get(key) != on_value:
            return LABELS.get((key, config.get(key)), ("other", f"{key}={config.get(key)}"))
    return ("baseline", "baseline (full stack)")


def latest_ablation_runs(base_dir: str) -> list[dict]:
    rows = [r for r in read_manifest(base_dir) if r.get("group") == "ablation"]
    by_name: dict[str, dict] = {}
    for r in rows:  # manifest is append-order; last wins
        by_name[r["name"]] = r
    summaries = []
    for r in by_name.values():
        try:
            summaries.append(read_summary(r["dir"]))
        except FileNotFoundError:
            continue
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="experiments")
    args = ap.parse_args()

    summaries = latest_ablation_runs(args.base_dir)
    if not summaries:
        print("No ablation runs found. Run scripts/run_ablation.py first.")
        return

    rows = []
    baseline_loss = None
    for s in summaries:
        group, label = classify(s["config"])
        if group == "baseline":
            baseline_loss = s.get("final_val_loss")
        rows.append({"group": group, "label": label, "s": s})

    all_smoke = all(r["s"]["config"].get("group") == "ablation"
                    and r["s"]["config"].get("dataset") == "tinyshakespeare"
                    for r in rows)

    bar = "=" * 78
    print(bar)
    if all_smoke:
        print("SMOKE-TEST PLUMBING (TinyShakespeare, CPU) -- NOT RESEARCH FINDINGS")
    else:
        print("ABLATION RESULTS")
    print(bar)
    header = f"{'variant':22} {'val_loss':>9} {'d vs base':>10} {'bpb':>7} {'max_logit':>10} {'tok/s':>9} {'params':>10}"

    def fmt(r):
        s = r["s"]
        vl = s.get("final_val_loss")
        delta = (vl - baseline_loss) if (vl is not None and baseline_loss is not None) else None
        return (f"{r['label']:22} "
                f"{_n(vl, '9.4f')} "
                f"{_n(delta, '+10.4f')} "
                f"{_n(s.get('bits_per_byte'), '7.3f')} "
                f"{_n(s.get('max_logit'), '10.2f')} "
                f"{_n(s.get('tokens_per_sec'), '9,.0f')} "
                f"{_n(s.get('n_params'), '10,')}")

    # baseline first
    print(header)
    print("-" * 78)
    for r in rows:
        if r["group"] == "baseline":
            print(fmt(r))
    for group in GROUP_ORDER:
        group_rows = [r for r in rows if r["group"] == group]
        if not group_rows:
            continue
        print(f"\n[{group}]")
        for r in group_rows:
            print(fmt(r))
    print(bar)
    print("d vs base: change in fixed-budget val loss when the component is turned off.")
    print("One-at-a-time flips measure conditional effects, not independent contributions.")


def _n(value, spec: str) -> str:
    if value is None:
        return "n/a".rjust(7)
    try:
        return format(value, spec)
    except (ValueError, TypeError):
        return str(value)


if __name__ == "__main__":
    main()

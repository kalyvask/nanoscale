"""Transfer analysis: paired effects, selection regret, and scale classification.

Written before the runs exist, deliberately. Deciding how the numbers will be
interpreted after seeing them is how a study talks itself into a result.

Definitions used here:

* **Paired effect.** For a recipe at a scale, the per-seed difference against the
  baseline *with the same seed*. Pairing removes seed-level variation, which is the
  dominant noise source, so the effect is a mean of differences, not a difference of
  means.
* **Selection regret.** Choose the recipe that looks best at a small scale, then read
  its loss at a large scale: ``regret = loss(chosen) - min(loss over recipes)``.
  Zero means the cheap experiment chose correctly.
* **Selection probability.** How often the small-scale choice is the large-scale
  best, estimated by resampling seeds. A single regret number hides whether it was
  luck.
* **Equivalence-aware classification.** An effect is only "real" if its confidence
  interval excludes an equivalence margin (a region of practical equivalence). Effects
  that are small and uncertain are reported ``unresolved`` rather than as zero, so
  "we could not tell" is never silently upgraded to "no difference".

No scipy dependency: rank correlations and bootstrap intervals are implemented here.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCALE_ORDER = ["S", "M", "L"]
DEFAULT_EQUIVALENCE_MARGIN = 0.01  # nats/token; below this we call it practically equal


# ---------------------------------------------------------------------- #
# loading
# ---------------------------------------------------------------------- #
def load_runs(base_dir: str | Path = "experiments",
              study_id: str | None = None) -> list[dict[str, Any]]:
    """Completed runs from the manifest, optionally filtered to one study."""
    from nanoscale.experiments import read_manifest, read_summary

    rows = []
    for row in read_manifest(base_dir):
        if row.get("status") != "completed":
            continue
        if study_id is not None and row.get("study_id") != study_id:
            continue
        try:
            rows.append(read_summary(row["dir"]))
        except (FileNotFoundError, KeyError):
            continue
    return rows


def check_protocol_consistency(runs: list[dict]) -> dict[str, Any]:
    """Refuse to pool runs from different protocols or evaluation sets."""
    phashes = {r.get("protocol_hash") for r in runs}
    ehashes = {r.get("eval_set_hash") for r in runs}
    thashes = {r.get("tokenizer_hash") for r in runs}
    return {
        "protocol_hashes": sorted(h for h in phashes if h is not None),
        "eval_set_hashes": sorted(h for h in ehashes if h is not None),
        "tokenizer_hashes": sorted(h for h in thashes if h is not None),
        "consistent": len(phashes) <= 1 and len(ehashes) <= 1 and len(thashes) <= 1,
    }


def index_by_scale_recipe_seed(runs: list[dict], metric: str = "final_val_loss"):
    """{scale: {recipe: {seed: value}}}"""
    out: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        scale, recipe, seed = r.get("scale_id"), r.get("recipe_id"), r.get("init_seed")
        value = r.get(metric)
        if scale is None or recipe is None or seed is None or value is None:
            continue
        out[scale][recipe][int(seed)] = float(value)
    return {s: {k: dict(v) for k, v in rec.items()} for s, rec in out.items()}


# ---------------------------------------------------------------------- #
# statistics (no scipy)
# ---------------------------------------------------------------------- #
def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks, handling ties."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # average tied ranks
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(a: Iterable[float], b: Iterable[float]) -> float:
    a, b = np.asarray(list(a), dtype=float), np.asarray(list(b), dtype=float)
    if len(a) < 2:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def kendall_tau(a: Iterable[float], b: Iterable[float]) -> float:
    """Kendall tau-b."""
    a, b = np.asarray(list(a), dtype=float), np.asarray(list(b), dtype=float)
    n = len(a)
    if n < 2:
        return float("nan")
    conc = disc = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            s = da * db
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
            else:
                if da == 0:
                    ta += 1
                if db == 0:
                    tb += 1
    n0 = n * (n - 1) / 2
    denom = math.sqrt((n0 - ta) * (n0 - tb))
    return float((conc - disc) / denom) if denom > 0 else float("nan")


def bootstrap_ci(values: Iterable[float], n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    v = np.asarray(list(values), dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    if len(v) == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


# ---------------------------------------------------------------------- #
# paired effects
# ---------------------------------------------------------------------- #
def paired_effects(index: dict, scale: str, baseline: str = "baseline",
                   margin: float = DEFAULT_EQUIVALENCE_MARGIN,
                   seed_for_boot: int = 0) -> dict[str, dict]:
    """Per-recipe paired difference vs baseline at one scale.

    Positive means the recipe is *worse* than baseline (higher loss), i.e. the removed
    component was helping.
    """
    at_scale = index.get(scale, {})
    base_by_seed = at_scale.get(baseline, {})
    out: dict[str, dict] = {}
    for recipe, by_seed in at_scale.items():
        if recipe == baseline:
            continue
        shared = sorted(set(by_seed) & set(base_by_seed))
        deltas = [by_seed[s] - base_by_seed[s] for s in shared]
        if not deltas:
            continue
        mean = float(np.mean(deltas))
        lo, hi = bootstrap_ci(deltas, seed=seed_for_boot)
        out[recipe] = {
            "n_pairs": len(deltas),
            "deltas": deltas,
            "mean_delta": mean,
            "ci_low": lo,
            "ci_high": hi,
            "verdict": classify_effect(mean, lo, hi, margin),
        }
    return out


def classify_effect(mean: float, lo: float, hi: float, margin: float) -> str:
    """Equivalence-aware verdict for a single effect."""
    if lo > margin:
        return "hurts_to_remove"      # removing the component costs quality
    if hi < -margin:
        return "helps_to_remove"      # the component was actively harmful
    if lo > -margin and hi < margin:
        return "practically_equal"    # CI inside the equivalence region
    return "unresolved"               # too wide to call either way


def seed_noise(index: dict, scale: str, baseline: str = "baseline") -> dict[str, float]:
    """Spread of the baseline across seeds: the floor any effect must clear."""
    vals = list(index.get(scale, {}).get(baseline, {}).values())
    if len(vals) < 2:
        return {"n": len(vals), "sd": float("nan"), "range": float("nan")}
    return {
        "n": len(vals),
        "sd": float(np.std(vals, ddof=1)),
        "range": float(max(vals) - min(vals)),
        "mean": float(np.mean(vals)),
    }


# ---------------------------------------------------------------------- #
# transfer: regret, selection probability, rank correlation
# ---------------------------------------------------------------------- #
def _mean_by_recipe(index: dict, scale: str) -> dict[str, float]:
    return {r: float(np.mean(list(v.values())))
            for r, v in index.get(scale, {}).items() if v}


def selection_regret(index: dict, small: str, large: str) -> dict[str, Any]:
    """Cost of choosing the recipe that looked best at ``small``, judged at ``large``."""
    small_means, large_means = _mean_by_recipe(index, small), _mean_by_recipe(index, large)
    shared = sorted(set(small_means) & set(large_means))
    if not shared:
        return {"small": small, "large": large, "regret": float("nan"), "n_recipes": 0}
    chosen = min(shared, key=lambda r: small_means[r])
    best_large = min(shared, key=lambda r: large_means[r])
    return {
        "small": small,
        "large": large,
        "n_recipes": len(shared),
        "chosen_at_small": chosen,
        "best_at_large": best_large,
        "chosen_large_loss": large_means[chosen],
        "best_large_loss": large_means[best_large],
        "regret": large_means[chosen] - large_means[best_large],
        "correct_selection": chosen == best_large,
    }


def selection_probability(index: dict, small: str, large: str, n_boot: int = 2000,
                          seed: int = 0) -> dict[str, Any]:
    """Resample seeds to ask how often the small-scale pick is the large-scale best.

    One regret number cannot distinguish a reliable proxy from a lucky draw.
    """
    rng = np.random.default_rng(seed)
    small_runs, large_runs = index.get(small, {}), index.get(large, {})
    shared = sorted(set(small_runs) & set(large_runs))
    if len(shared) < 2:
        return {"small": small, "large": large, "p_correct": float("nan"), "n_boot": 0}

    correct = 0
    regrets = []
    for _ in range(n_boot):
        small_m, large_m = {}, {}
        for r in shared:
            sv = list(small_runs[r].values())
            lv = list(large_runs[r].values())
            small_m[r] = float(np.mean(rng.choice(sv, size=len(sv), replace=True)))
            large_m[r] = float(np.mean(rng.choice(lv, size=len(lv), replace=True)))
        chosen = min(shared, key=lambda r: small_m[r])
        best = min(shared, key=lambda r: large_m[r])
        correct += chosen == best
        regrets.append(large_m[chosen] - large_m[best])
    return {
        "small": small,
        "large": large,
        "n_boot": n_boot,
        "p_correct": correct / n_boot,
        "mean_regret": float(np.mean(regrets)),
        "regret_ci": (float(np.quantile(regrets, 0.025)),
                      float(np.quantile(regrets, 0.975))),
    }


def rank_transfer(index: dict, small: str, large: str) -> dict[str, Any]:
    """Descriptive rank agreement between two scales.

    Descriptive on purpose: with only a handful of recipes these coefficients are
    badly underpowered, so they illustrate the ordering rather than test it.
    """
    small_means, large_means = _mean_by_recipe(index, small), _mean_by_recipe(index, large)
    shared = sorted(set(small_means) & set(large_means))
    a = [small_means[r] for r in shared]
    b = [large_means[r] for r in shared]
    return {
        "small": small,
        "large": large,
        "n_recipes": len(shared),
        "spearman": spearman(a, b),
        "kendall_tau": kendall_tau(a, b),
        "underpowered": len(shared) < 8,
        "recipes": shared,
    }


def effect_trajectory(index: dict, recipe: str, scales: list[str] | None = None,
                      margin: float = DEFAULT_EQUIVALENCE_MARGIN) -> dict[str, Any]:
    """How one recipe's effect moves across scales: grows, holds, reverses, unresolved."""
    scales = [s for s in (scales or SCALE_ORDER) if s in index]
    per_scale = {}
    for s in scales:
        eff = paired_effects(index, s, margin=margin).get(recipe)
        if eff:
            per_scale[s] = eff
    if len(per_scale) < 2:
        return {"recipe": recipe, "per_scale": per_scale, "trajectory": "insufficient_scales"}

    ordered = [s for s in SCALE_ORDER if s in per_scale]
    first, last = per_scale[ordered[0]], per_scale[ordered[-1]]
    if first["verdict"] == "unresolved" or last["verdict"] == "unresolved":
        traj = "unresolved"
    elif first["mean_delta"] * last["mean_delta"] < 0:
        traj = "reverses"
    elif abs(last["mean_delta"]) > abs(first["mean_delta"]) + margin:
        traj = "grows"
    elif abs(last["mean_delta"]) < abs(first["mean_delta"]) - margin:
        traj = "shrinks"
    else:
        traj = "holds"
    return {"recipe": recipe, "per_scale": per_scale, "trajectory": traj,
            "scales": ordered}


# ---------------------------------------------------------------------- #
# top-level report
# ---------------------------------------------------------------------- #
def analyze(runs: list[dict], metric: str = "final_val_loss",
            margin: float = DEFAULT_EQUIVALENCE_MARGIN) -> dict[str, Any]:
    index = index_by_scale_recipe_seed(runs, metric=metric)
    scales = [s for s in SCALE_ORDER if s in index]
    recipes = sorted({r for s in index.values() for r in s})

    pairs = [(a, b) for i, a in enumerate(scales) for b in scales[i + 1:]]
    return {
        "metric": metric,
        "equivalence_margin": margin,
        "n_runs": len(runs),
        "scales": scales,
        "recipes": recipes,
        "protocol": check_protocol_consistency(runs),
        "seed_noise": {s: seed_noise(index, s) for s in scales},
        "paired_effects": {s: paired_effects(index, s, margin=margin) for s in scales},
        "regret": [selection_regret(index, a, b) for a, b in pairs],
        "selection_probability": [selection_probability(index, a, b) for a, b in pairs],
        "rank_transfer": [rank_transfer(index, a, b) for a, b in pairs],
        "trajectories": {
            r: effect_trajectory(index, r, scales, margin)
            for r in recipes if r != "baseline"
        },
    }


def save_json(result: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

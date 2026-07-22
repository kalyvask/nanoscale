"""Study planning: build the run grid and price it before spending anything.

The recipe grid, size tiers, and seed plan live here so the local ablation runner,
the Modal runner, and the cost estimator all read the same definition.

Cost numbers produced here are *estimates from FLOP accounting*. They are optimistic
for small models, which underutilize a large GPU badly: a 15M-parameter model at 512
context will not reach the assumed MFU. Replace the assumed throughput with a measured
one from the calibration run before trusting any total.
"""

from __future__ import annotations

from dataclasses import dataclass

from nanoscale.config import Config

# recipe name -> (group, override kwargs relative to the full-stack baseline)
RECIPES: dict[str, tuple[str, dict]] = {
    "baseline":   ("baseline",   {}),
    "no_rope":    ("quality",    {"pos": "learned"}),
    "no_rmsnorm": ("quality",    {"norm": "layer"}),
    "no_swiglu":  ("quality",    {"activation": "gelu"}),
    "no_qknorm":  ("stability",  {"qk_norm": False}),
    "no_zloss":   ("stability",  {"z_loss": 0.0}),
    "untied":     ("efficiency", {"tie_weights": False}),
}

# approved geometries (vocab 16384, block 512, head_dim 64)
SIZES: dict[str, dict] = {
    "S": {"n_layer": 6,  "n_embd": 384,  "n_head": 6},
    "M": {"n_layer": 8,  "n_embd": 512,  "n_head": 8},
    "L": {"n_layer": 12, "n_embd": 768,  "n_head": 12},
}

SEEDS: dict[str, int] = {"S": 3, "M": 3, "L": 3}

# A representative variant for the power pilot. Not the largest observed effect: the
# pilot must estimate whether a *typical* effect is resolvable, not a best case.
PILOT_VARIANT = "no_swiglu"

# Approximate on-demand rates and assumed *effective* bf16 throughput.
# Rates change; verify against current pricing. Effective TFLOP/s assumes moderate
# utilization that small models will not achieve -- calibrate, do not trust.
GPU_PRESETS: dict[str, tuple[float, float]] = {
    # name: (usd_per_hour, effective_tflops)
    "l4":   (0.80, 60.0),
    "a10g": (1.10, 90.0),
    "l40s": (1.95, 180.0),
    "a100": (2.50, 150.0),
    "h100": (3.95, 350.0),
}


@dataclass(frozen=True)
class PlannedRun:
    size: str
    recipe: str
    group: str
    seed_index: int
    config: Config

    def estimated_gpu_hours(self, effective_tflops: float) -> float:
        """Estimate only. Replace ``effective_tflops`` with a measured value."""
        flops = self.config.estimated_flops_per_token() * self.config.total_tokens()
        return flops / (effective_tflops * 1e12) / 3600.0

    # deprecated alias
    def gpu_hours(self, effective_tflops: float) -> float:
        return self.estimated_gpu_hours(effective_tflops)


def scale_target_tokens(base: Config, geo: dict) -> int:
    """Token budget for a scale, computed from the BASELINE geometry only.

    Every recipe at this scale then trains on exactly this many tokens. Deriving the
    budget per-recipe would give learned positions and untied embeddings more data
    than the baseline purely because they have more parameters, which would confound
    the intervention with the training budget.
    """
    baseline_flip = RECIPES["baseline"][1]
    baseline_cfg = base.override(**geo, **baseline_flip, target_train_tokens=None)
    return int(baseline_cfg.tokens_per_param * baseline_cfg.n_params())


def build_grid(
    base: Config,
    sizes: dict[str, dict] | None = None,
    seeds: dict[str, int] | None = None,
    recipes: dict[str, tuple[str, dict]] | None = None,
    study_id: str = "transfer",
) -> list[PlannedRun]:
    """Full recipe x size x seed grid, as concrete validated configs.

    Within a scale every recipe receives an identical token budget and therefore an
    identical ``max_steps``. ``data_seed`` is shared across scales for a given seed
    index so that S, M and L consume nested prefixes of the same stream order.
    """
    sizes = SIZES if sizes is None else sizes
    seeds = SEEDS if seeds is None else seeds
    recipes = RECIPES if recipes is None else recipes

    runs: list[PlannedRun] = []
    for size_name, geo in sizes.items():
        target_tokens = scale_target_tokens(base, geo)
        for s in range(seeds.get(size_name, 1)):
            for recipe, (group, flip) in recipes.items():
                cfg = base.override(
                    **geo,
                    **flip,
                    target_train_tokens=target_tokens,
                    max_steps=None,
                    seed=base.seed + s,
                    init_seed=base.seed + s,
                    data_seed=base.seed + s,   # shared across scales -> nested prefixes
                    study_id=study_id,
                    scale_id=size_name,
                    recipe_id=recipe,
                    name=f"{size_name}-{recipe}-s{s}",
                    group="transfer",
                )
                runs.append(PlannedRun(size_name, recipe, group, s, cfg))
    return runs


def variance_pilot(
    base: Config,
    size: str = "S",
    n_seeds: int = 5,
    variant: str | None = PILOT_VARIANT,
) -> list[PlannedRun]:
    """Power pilot: baseline plus one representative variant, repeated seeds.

    Baseline-only seeds give the noise floor. Adding one representative variant also
    gives a rough effect size, and the ratio of the two is what decides whether the
    planned seed count can resolve anything. Without this you cannot distinguish
    'no effect' from 'not enough seeds', and every result risks coming back
    unresolved after the budget is already spent.
    """
    recipes = {"baseline": RECIPES["baseline"]}
    if variant is not None:
        recipes[variant] = RECIPES[variant]
    return build_grid(
        base,
        sizes={size: SIZES[size]},
        seeds={size: n_seeds},
        recipes=recipes,
        study_id="pilot",
    )


def protocol_hash(base: Config, sizes: dict[str, dict] | None = None,
                  recipes: dict[str, tuple[str, dict]] | None = None,
                  extra: dict | None = None) -> str:
    """Stable hash over everything that must stay constant for the study to be valid.

    Changing any of it invalidates cross-run comparability, so runs carry the hash and
    analysis can refuse to pool runs from different protocols.
    """
    import hashlib
    import json

    payload = {
        "protocol_fields": base.protocol_fields(),
        "sizes": sizes if sizes is not None else SIZES,
        "recipes": {k: v[1] for k, v in (recipes or RECIPES).items()},
        "seeds": SEEDS,
        "extra": extra or {},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def summarize(runs: list[PlannedRun], effective_tflops: float, usd_per_hour: float) -> dict:
    """Per-size and total GPU-hours and cost for a planned set of runs."""
    by_size: dict[str, dict] = {}
    for r in runs:
        e = by_size.setdefault(
            r.size, {"runs": 0, "gpu_hours": 0.0, "params": r.config.n_params(),
                     "tokens_per_run": r.config.total_tokens()}
        )
        e["runs"] += 1
        e["gpu_hours"] += r.gpu_hours(effective_tflops)
    for e in by_size.values():
        e["usd"] = e["gpu_hours"] * usd_per_hour
    total_h = sum(e["gpu_hours"] for e in by_size.values())
    return {
        "by_size": by_size,
        "total_runs": len(runs),
        "total_gpu_hours": total_h,
        "total_usd": total_h * usd_per_hour,
        "effective_tflops": effective_tflops,
        "usd_per_hour": usd_per_hour,
    }


def fits_budget(summary: dict, budget_usd: float) -> bool:
    return summary["total_usd"] <= budget_usd


def largest_affordable_plan(
    base: Config, budget_usd: float, effective_tflops: float, usd_per_hour: float,
    reserve_frac: float = 0.25,
) -> tuple[list[str], dict]:
    """Pick the largest prefix of size tiers (S, then S+M, ...) that fits the budget.

    ``reserve_frac`` holds back part of the budget for reruns, failures, and the
    calibration pass, because a plan that exactly exhausts the budget cannot recover
    from a single crashed run.
    """
    spendable = budget_usd * (1.0 - reserve_frac)
    order = [s for s in ("S", "M", "L") if s in SIZES]
    chosen: list[str] = []
    last_summary = summarize([], effective_tflops, usd_per_hour)
    for i in range(1, len(order) + 1):
        candidate = order[:i]
        runs = build_grid(base, sizes={k: SIZES[k] for k in candidate},
                          seeds={k: SEEDS[k] for k in candidate})
        summary = summarize(runs, effective_tflops, usd_per_hour)
        if summary["total_usd"] > spendable:
            break
        chosen, last_summary = candidate, summary
    return chosen, last_summary

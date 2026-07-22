import pytest

from nanoscale.config import Config
from nanoscale.planning import (
    RECIPES,
    SEEDS,
    SIZES,
    build_grid,
    fits_budget,
    largest_affordable_plan,
    summarize,
    variance_pilot,
)

BASE = Config(vocab_size=16384, block_size=512, tokens_per_param=20)


def test_grid_size_matches_recipes_times_seeds():
    runs = build_grid(BASE)
    expected = sum(len(RECIPES) * SEEDS[s] for s in SIZES)
    assert len(runs) == expected
    # every planned config is valid and distinctly named
    names = {r.config.name for r in runs}
    assert len(names) == len(runs)


def test_grid_applies_geometry_and_flip():
    runs = build_grid(BASE)
    s_gelu = next(r for r in runs if r.size == "S" and r.recipe == "no_swiglu")
    assert s_gelu.config.n_layer == SIZES["S"]["n_layer"]
    assert s_gelu.config.activation == "gelu"
    assert s_gelu.group == "quality"
    # baseline keeps the full stack
    s_base = next(r for r in runs if r.size == "S" and r.recipe == "baseline")
    assert s_base.config.activation == "swiglu" and s_base.config.qk_norm is True


def test_seeds_differ_within_a_recipe():
    runs = build_grid(BASE)
    seeds = {r.config.seed for r in runs if r.size == "S" and r.recipe == "baseline"}
    assert len(seeds) == SEEDS["S"]


def test_gpu_hours_scale_with_size():
    runs = build_grid(BASE)
    s = next(r for r in runs if r.size == "S" and r.recipe == "baseline")
    l = next(r for r in runs if r.size == "L" and r.recipe == "baseline")
    assert l.gpu_hours(150.0) > s.gpu_hours(150.0)
    # halving throughput doubles the time
    assert s.gpu_hours(75.0) == pytest.approx(2 * s.gpu_hours(150.0))


def test_largest_tier_dominates_cost():
    """The L tier should be the large majority of total compute."""
    summary = summarize(build_grid(BASE), 150.0, 2.5)
    l_share = summary["by_size"]["L"]["gpu_hours"] / summary["total_gpu_hours"]
    assert l_share > 0.7


def test_variance_pilot_is_baseline_only():
    pilot = variance_pilot(BASE, size="S", n_seeds=3)
    assert len(pilot) == 3
    assert {r.recipe for r in pilot} == {"baseline"}
    assert len({r.config.seed for r in pilot}) == 3


def test_summarize_and_budget_check():
    summary = summarize(build_grid(BASE), 150.0, 2.5)
    assert summary["total_runs"] == len(build_grid(BASE))
    assert summary["total_usd"] == pytest.approx(
        summary["total_gpu_hours"] * 2.5
    )
    assert fits_budget(summary, summary["total_usd"] + 1)
    assert not fits_budget(summary, summary["total_usd"] - 1)


def test_largest_affordable_plan_prefers_prefix():
    # a tiny budget affords nothing
    chosen, _ = largest_affordable_plan(BASE, 0.01, 150.0, 2.5)
    assert chosen == []
    # a huge budget affords everything
    chosen, summary = largest_affordable_plan(BASE, 10_000.0, 150.0, 2.5)
    assert chosen == ["S", "M", "L"]
    # a middling budget affords a prefix, never a gap
    chosen, _ = largest_affordable_plan(BASE, 30.0, 150.0, 2.5)
    assert chosen in ([], ["S"], ["S", "M"], ["S", "M", "L"])


def test_reserve_fraction_holds_budget_back():
    generous, _ = largest_affordable_plan(BASE, 30.0, 150.0, 2.5, reserve_frac=0.0)
    cautious, _ = largest_affordable_plan(BASE, 30.0, 150.0, 2.5, reserve_frac=0.5)
    assert len(cautious) <= len(generous)

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


def test_every_recipe_at_a_scale_gets_identical_token_budget():
    """The core protocol-validity property: budget must not depend on the recipe."""
    runs = build_grid(BASE)
    for scale in SIZES:
        at_scale = [r for r in runs if r.size == scale]
        budgets = {r.config.total_tokens() for r in at_scale}
        steps = {r.config.derived_max_steps() for r in at_scale}
        assert len(budgets) == 1, f"{scale}: recipes disagree on token budget {budgets}"
        assert len(steps) == 1, f"{scale}: recipes disagree on max_steps {steps}"


def test_parameter_adding_recipes_get_no_extra_data():
    """Regression: learned positions and untied embeddings add parameters.

    Deriving the budget from each run's own n_params would silently hand those two
    recipes more training tokens than the baseline, confounding the intervention with
    the amount of data seen.
    """
    runs = build_grid(BASE)
    for scale in SIZES:
        at_scale = {r.recipe: r for r in runs if r.size == scale and r.seed_index == 0}
        base_run = at_scale["baseline"]
        for recipe in ("no_rope", "untied"):
            r = at_scale[recipe]
            # these recipes really do have more parameters ...
            assert r.config.n_params() > base_run.config.n_params()
            # ... but must still train on exactly the baseline's token budget
            assert r.config.total_tokens() == base_run.config.total_tokens()
            assert r.config.derived_max_steps() == base_run.config.derived_max_steps()


def test_data_seed_shared_across_scales_for_nested_prefixes():
    runs = build_grid(BASE)
    for s in range(SEEDS["S"]):
        seeds_at_index = {
            r.config.resolved_data_seed
            for r in runs if r.seed_index == s and r.recipe == "baseline"
        }
        assert len(seeds_at_index) == 1, "data_seed must be shared across scales"


def test_identity_fields_populated():
    runs = build_grid(BASE, study_id="unit")
    for r in runs:
        c = r.config
        assert c.study_id == "unit"
        assert c.scale_id == r.size and c.recipe_id == r.recipe
        assert c.init_seed is not None and c.data_seed is not None


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


def test_variance_pilot_pairs_baseline_with_a_representative_variant():
    """Noise floor alone is not enough; the pilot needs an effect size to compare to."""
    from nanoscale.planning import PILOT_VARIANT

    pilot = variance_pilot(BASE, size="S", n_seeds=5)
    assert len(pilot) == 10  # 2 recipes x 5 seeds
    assert {r.recipe for r in pilot} == {"baseline", PILOT_VARIANT}
    assert len({r.config.resolved_init_seed for r in pilot}) == 5
    # the pilot is a separate study, not pooled with the main grid
    assert {r.config.study_id for r in pilot} == {"pilot"}
    # and it inherits the equal-budget property
    assert len({r.config.total_tokens() for r in pilot}) == 1


def test_variance_pilot_can_be_baseline_only():
    pilot = variance_pilot(BASE, size="S", n_seeds=3, variant=None)
    assert len(pilot) == 3
    assert {r.recipe for r in pilot} == {"baseline"}


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

import pytest

from nanoscale.config import (
    Config,
    coerce_overrides,
    load_config,
    parse_overrides,
)


def test_default_config_is_valid():
    cfg = Config()
    assert cfg.n_params() > 0
    assert cfg.head_dim == cfg.n_embd // cfg.n_head


def test_head_divisibility_rejected():
    with pytest.raises(ValueError, match="divisible"):
        Config(n_embd=100, n_head=7)


def test_rope_requires_even_head_dim():
    # 12 / 4 = head_dim 3 (odd) -> invalid for rope
    with pytest.raises(ValueError, match="even head dimension"):
        Config(n_embd=12, n_head=4, pos="rope")
    # learned positions do not require even head_dim
    Config(n_embd=12, n_head=4, pos="learned")


def test_vocab_must_fit_uint16():
    with pytest.raises(ValueError, match="uint16"):
        Config(vocab_size=70000)


def test_bad_enum_rejected():
    with pytest.raises(ValueError, match="pos"):
        Config(pos="absolute")
    with pytest.raises(ValueError, match="attention_backend"):
        Config(attention_backend="flash3")


def test_positivity_checks():
    with pytest.raises(ValueError):
        Config(n_layer=0)
    with pytest.raises(ValueError):
        Config(batch_size=-1)


def test_override_revalidates():
    cfg = Config()
    cfg2 = cfg.override(n_layer=8)
    assert cfg2.n_layer == 8
    assert cfg.n_layer == 6  # original unchanged
    with pytest.raises(ValueError):
        cfg.override(n_head=7)  # 512 % 7 != 0
    with pytest.raises(ValueError, match="unknown"):
        cfg.override(not_a_field=1)


def test_ffn_hidden_parameter_matching():
    gelu = Config(activation="gelu", n_embd=512)
    swiglu = Config(activation="swiglu", n_embd=512)
    assert gelu.ffn_hidden == 4 * 512
    assert swiglu.ffn_hidden % 64 == 0
    # swiglu (3 matrices) and gelu (2 matrices) FFN param counts should be close
    gelu_ffn = 2 * gelu.n_embd * gelu.ffn_hidden
    swiglu_ffn = 3 * swiglu.n_embd * swiglu.ffn_hidden
    assert abs(gelu_ffn - swiglu_ffn) / gelu_ffn < 0.05


def test_tie_weights_changes_param_count_by_embedding():
    tied = Config(tie_weights=True)
    untied = Config(tie_weights=False)
    assert untied.n_params() - tied.n_params() == tied.vocab_size * tied.n_embd


def test_learned_positions_add_params():
    rope = Config(pos="rope")
    learned = Config(pos="learned")
    assert learned.n_params() - rope.n_params() == rope.block_size * rope.n_embd


def test_derived_budget():
    cfg = Config(max_steps=None, tokens_per_param=20, batch_size=8, block_size=128)
    steps = cfg.derived_max_steps()
    assert steps == cfg.total_tokens() // cfg.tokens_per_step()
    assert Config(max_steps=123).derived_max_steps() == 123


def test_flops_positive_and_scales_with_layers():
    small = Config(n_layer=2)
    big = Config(n_layer=8)
    assert small.flops_per_token() > 0
    assert big.flops_per_token() > small.flops_per_token()


def test_yaml_round_trip(tmp_path):
    cfg = Config(name="rt", n_layer=3, activation="gelu")
    path = tmp_path / "c.yaml"
    cfg.save_yaml(path)
    loaded = Config.from_yaml(path)
    assert loaded == cfg


def test_unknown_yaml_field_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("mystery: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        Config.from_yaml(path)


def test_parse_and_coerce_overrides():
    raw = parse_overrides(["--n_layer", "10", "--lr=1e-3", "--qk_norm", "false",
                           "--tokenizer_path", "none"])
    assert raw == {"n_layer": "10", "lr": "1e-3", "qk_norm": "false",
                   "tokenizer_path": "none"}
    coerced = coerce_overrides(Config(), raw)
    assert coerced["n_layer"] == 10 and isinstance(coerced["n_layer"], int)
    assert coerced["lr"] == pytest.approx(1e-3)
    assert coerced["qk_norm"] is False
    assert coerced["tokenizer_path"] is None


def test_load_config_applies_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    Config(name="x").save_yaml(path)
    cfg = load_config(path, ["--n_layer", "4", "--activation", "gelu"])
    assert cfg.n_layer == 4 and cfg.activation == "gelu"


def test_parse_overrides_missing_value():
    with pytest.raises(ValueError, match="missing value"):
        parse_overrides(["--lr"])

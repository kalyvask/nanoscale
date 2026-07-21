import itertools

import pytest

torch = pytest.importorskip("torch")

from nanoscale.config import Config
from nanoscale.model import (
    GPT,
    RMSNorm,
    Attention,
    apply_rope,
    build_rope_cache,
)


def make_cfg(**over):
    base = dict(vocab_size=64, block_size=16, n_layer=2, n_head=4, n_embd=32,
                z_loss=1e-4, attention_backend="reference")
    base.update(over)
    return Config(**base)


def test_forward_shapes():
    cfg = make_cfg()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_backward_runs_and_updates():
    cfg = make_cfg()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    _, loss = model(idx, idx)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize(
    "pos,norm,activation,qk_norm,tie,bias",
    list(itertools.product(
        ["rope", "learned"], ["rms", "layer"], ["swiglu", "gelu"],
        [True, False], [True, False], [True, False],
    )),
)
def test_exact_param_accounting(pos, norm, activation, qk_norm, tie, bias):
    cfg = make_cfg(pos=pos, norm=norm, activation=activation,
                   qk_norm=qk_norm, tie_weights=tie, bias=bias)
    model = GPT(cfg)
    assert model.num_parameters() == cfg.n_params()


def test_tied_weights_share_storage():
    model = GPT(make_cfg(tie_weights=True))
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
    untied = GPT(make_cfg(tie_weights=False))
    assert untied.lm_head.weight.data_ptr() != untied.tok_emb.weight.data_ptr()


def test_causal_masking_no_future_leakage():
    cfg = make_cfg(attention_backend="reference")
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    with torch.no_grad():
        logits_a, _ = model(idx)
        idx2 = idx.clone()
        idx2[0, -1] = (idx2[0, -1] + 1) % cfg.vocab_size  # change only the last token
        logits_b, _ = model(idx2)
    # every position except the last must be unchanged
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)
    assert not torch.allclose(logits_a[:, -1], logits_b[:, -1])


def test_reference_and_sdpa_agree():
    cfg = make_cfg(attention_backend="reference", dropout=0.0)
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    with torch.no_grad():
        ref_logits, _ = model(idx)
        for m in model.modules():
            if isinstance(m, Attention):
                m.backend = "sdpa"
        sdpa_logits, _ = model(idx)
    assert torch.allclose(ref_logits, sdpa_logits, atol=1e-4, rtol=1e-4)


def test_rmsnorm_matches_manual():
    x = torch.randn(3, 8)
    norm = RMSNorm(8)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, 8))
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * norm.weight
    assert torch.allclose(norm(x), expected, atol=1e-6)


def test_rope_is_identity_at_pos_zero_and_preserves_norm():
    head_dim = 8
    cos, sin = build_rope_cache(head_dim, max_seq=4)
    x = torch.randn(1, 2, 4, head_dim)  # (B, n_head, T, head_dim)
    out = apply_rope(x, cos, sin)
    # position 0 has angle 0 -> unchanged
    assert torch.allclose(out[:, :, 0], x[:, :, 0], atol=1e-6)
    # rotation preserves vector norm at every position
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_z_loss_formula():
    cfg = make_cfg(z_loss=0.1)
    model = GPT(cfg)
    logits = torch.randn(2, 5, cfg.vocab_size)
    targets = torch.randint(0, cfg.vocab_size, (2, 5))
    flat = logits.view(-1, cfg.vocab_size)
    ce = torch.nn.functional.cross_entropy(flat, targets.view(-1))
    z = torch.logsumexp(flat, dim=-1).pow(2).mean()
    expected = ce + 0.1 * z
    assert torch.allclose(model.loss_fn(logits, targets), expected, atol=1e-6)


def test_same_seed_same_cpu_result():
    torch.manual_seed(0)
    a = GPT(make_cfg())
    torch.manual_seed(0)
    b = GPT(make_cfg())
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    idx = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        assert torch.equal(a(idx)[0], b(idx)[0])


def test_can_overfit_one_batch():
    cfg = make_cfg(z_loss=0.0)
    model = GPT(cfg).train()
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = None
    for _ in range(200):
        opt.zero_grad()
        _, loss = model(idx, idx)
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 < first


def test_generate_extends_sequence():
    cfg = make_cfg()
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=5)
    assert out.shape == (1, 9)

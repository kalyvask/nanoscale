"""Decoder-only, pre-norm Transformer with every studied component behind a flag.

Interventions selected by ``Config``:

* ``pos``:        rope | learned  (rotary vs learned absolute positions)
* ``norm``:       rms  | layer
* ``activation``: swiglu | gelu   (parameter-matched FFN width)
* ``qk_norm``:    RMSNorm on q,k before attention
* ``z_loss``:     output-logit regularization
* ``tie_weights``: share the token embedding with the LM head
* ``attention_backend``: reference | sdpa

The ``reference`` attention path is a plain, readable softmax implementation used for
testing; the ``sdpa`` path uses ``F.scaled_dot_product_attention`` for speed. With
dropout disabled the two agree numerically (tested).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoscale.config import Config


# ---------------------------------------------------------------------- #
# norms
# ---------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def make_norm(cfg: Config, dim: int) -> nn.Module:
    if cfg.norm == "rms":
        return RMSNorm(dim)
    return nn.LayerNorm(dim, bias=cfg.bias)


# ---------------------------------------------------------------------- #
# rotary position embedding
# ---------------------------------------------------------------------- #
def build_rope_cache(head_dim: int, max_seq: int, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)              # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)       # (T, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_head, T, head_dim); cos/sin: (T, head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------- #
# attention
# ---------------------------------------------------------------------- #
class Attention(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.backend = cfg.attention_backend
        self.dropout = cfg.dropout
        self.use_rope = cfg.pos == "rope"
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_drop = nn.Dropout(cfg.dropout)
        if cfg.qk_norm:
            self.q_norm: nn.Module | None = RMSNorm(self.head_dim)
            self.k_norm: nn.Module | None = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x: torch.Tensor, cos=None, sin=None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.use_rope:
            q = apply_rope(q, cos[:T], sin[:T])
            k = apply_rope(k, cos[:T], sin[:T])

        if self.backend == "sdpa":
            drop = self.dropout if self.training else 0.0
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        else:
            y = self._reference_attention(q, k, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

    def _reference_attention(self, q, k, v) -> torch.Tensor:
        T = q.shape[-2]
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        if self.training and self.dropout > 0:
            att = F.dropout(att, p=self.dropout)
        return att @ v


# ---------------------------------------------------------------------- #
# feed-forward
# ---------------------------------------------------------------------- #
class GeluMLP(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        h = cfg.ffn_hidden
        self.fc = nn.Linear(cfg.n_embd, h, bias=cfg.bias)
        self.proj = nn.Linear(h, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class SwiGluMLP(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        h = cfg.ffn_hidden
        self.gate = nn.Linear(cfg.n_embd, h, bias=cfg.bias)
        self.up = nn.Linear(cfg.n_embd, h, bias=cfg.bias)
        self.down = nn.Linear(h, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


def make_mlp(cfg: Config) -> nn.Module:
    return SwiGluMLP(cfg) if cfg.activation == "swiglu" else GeluMLP(cfg)


# ---------------------------------------------------------------------- #
# block and model
# ---------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.norm1 = make_norm(cfg, cfg.n_embd)
        self.attn = Attention(cfg)
        self.norm2 = make_norm(cfg, cfg.n_embd)
        self.mlp = make_mlp(cfg)

    def forward(self, x, cos=None, sin=None):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = (
            nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos == "learned" else None
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = make_norm(cfg, cfg.n_embd)
        self.lm_head = nn.Linear(
            cfg.n_embd, cfg.vocab_size, bias=cfg.bias and not cfg.tie_weights
        )
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        if cfg.pos == "rope":
            cos, sin = build_rope_cache(cfg.head_dim, cfg.block_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.rope_cos = self.rope_sin = None

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        cos = self.rope_cos.to(x.dtype) if self.rope_cos is not None else None
        sin = self.rope_sin.to(x.dtype) if self.rope_sin is not None else None
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = self.loss_fn(logits, targets)
        return logits, loss

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        V = logits.size(-1)
        flat = logits.view(-1, V)
        ce = F.cross_entropy(flat, targets.view(-1))
        if self.cfg.z_loss > 0:
            z = torch.logsumexp(flat, dim=-1).pow(2).mean()
            return ce + self.cfg.z_loss * z
        return ce

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
        if was_training:
            self.train()
        return idx

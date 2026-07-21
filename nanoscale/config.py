"""Frozen, validated experiment configuration.

A ``Config`` fully describes one run: model shape, the recipe (which modern
components are on), the controlled training budget, and system settings. It is
immutable; produce a variant with :meth:`Config.override`, which re-validates.

Parameter and FLOP accounting live here so the training budget can be derived
from ``tokens_per_param`` (D/N) without instantiating a model, and so tests can
check the analytical counts against the built module.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Allowed values for enum-like string fields.
POS_CHOICES = ("rope", "learned")
NORM_CHOICES = ("rms", "layer")
ACT_CHOICES = ("swiglu", "gelu")
ATTN_CHOICES = ("reference", "sdpa")
DEVICE_CHOICES = ("auto", "cpu", "cuda")
DTYPE_CHOICES = ("auto", "fp32", "bf16", "fp16")

# uint16 token storage: vocab must fit.
MAX_VOCAB = 1 << 16  # 65536


@dataclass(frozen=True)
class Config:
    # --- identity ---
    name: str = "base"
    group: str | None = None
    seed: int = 1337

    # --- model ---
    vocab_size: int = 16384
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = False

    # --- recipe surface (the interventions) ---
    pos: str = "rope"          # rope | learned
    norm: str = "rms"          # rms | layer
    activation: str = "swiglu" # swiglu | gelu
    qk_norm: bool = True
    z_loss: float = 1.0e-4
    tie_weights: bool = True
    attention_backend: str = "sdpa"  # reference | sdpa

    # --- controlled training budget ---
    tokens_per_param: float = 20.0   # D/N
    batch_size: int = 32
    grad_accum: int = 1
    max_steps: int | None = None     # if set, overrides the derived budget
    warmup_frac: float = 0.05
    lr: float = 3.0e-4
    min_lr_frac: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 100

    # --- data / system ---
    dataset: str = "tinyshakespeare"
    data_dir: str | None = None
    tokenizer_path: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    compile: bool = False

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        self._check_positive()
        self._check_enums()
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        head_dim = self.n_embd // self.n_head
        if self.pos == "rope" and head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires an even head dimension; got head_dim={head_dim} "
                f"(n_embd={self.n_embd}, n_head={self.n_head})"
            )
        if not (0 < self.vocab_size <= MAX_VOCAB):
            raise ValueError(
                f"vocab_size ({self.vocab_size}) must be in (0, {MAX_VOCAB}] to fit uint16 storage"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1); got {self.dropout}")
        if self.z_loss < 0:
            raise ValueError(f"z_loss must be >= 0; got {self.z_loss}")
        if not (0.0 <= self.warmup_frac < 1.0):
            raise ValueError(f"warmup_frac must be in [0, 1); got {self.warmup_frac}")
        if not (0.0 < self.min_lr_frac <= 1.0):
            raise ValueError(f"min_lr_frac must be in (0, 1]; got {self.min_lr_frac}")
        if self.tokens_per_param <= 0:
            raise ValueError(f"tokens_per_param must be > 0; got {self.tokens_per_param}")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps must be > 0 when set; got {self.max_steps}")

    def _check_positive(self) -> None:
        positive_ints = {
            "block_size": self.block_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "eval_interval": self.eval_interval,
            "eval_iters": self.eval_iters,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int; got {value!r}")

    def _check_enums(self) -> None:
        for name, value, choices in (
            ("pos", self.pos, POS_CHOICES),
            ("norm", self.norm, NORM_CHOICES),
            ("activation", self.activation, ACT_CHOICES),
            ("attention_backend", self.attention_backend, ATTN_CHOICES),
            ("device", self.device, DEVICE_CHOICES),
            ("dtype", self.dtype, DTYPE_CHOICES),
        ):
            if value not in choices:
                raise ValueError(f"{name} must be one of {choices}; got {value!r}")

    # ------------------------------------------------------------------ #
    # derived geometry
    # ------------------------------------------------------------------ #
    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def ffn_hidden(self) -> int:
        """FFN inner width, parameter-matched across activations.

        GeLU uses 4*d. SwiGLU has three projections instead of two, so we size it
        to ~8/3*d (rounded to a multiple of 64) to match the GeLU parameter count.
        """
        if self.activation == "gelu":
            return 4 * self.n_embd
        target = int(round(8 * self.n_embd / 3))
        return _round_to_multiple(target, 64)

    # ------------------------------------------------------------------ #
    # parameter and FLOP accounting (analytical; matched by tests)
    # ------------------------------------------------------------------ #
    def n_params_non_embedding(self) -> int:
        d = self.n_embd
        n_layers = self.n_layer
        per_layer = 0
        # attention projections q,k,v,o
        per_layer += 4 * d * d
        if self.bias:
            per_layer += 4 * d
        # qk norm weights (one vector of head_dim for q and one for k)
        if self.qk_norm:
            per_layer += 2 * self.head_dim
        # two block norms
        per_layer += 2 * self._norm_params(d)
        # feed-forward
        h = self.ffn_hidden
        if self.activation == "gelu":
            per_layer += d * h + h * d  # fc, proj
            if self.bias:
                per_layer += h + d
        else:  # swiglu: gate, up, down
            per_layer += 3 * d * h
            if self.bias:
                per_layer += 2 * h + d
        total = per_layer * n_layers
        total += self._norm_params(d)  # final norm
        return total

    def n_params(self) -> int:
        d = self.n_embd
        total = self.n_params_non_embedding()
        total += self.vocab_size * d  # token embedding
        if self.pos == "learned":
            total += self.block_size * d
        if not self.tie_weights:
            total += self.vocab_size * d  # separate LM head
            if self.bias:
                total += self.vocab_size
        return total

    def _norm_params(self, d: int) -> int:
        # RMSNorm: weight only. LayerNorm: weight (+ bias if enabled).
        if self.norm == "rms":
            return d
        return 2 * d if self.bias else d

    def flops_per_token(self) -> int:
        """Approximate training FLOPs per token (forward + backward).

        6 * N for the parameter matmuls plus a sequence-dependent attention term
        ~ 12 * n_layer * n_embd * block_size. This is an estimate and is logged as
        such, not a precise hardware count.
        """
        n = self.n_params_non_embedding()
        attn = 12 * self.n_layer * self.n_embd * self.block_size
        return 6 * n + attn

    # ------------------------------------------------------------------ #
    # training budget
    # ------------------------------------------------------------------ #
    def tokens_per_step(self) -> int:
        return self.batch_size * self.grad_accum * self.block_size

    def total_tokens(self) -> int:
        return int(self.tokens_per_param * self.n_params())

    def derived_max_steps(self) -> int:
        if self.max_steps is not None:
            return self.max_steps
        steps = self.total_tokens() // self.tokens_per_step()
        return max(1, int(steps))

    # ------------------------------------------------------------------ #
    # serialization and overrides
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def override(self, **kwargs: Any) -> "Config":
        unknown = set(kwargs) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise ValueError(f"unknown config field(s): {sorted(unknown)}")
        return dataclasses.replace(self, **kwargs)

    def save_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config field(s) in input: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"config file {path} must contain a mapping")
        return cls.from_dict(data)


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


# ---------------------------------------------------------------------- #
# CLI overrides
# ---------------------------------------------------------------------- #
def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse ``--key value`` and ``--key=value`` tokens into a raw dict.

    Values remain strings here; :func:`coerce_overrides` casts them against the
    config field types.
    """
    out: dict[str, Any] = {}
    i = 0
    while i < len(pairs):
        tok = pairs[i]
        if not tok.startswith("--"):
            raise ValueError(f"expected --key, got {tok!r}")
        key = tok[2:]
        if "=" in key:
            key, value = key.split("=", 1)
            out[key] = value
            i += 1
        else:
            if i + 1 >= len(pairs):
                raise ValueError(f"missing value for --{key}")
            out[key] = pairs[i + 1]
            i += 2
    return out


def coerce_overrides(base: Config, raw: dict[str, Any]) -> dict[str, Any]:
    """Cast raw string overrides to the types of the matching config fields."""
    fields = {f.name: f for f in dataclasses.fields(base)}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in fields:
            raise ValueError(f"unknown config field: {key}")
        current = getattr(base, key)
        out[key] = _coerce_value(value, current, key)
    return out


def _coerce_value(value: Any, current: Any, key: str) -> Any:
    if not isinstance(value, str):
        return value
    low = value.strip().lower()
    if low in ("none", "null"):
        return None
    if isinstance(current, bool):
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise ValueError(f"cannot parse bool for {key}: {value!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if current is None:
        # optional field currently None: best-effort numeric, else string
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                pass
        return value
    return value


def load_config(yaml_path: str | Path | None, overrides: list[str] | None = None) -> Config:
    """Load a config from YAML (or defaults) and apply CLI ``--key value`` overrides."""
    cfg = Config.from_yaml(yaml_path) if yaml_path else Config()
    if overrides:
        raw = parse_overrides(overrides)
        cfg = cfg.override(**coerce_overrides(cfg, raw))
    return cfg

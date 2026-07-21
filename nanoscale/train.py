"""Training loop.

AdamW with linear warmup then cosine decay, gradient clipping, CPU fp32 / CUDA bf16
autocast, reproducible seeding, and fail-loud NaN/Inf handling. Every run writes a
full record (see :mod:`nanoscale.experiments`) and reports the loss at a predeclared
fixed token budget, not the best-looking checkpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from nanoscale.config import Config, load_config
from nanoscale.data import get_batch, load_meta, load_split
from nanoscale.eval import bits_per_byte, estimate_loss
from nanoscale.experiments import RunRecord


# ---------------------------------------------------------------------- #
# setup helpers
# ---------------------------------------------------------------------- #
def resolve_device(cfg: Config) -> str:
    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device


def resolve_autocast_dtype(cfg: Config, device: str):
    if device != "cuda":
        return None  # CPU: fp32, no autocast
    name = cfg.dtype
    if name == "auto" or name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def lr_at(step: int, cfg: Config, max_steps: int) -> float:
    warmup = int(cfg.warmup_frac * max_steps)
    min_lr = cfg.lr * cfg.min_lr_frac
    if step < warmup:
        return cfg.lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (cfg.lr - min_lr)


# ---------------------------------------------------------------------- #
# training
# ---------------------------------------------------------------------- #
def train(cfg: Config, base_dir: str | Path = "experiments",
          data_dir: str | Path | None = None) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg)
    autocast_dtype = resolve_autocast_dtype(cfg, device)
    data_dir = Path(data_dir) if data_dir else Path(cfg.data_dir or Path("data") / cfg.dataset)

    train_data = load_split(data_dir, "train")
    val_data = load_split(data_dir, "val")
    meta = load_meta(data_dir)

    data_vocab = meta.get("vocab_size")
    if data_vocab is not None and cfg.vocab_size < data_vocab:
        raise ValueError(
            f"config vocab_size ({cfg.vocab_size}) is smaller than the prepared "
            f"data's vocab ({data_vocab}); token ids would exceed the embedding. "
            f"Set vocab_size >= {data_vocab}."
        )

    from nanoscale.model import GPT  # local import keeps torch optional elsewhere

    model = GPT(cfg).to(device)
    if cfg.compile:
        model = torch.compile(model)
    optimizer = build_optimizer(model, cfg)
    max_steps = cfg.derived_max_steps()
    rng = np.random.default_rng(cfg.seed)

    autocast = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else contextlib.nullcontext()
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with RunRecord.create(
        cfg, base_dir=base_dir,
        extra_summary={
            "device": device,
            "dataset_hash": meta.get("dataset_hash"),
            "tokenizer_hash": meta.get("tokenizer_hash"),
        },
    ) as rec:
        import time

        model.train()
        tokens_per_step = cfg.tokens_per_step()
        tokens_seen = 0
        last_loss = float("nan")
        last_grad_norm = 0.0
        max_logit = 0.0
        t0 = time.time()

        for step in range(max_steps):
            lr = lr_at(step, cfg, max_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            for _ in range(cfg.grad_accum):
                x, y = get_batch(train_data, cfg.block_size, cfg.batch_size, device, rng)
                with autocast:
                    logits, loss = model(x, y)
                    loss = loss / cfg.grad_accum
                if not torch.isfinite(loss):
                    raise ValueError(f"non-finite loss at step {step}")
                loss.backward()
                last_loss = loss.item() * cfg.grad_accum
                max_logit = float(logits.detach().abs().max().item())

            if cfg.grad_clip > 0:
                last_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                )
            optimizer.step()
            tokens_seen += tokens_per_step

            is_last = step == max_steps - 1
            if step % cfg.eval_interval == 0 or is_last:
                val_loss = estimate_loss(
                    model, val_data, cfg.block_size, cfg.batch_size,
                    cfg.eval_iters, device, seed=cfg.seed,
                )
                rec.log_metrics({
                    "step": step,
                    "lr": lr,
                    "train_loss": last_loss,
                    "val_loss": val_loss,
                    "grad_norm": last_grad_norm,
                    "max_logit": max_logit,
                    "tokens_seen": tokens_seen,
                })

        train_wall = time.time() - t0
        final_val = estimate_loss(
            model, val_data, cfg.block_size, cfg.batch_size,
            max(cfg.eval_iters, 50), device, seed=cfg.seed + 1,
        )
        bpb = bits_per_byte(final_val, meta.get("compression_val", 1.0))
        if cfg.save_checkpoint:
            torch.save(
                {"model": model.state_dict(), "config": cfg.to_dict()},
                rec.dir / "checkpoint.pt",
            )
        peak_mem = (
            int(torch.cuda.max_memory_allocated()) if device == "cuda" else None
        )
        rec.finish(
            tokens_seen=tokens_seen,
            final_val_loss=final_val,
            bits_per_byte=bpb,
            tokens_per_sec=tokens_seen / max(train_wall, 1e-9),
            peak_memory_bytes=peak_mem,
            max_logit=max_logit,
        )
        summary = dict(rec._summary)

    return summary


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="Train a nanoscale model.")
    ap.add_argument("--config", default=None, help="YAML config path")
    ap.add_argument("--data-dir", default=None, help="override prepared-data directory")
    args, overrides = ap.parse_known_args(argv)

    cfg = load_config(args.config, overrides)
    print("Resolved config:")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
    print(f"  -> n_params={cfg.n_params():,}  steps={cfg.derived_max_steps()}  "
          f"tokens={cfg.total_tokens():,}")

    summary = train(cfg, data_dir=args.data_dir)
    print(f"\nRun {summary['run_id']} [{summary['status']}]")
    print(f"  final_val_loss = {summary['final_val_loss']:.4f}")
    print(f"  bits_per_byte  = {summary['bits_per_byte']:.4f}")
    print(f"  tokens/sec     = {summary['tokens_per_sec']:,.0f}")


if __name__ == "__main__":
    main()

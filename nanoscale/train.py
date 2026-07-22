"""Training loop.

Protocol-relevant behaviour:

* the token budget comes from ``config.total_tokens()``, which the study fixes per
  scale from the baseline geometry, so every recipe at a scale trains on the same data
* training consumes a deterministic packed stream ordered by ``data_seed``, shared
  across scales so budgets nest
* evaluation uses a frozen example set keyed to ``eval_seed`` and independent of the
  training seed
* every run records study/scale/recipe ids, both seeds, the protocol hash and the
  eval-set hash, so runs can be grouped and mismatched protocols refused
* estimated FLOPs and measured throughput are recorded as separate fields
"""

from __future__ import annotations

import argparse
import contextlib
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from nanoscale.config import Config, load_config
from nanoscale.data import FrozenEvalSet, PackedStream, load_meta, load_split, to_torch
from nanoscale.eval import bits_per_byte, evaluate_frozen
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
        return None
    if cfg.dtype in ("auto", "bf16"):
        return torch.bfloat16
    if cfg.dtype == "fp16":
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
def train(
    cfg: Config,
    base_dir: str | Path = "experiments",
    data_dir: str | Path | None = None,
    protocol_hash: str | None = None,
    resume_dir: str | Path | None = None,
) -> dict:
    set_seed(cfg.resolved_init_seed)
    device = resolve_device(cfg)
    autocast_dtype = resolve_autocast_dtype(cfg, device)
    data_dir = Path(data_dir) if data_dir else Path(cfg.data_dir or Path("data") / cfg.dataset)

    train_data = load_split(data_dir, "train")
    val_data = load_split(data_dir, "val")
    meta = load_meta(data_dir)

    data_vocab = meta.get("vocab_size")
    if data_vocab is not None and cfg.vocab_size < data_vocab:
        raise ValueError(
            f"config vocab_size ({cfg.vocab_size}) is smaller than the prepared data's "
            f"vocab ({data_vocab}); set vocab_size >= {data_vocab}."
        )

    from nanoscale.model import GPT

    model = GPT(cfg).to(device)
    if cfg.compile:
        model = torch.compile(model)
    optimizer = build_optimizer(model, cfg)
    max_steps = cfg.derived_max_steps()

    stream = PackedStream(train_data, cfg.block_size, cfg.resolved_data_seed)
    eval_set = FrozenEvalSet(
        val_data, cfg.block_size, n_batches=cfg.eval_iters,
        batch_size=cfg.batch_size, eval_seed=cfg.eval_seed,
    )
    eval_hash = eval_set.content_hash()

    autocast = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None else contextlib.nullcontext()
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_step = 0
    ckpt_path = Path(resume_dir) / "checkpoint.pt" if resume_dir else None
    if ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state.get("step", 0)) + 1

    extra = {
        "device": device,
        "dataset_hash": meta.get("dataset_hash"),
        "tokenizer_hash": meta.get("tokenizer_hash"),
        "study_id": cfg.study_id,
        "scale_id": cfg.scale_id,
        "recipe_id": cfg.recipe_id,
        "init_seed": cfg.resolved_init_seed,
        "data_seed": cfg.resolved_data_seed,
        "eval_set_hash": eval_hash,
        "protocol_hash": protocol_hash,
        "estimated_flops_per_token": cfg.estimated_flops_per_token(),
        "target_train_tokens": cfg.total_tokens(),
        "data_epochs": round(stream.epochs_for_tokens(cfg.total_tokens()), 4),
        "corpus": meta.get("corpus", {}),
    }

    with RunRecord.create(cfg, base_dir=base_dir, extra_summary=extra) as rec:
        model.train()
        tokens_per_step = cfg.tokens_per_step()
        tokens_seen = start_step * tokens_per_step
        last_loss = float("nan")
        last_grad_norm = 0.0
        max_logit = 0.0
        micro = start_step * cfg.grad_accum
        t0 = time.time()

        for step in range(start_step, max_steps):
            lr = lr_at(step, cfg, max_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            for _ in range(cfg.grad_accum):
                xb, yb = stream.batch(micro, cfg.batch_size)
                micro += 1
                x, y = to_torch(xb, yb, device)
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
                val_loss = evaluate_frozen(model, eval_set, device)
                rec.log_metrics({
                    "step": step, "lr": lr, "train_loss": last_loss,
                    "val_loss": val_loss, "grad_norm": last_grad_norm,
                    "max_logit": max_logit, "tokens_seen": tokens_seen,
                })

            if cfg.checkpoint_every and (step + 1) % cfg.checkpoint_every == 0:
                _atomic_save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "step": step, "config": cfg.to_dict()},
                    rec.dir / "checkpoint.pt",
                )

        train_wall = time.time() - t0
        final_val = evaluate_frozen(model, eval_set, device)
        bpb = bits_per_byte(final_val, meta.get("compression_val", 1.0))
        tok_per_sec = tokens_seen / max(train_wall, 1e-9)
        # FLOPs-estimate over measured time: an achieved-throughput figure, not a
        # hardware counter. Kept distinct from the pure estimate above.
        measured_tflops = cfg.estimated_flops_per_token() * tok_per_sec / 1e12

        if cfg.save_checkpoint:
            _atomic_save({"model": model.state_dict(), "config": cfg.to_dict()},
                         rec.dir / "checkpoint.pt")

        rec.finish(
            tokens_seen=tokens_seen,
            final_val_loss=final_val,
            bits_per_byte=bpb,
            tokens_per_sec=tok_per_sec,
            measured_tflops=measured_tflops,
            peak_memory_bytes=int(torch.cuda.max_memory_allocated()) if device == "cuda" else None,
            max_logit=max_logit,
        )
        summary = dict(rec._summary)

    return summary


def _atomic_save(obj, path: Path) -> None:
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="Train a nanoscale model.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None)
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

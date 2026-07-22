"""Evaluation: fixed-budget validation loss, bits-per-byte, and sampling.

Bits-per-byte is tokenizer-independent (it divides out the compression ratio), so
runs with different tokenizers can be compared fairly.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from nanoscale.data import sample_batch


@torch.no_grad()
def estimate_loss(
    model,
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    iters: int,
    device: str,
    seed: int = 0,
) -> float:
    """Average cross-entropy over ``iters`` random batches (seeded, deterministic).

    Legacy path for smoke runs. The study uses :func:`evaluate_frozen`, which scores
    every run on identical examples.
    """
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(iters):
        x, y = sample_batch(data, block_size, batch_size, rng)
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)
        _, loss = model(xt, yt)
        losses.append(loss.item())
    if was_training:
        model.train()
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_frozen(model, eval_set, device: str) -> float:
    """Score the model on a fixed, seed-independent set of examples.

    Every run in the study is evaluated on exactly these blocks, so differences in
    reported loss cannot come from having drawn an easier evaluation sample.
    """
    was_training = model.training
    model.eval()
    losses = []
    for x, y in eval_set.batches():
        xt = torch.from_numpy(x).to(device)
        yt = torch.from_numpy(y).to(device)
        _, loss = model(xt, yt)
        losses.append(loss.item())
    if was_training:
        model.train()
    return float(np.mean(losses))


def bits_per_byte(loss_nats: float, compression: float) -> float:
    """Convert nats/token to bits/byte using bytes-per-token (``compression``)."""
    bits_per_token = loss_nats / math.log(2)
    return bits_per_token / max(compression, 1e-9)


@torch.no_grad()
def sample_text(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    ids = tokenizer.encode_ordinary(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens)
    return tokenizer.decode(out[0].tolist())

# nanoscale

> *A from-scratch language model built to test whether cheap experiments can be trusted.*

Almost nobody tunes architecture on the model they ship. They tune on something small
and cheap, pick the winner, and scale it up. Every ablation table in every blog post is
run at one size, and the transfer to the size that matters is assumed rather than
measured.

That assumption is load-bearing and largely untested. If the recipe that wins at 15M
parameters is not the recipe that wins at 100M, then the small experiment did not save
money, it bought a wrong answer cheaply.

nanoscale measures that. One small Transformer, every modern component behind a config
flag, the same recipe grid run across model sizes with multiple seeds, and two numbers
that say whether the cheap experiment was worth trusting:

1. **Rank transfer.** Does the ordering of recipes at a small size survive at a large one?
2. **Selection regret.** If you pick the small-scale winner and deploy it at scale, how
   much worse is it than the best recipe you could have chosen there?

   ```
   regret = large_loss(recipe chosen at small scale) - best large_loss available
   ```

   Zero means the proxy told the truth. Large regret means the standard practice quietly
   costs you.

**Why it matters.** Compute budgets are decided before the expensive run, on evidence
from cheap runs. If that evidence has poor predictive validity, the cost is not a bad
table, it is a mis-specified training run at the scale where money is actually spent.
The same logic applies to any staged decision made on a cheap proxy: the useful quantity
is not "which option won the pilot" but "how often does the pilot pick the right option,
and what does being wrong cost." Effects are reported separately for **quality**, for
**stability**, and for **efficiency**, because a component can help one and hurt another,
and a single averaged score hides exactly that.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-140%20passing-brightgreen.svg)](tests/)
[![Built for Stanford CS336](https://img.shields.io/badge/built--for-Stanford%20CS336-red.svg)](https://cs336.stanford.edu)

## Try it in 5 minutes, on a laptop CPU, no GPU and no API keys

The whole pipeline runs end to end on a ~1M-parameter model. Nothing here needs an
accelerator or a key.

```bash
pip install -e ".[dev]"
python scripts/prepare_data.py --dataset tinyshakespeare   # download, train BPE, write memmaps
python -m nanoscale.train --config configs/cpu_smoke.yaml  # trains in ~2 min
python scripts/generate.py --prompt "ROMEO:"               # sample from the checkpoint
python scripts/run_ablation.py --config configs/cpu_smoke.yaml --max_steps 100
python scripts/make_table.py                               # grouped ablation table
```

Tokenizer study (bytes vs BPE at several vocabularies):

```bash
python scripts/tokenizer_study.py --dataset tinyshakespeare --with-model \
    --vocab-sizes 1024,4096,8192,16384
```

## What is built

Written from scratch. No training framework, no experiment-tracking service.

- **`tokenizer.py`** byte-level BPE: regex pre-tokenization with frequency-weighted merges
  over unique chunks (16K vocab in minutes, not hours); deterministic tie-breaking;
  special tokens above the BPE range; bounded training sample; save/load with a content hash
- **`data.py`** TXT/JSONL ingest; document-level train/val split *before* concatenation so
  no document straddles the boundary; uint16 memmaps; dataset and tokenizer hashes recorded
- **`model.py`** decoder-only pre-norm Transformer; RoPE or learned positions; RMSNorm or
  LayerNorm; SwiGLU or GeLU (parameter-matched through the FFN width); QK-norm; weight
  tying; z-loss; `reference` and `sdpa` attention paths that agree to 1e-4
- **`train.py`** AdamW, warmup then cosine, gradient clipping, CPU fp32 or CUDA bf16,
  reproducible seeding, loss reported at a predeclared token budget, fail-loud on NaN
- **`experiments.py`** one authoritative directory per run (resolved config, environment,
  git SHA, streamed metrics, summary), with the manifest as a mere index
- **`config.py`** frozen validated config; analytical parameter and FLOP accounting that is
  tested against the built module across all 64 toggle combinations

140 tests, including exact parameter accounting, causal-masking leakage, reference versus
SDPA agreement, tokenizer round-trips on arbitrary bytes, document-split leakage,
overfit-one-batch, and same-seed determinism.

## What has been measured, and what has not

Reported the way the design demands, with the boundary stated rather than blurred.

**Measured.** The instrument works. A 977K-parameter model trains on TinyShakespeare
(validation loss 5.49 to 2.06), generates text with speaker structure, and the
seven-config ablation grid runs end to end. The tokenizer study runs across bytes and BPE
at 1K/4K/8K/16K: BPE beats bytes decisively (4.13 to 2.65 bits per byte), gains saturate
(8K to 16K buys 0.05 bits per byte for twice the tokenizer training time and roughly half
the model throughput), and segmentation visibly sharpens toward word level.

**Not measured, and not claimed.** Nothing about rank transfer, selection regret, or
whether any component's effect reverses with scale. The numbers above come from a
character-level corpus, one seed, a few hundred steps, and a model three orders of
magnitude below the interesting regime. They are plumbing checks that prove the pipeline,
and the tooling stamps them as such in its own output rather than relying on the reader
to remember. A working pipeline feels like progress on the research question and is not.

Two honesty constraints are built into the method rather than left to discipline:
one-at-a-time flips measure **conditional** effects given the rest of the baseline, not
independent contributions; and all comparisons report loss at a **fixed token budget**,
never the best-looking checkpoint.

## Where this is going

The real study runs on FineWeb-Edu at roughly 15M, 40M, and 100M parameters, holding the
tokenizer, corpus, context length, optimizer, schedule shape, and tokens-per-parameter
ratio constant, with 3/3/2 seeds and two preregistered interaction checks. Estimated cost
is about 36 GPU-hours, near $54 on an A100, priced from the FLOP accounting before
anything is launched.

A deliberate non-goal: no claim that a baseline-only fit of `L(N) = E + A/N^alpha`
identifies a compute-optimal model. That needs size, data, and compute varied jointly.
The baseline fit stays a secondary diagnostic.

Progress is gated. A CPU acceptance gate had to pass before any GPU work, and a GPU
spending gate has to pass before any large run.

- Design and methodology: [DESIGN.md](DESIGN.md)
- Milestones, protocol, and gates: [ROADMAP.md](ROADMAP.md)

Status: M0 through M5 built, CPU gate passed, tokenizer study harness validated. Next is
the FineWeb-Edu adapter, which unblocks freezing the tokenizer and the scale configs. No
GPU runs yet.

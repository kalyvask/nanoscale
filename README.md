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

## Protocol constants (frozen)

The study only means anything if these are held fixed, so they are pinned, hashed, and
recorded on every run:

| constant | value |
|---|---|
| corpus | FineWeb-Edu, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, 4 explicit shards |
| tokenizer | byte-level BPE, vocab 16,384, hash `03be9e0e34d77bec` |
| protocol hash | `31f6c08926236ff0` |
| budget | D/N = 20 on total parameters, one target per scale from the baseline geometry |
| grid | 7 recipes x 3 scales x 3 seeds = 63 runs |

Two properties are enforced in code rather than left to care. Every recipe at a scale
trains on an identical token budget, so the recipes that add parameters (learned
positions, untied embeddings) cannot quietly receive more data. And the training stream
is a deterministic permutation keyed only to `data_seed`, so S, M and L consume nested
prefixes of the same data.

## What has been measured, and what has not

Reported the way the design demands, with the boundary stated rather than blurred.

**Measured.** The instrument works, and the tokenizer decision is real. On FineWeb-Edu
with a 25 MB training sample and a 10 MB held-out slice, decided on equal-budget bits per
byte as preregistered:

| tokenizer | compression | fertility | utilization | bits/byte | model tok/s |
|---|---|---|---|---|---|
| bytes | 1.00 | 6.15 | 75.1% | 3.593 | 6,693 |
| bpe_1024 | 2.52 | 2.44 | 93.8% | 3.273 | 5,783 |
| bpe_4096 | 3.44 | 1.79 | 98.4% | 2.823 | 4,532 |
| bpe_8192 | 3.88 | 1.59 | 98.8% | 2.631 | 3,060 |
| **bpe_16384** | 4.27 | 1.44 | 97.8% | **2.466** | 1,960 |

Quality keeps improving with vocabulary while throughput falls by roughly a third from
8K to 16K, which is a genuine trade rather than a free win. The pipeline itself is
proven separately on CPU: a 977K-parameter model trains on TinyShakespeare (validation
loss 5.49 to 2.06), generates text, and the seven-config grid runs end to end.

**Not measured, and not claimed.** Nothing about rank transfer, selection regret, or
whether any component's effect reverses with scale. No model has been trained on
FineWeb-Edu. The CPU numbers come from a character-level corpus, one seed and a few
hundred steps, and the tooling stamps them as plumbing in its own output rather than
relying on the reader to remember. A working pipeline feels like progress on the
research question and is not.

One measurement lesson already earned: on TinyShakespeare, vocabulary utilization
appeared to collapse at 16K (27.8%). That was an artifact of a 150 KB evaluation slice
being too small for rare tokens to appear. With a 10 MB slice utilization is 93-99%
across every BPE size. The harness now refuses a slice under 1 KB and warns below
100 KB, because metrics computed on too little data look plausible rather than wrong.

Two honesty constraints are built into the method rather than left to discipline:
one-at-a-time flips measure **conditional** effects given the rest of the baseline, not
independent contributions; and all comparisons report loss at a **fixed token budget**,
never the best-looking checkpoint.

## Where this is going

The real study runs on FineWeb-Edu at roughly 17M, 33M and 98M parameters with 3 seeds
each: 63 runs, about 24 GPU-hours, near $95 on an H100, priced from the FLOP accounting
before anything is launched. The 98M tier is about 87% of that compute. Interaction
cells are deferred until the balanced three-seed grid at the largest tier is funded,
because buying breadth before the main grid can resolve anything is a bad trade.

The analysis is already written, deliberately: paired per-seed effects, selection
regret, selection probability by seed resampling, descriptive rank correlation (flagged
underpowered at seven recipes), and equivalence-aware verdicts where a small effect with
a wide interval reads `unresolved` rather than "no effect". Choosing how to read numbers
after seeing them is how a study argues itself into a result.

Before the grid there is a ten-run power pilot at the smallest scale: baseline plus one
representative variant across five seeds. It measures the seed-noise floor and a typical
effect size, which together decide whether three seeds can resolve anything at all. It
also sets the equivalence margin, which should come from measured noise rather than
being chosen.

A deliberate non-goal: no claim that a baseline-only fit of `L(N) = E + A/N^alpha`
identifies a compute-optimal model. That needs size, data, and compute varied jointly.
The baseline fit stays a secondary diagnostic.

Progress is gated. A CPU acceptance gate had to pass before any GPU work, and a GPU
spending gate has to pass before any large run.

- Design and methodology: [DESIGN.md](DESIGN.md)
- Milestones, protocol, and gates: [ROADMAP.md](ROADMAP.md)

Status: pipeline built and CPU-verified; protocol hardened; corpus pinned; tokenizer
frozen; runner, analysis and Modal integration in place. The remaining prerequisite is
tokenizing the corpus once, sized for the largest tier that will ever run, because the
stream permutation depends on the block count and appending data later would break the
nested prefixes. Then the power pilot, then the spending gate. **No GPU runs yet.**

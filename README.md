# nanoscale

**Can cheap tiny-model experiments be trusted to pick the training recipe for a much
larger model?**

That is the whole project. People routinely tune architecture and training choices on
small models because large runs are expensive, then assume the winners carry over. This
repo measures how often that assumption holds, and what it costs when it fails.

## The question, precisely

We build one small from-scratch Transformer where every modern component is a config
flag (RoPE vs learned positions, RMSNorm vs LayerNorm, SwiGLU vs GeLU, QK-norm, z-loss,
weight tying). We run the same recipe grid at several model sizes, with multiple seeds,
on a fixed token budget, and ask:

1. **Rank transfer.** Does the ranking of recipes at a small size agree with the ranking
   at a large size? (rank correlation across scales)
2. **Selection regret.** If you *pick* the recipe that looked best at a small size and
   use it at a large size, how much worse is it than the best large-size recipe you could
   have chosen?

   ```
   regret(size) = large_model_loss(recipe chosen at small size)
                - best large_model_loss over all recipes evaluated at that size
   ```

   Regret of zero means the cheap experiment made the right call. Large regret means the
   proxy lied to you.
3. **Effect direction with scale.** For each intervention, does its effect grow, hold,
   reverse, or become statistically unresolved as the model gets bigger?

We report those effects separately for **quality** (val loss, bits/byte), **stability**
(divergence, gradient norms, max logits, learning-rate tolerance), and **efficiency**
(tokens/sec, MFU, peak memory, loss per FLOP), because a trick can help one and hurt
another.

## What this is not

- Not a claim that a baseline scaling curve `L(N) = E + A/N^alpha` identifies a
  compute-optimal model. That needs model size, data, and compute varied jointly. A
  baseline scaling fit is kept as a *secondary* diagnostic only.
- Not a nanoGPT clone. Shared DNA (small decoder-only Transformer), different purpose:
  the object of study is the *decision procedure*, not the model.
- Not a distributed-training framework. Single CPU or single GPU.

## Honesty constraints baked into the method

- One-at-a-time ablations measure **conditional** effects given the rest of the baseline,
  not universal independent contributions. We say so, and add preregistered interaction
  checks.
- The full recipe grid is evaluated at multiple sizes. We do **not** select only the
  small-scale winners and then assert they were optimal at large scale without testing
  credible alternatives there.
- All comparisons are at a **fixed token budget**, reporting the loss at that budget, not
  whichever checkpoint happened to look best.
- SwiGLU and GeLU are parameter-matched through the FFN width. Weight tying is treated
  separately because it materially changes capacity and memory.

## Course context

Built for Stanford CS336 (Language Modeling from Scratch). TinyShakespeare on CPU is a
**plumbing check**, not a finding. The real study runs on FineWeb-Edu on one GPU.

- Design and methodology: [DESIGN.md](DESIGN.md)
- Milestones, protocol, and the GPU spending gate: [ROADMAP.md](ROADMAP.md)

## What is built

Everything from scratch, no training framework:

- `tokenizer.py`: byte-level BPE with regex pre-tokenization and frequency-weighted
  merges over unique chunks; deterministic tie-breaking; special tokens; bounded
  training sample; save/load with a content hash
- `data.py`: TXT/JSONL ingest; document-level train/val split before concatenation so no
  document straddles the boundary; uint16 memmap storage; dataset and tokenizer hashes
- `model.py`: decoder-only pre-norm Transformer; RoPE or learned positions; RMSNorm or
  LayerNorm; SwiGLU or GeLU (parameter-matched); QK-norm; weight tying; z-loss;
  `reference` and `sdpa` attention paths that agree numerically
- `train.py`: AdamW, warmup plus cosine, gradient clipping, CPU fp32 or CUDA bf16,
  reproducible seeding, fixed-budget validation loss, fail-loud NaN handling
- `experiments.py`: one authoritative directory per run (resolved config, environment,
  streamed metrics, summary) with a manifest index
- `config.py`: frozen validated config; analytical parameter and FLOP accounting that
  matches the built module across all 64 toggle combinations

140 tests, including exact parameter accounting, causal-masking leakage, reference
versus SDPA agreement, tokenizer round-trips, document-split leakage, overfit-one-batch,
and same-seed determinism.

## Quickstart (CPU, minutes)

```bash
pip install -e ".[dev]"
python scripts/prepare_data.py --dataset tinyshakespeare
python -m nanoscale.train --config configs/cpu_smoke.yaml
python scripts/generate.py --prompt "ROMEO:"
python scripts/run_ablation.py --config configs/cpu_smoke.yaml --max_steps 100
python scripts/make_table.py
```

Tokenizer study (M1.5):

```bash
python scripts/tokenizer_study.py --dataset tinyshakespeare --with-model --vocab-sizes 1024,4096,8192,16384
```

## Status

M0 through M5 built and the CPU acceptance gate passed. A 977K-parameter model trains on
TinyShakespeare (validation loss 5.49 to 2.06), generates text, and the seven-config
ablation grid runs end to end. Those numbers are plumbing checks, not findings, and the
table says so.

Next: M6 experimental protocol (FineWeb-Edu adapter, freeze the tokenizer, scale configs,
transfer-analysis code), then the GPU spending gate. No GPU runs yet. Estimated core
study cost is about 36 GPU-hours.

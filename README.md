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

Status: building M0-M4 and the CPU ablation harness. No GPU runs yet.

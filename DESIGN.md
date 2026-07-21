# nanoscale — design

## 1. Goal and non-goals

**Goal.** Measure the predictive validity of cheap small-model experiments for
large-model architecture and training decisions. Concretely: run one from-scratch
Transformer recipe grid at several model sizes and quantify how well small-size results
transfer to large size, using rank correlation and selection regret.

**The research question.** *How much regret do we incur when we choose a larger-model
training recipe from tiny-model experiments?* Small-model ablation is standard practice
because large runs are expensive. This project tests whether that practice is
trustworthy and what it costs when it is not.

**Primary outcomes.**
1. **Rank transfer** — rank correlation (Spearman / Kendall) between recipe performance
   at pairs of model sizes.
2. **Selection regret** — `regret(size) = large_loss(recipe picked at small size) -
   best large_loss over recipes evaluated`. Zero means the small experiment chose
   correctly.
3. **Effect direction with scale** — for each intervention, whether its effect grows,
   holds, reverses, or is statistically unresolved as size increases.
4. **Grouped effects** — quality vs stability vs efficiency, reported separately.

**Non-goals.**
- No claim that a baseline scaling fit `L(N)=E+A/N^alpha` yields a compute-optimal model.
  That requires jointly varying size, data, and compute. The baseline fit is a
  **secondary** diagnostic (Section 9), not a headline.
- Not a general training framework, not nanoGPT-wholesale, not multi-node/distributed.
- No custom CUDA/Triton kernel in the core project (Section 11, moved to a stretch goal).

## 2. Design principles

1. **A recipe is a config, never a code fork.** One `model.py`, all components behind
   flags. Every experiment is fully described by its resolved config.
2. **Run directories are authoritative; the manifest is an index.** Each run writes a
   directory of artifacts (Section 8). A top-level `manifest.jsonl` indexes completed
   runs for analysis but is not the source of truth.
3. **Provable before expensive.** The default `cpu_smoke` config trains a ~1M-param model
   on TinyShakespeare in minutes. The identical code path scales to GPU by config only.
   TinyShakespeare numbers are plumbing checks, never findings.
4. **From scratch where it teaches, library where it does not.** We write the tokenizer,
   the Transformer, the training loop, the analysis. We use PyTorch autograd and
   `F.scaled_dot_product_attention` for the fast path, plus a simple **reference**
   attention path for testing. No experiment-tracking service; artifacts are plain files.
5. **Honesty about what an ablation is.** One-at-a-time removals measure conditional
   effects given the baseline, not universal independent contributions. Stated in
   outputs; interaction checks added (Section 7).

## 3. Repository layout

```
nanoscale/
  __init__.py
  config.py         # frozen validated dataclass, YAML load, CLI override, param/FLOP accounting
  tokenizer.py      # byte-level BPE: train / encode / decode / save / load, special tokens
  data.py           # local TXT/JSONL ingest, doc-level split, memmap tokens, batching
  model.py          # Transformer; every component behind a flag; reference|sdpa attention
  train.py          # loop, schedule, mixed precision, seeding, NaN guards
  eval.py           # val loss, bits-per-byte, generation
  experiments.py    # run directories, records, manifest index, env capture, hashing
configs/
  cpu_smoke.yaml    # ~1M params, TinyShakespeare, minutes on CPU
  base.yaml         # full-stack baseline recipe (real-study defaults)
  scales/           # size overrides (added at M6/protocol time): s/m/l[/xl]
scripts/
  prepare_data.py   # ingest a corpus, train/load tokenizer, write memmap bins
  run_ablation.py   # baseline + one-trick-removed variants (CPU plumbing at M5)
  make_table.py     # grouped smoke-test report from run records
analysis/           # added at protocol time: rank correlation, regret, plots
experiments/
  runs/<run_id>/    # authoritative per-run artifacts (gitignored)
  manifest.jsonl    # index of completed runs (gitignored)
tests/
DESIGN.md  ROADMAP.md  README.md  pyproject.toml
```

## 4. Module responsibilities

### config.py
A **frozen, validated** dataclass `Config` (model + training + data + system fields).
- `Config.from_yaml(path)`, `Config.override(**kwargs)` (returns a validated copy),
  CLI `--key value` parsing with type coercion.
- **Early validation, fail fast:** `n_embd % n_head == 0`; RoPE head-dim even and valid;
  `vocab_size` fits the storage dtype (uint16 -> <= 65536); enum fields
  (`pos`, `norm`, `activation`, `attention_backend`, `device`, `dtype`) are legal;
  all of batch/context/model dims positive; `z_loss >= 0`.
- `n_params()` / `n_params_non_embedding()`; `flops_per_token()` (6N plus the attention
  term). The fully resolved config is printed and written before training.

### tokenizer.py
Byte-level BPE from scratch (CS336 Lec 1 / Assignment 1).
- `train(text, vocab_size, max_bytes=None)` with **deterministic tie-breaking** (fixed
  ordering on equal pair counts). `max_bytes` bounds the training sample so we never
  blindly train on an entire corpus.
- `encode`/`decode` with a UTF-8 round-trip guarantee on arbitrary bytes.
- **Special tokens**, including `<|endoftext|>` / document boundary, reserved and never
  produced by merges.
- `save`/`load` equivalence (merge table + vocab + special tokens).
- `bytes` fallback mode (vocab 256, no merges) for the smoke test.
- Reports **fertility** (tokens/word) and **compression** (bytes/token).

### data.py
- Ingest local **TXT and JSONL** (JSONL: a configurable text field per line).
- **Document-level train/val split before concatenation**, deterministic under a recorded
  seed, so no document spans both splits (no leakage).
- Write `train.bin`/`val.bin` as memmapped `uint16` plus `meta.json` recording dataset
  hash, tokenizer hash, document counts, token counts, split seed.
- `get_batch(split, block_size, batch_size, device, generator)`: random contiguous
  windows, targets shifted by one, reproducible under a seed.
- FineWeb-Edu streaming adapter is added **after** the local pipeline is proven.

### model.py
Decoder-only, pre-norm Transformer; every component in Section 6 selected by config.
- Blocks: `norm -> attention -> residual`, `norm -> MLP -> residual`; final norm; LM head.
- **`attention_backend: reference | sdpa`.** The reference path is a plain, readable
  softmax-attention implementation used for testing; the SDPA path
  (`F.scaled_dot_product_attention`) is the performance path. The two must agree within
  tolerance when dropout is disabled (tested).
- Loss = cross-entropy + optional z-loss `z_coef * mean(logsumexp(logits)^2)`.
- `generate(idx, max_new_tokens)` for qualitative samples.

### train.py
- AdamW; linear warmup then cosine decay; gradient clipping.
- CPU fp32; CUDA bf16 autocast. **Seed Python, NumPy, torch, and CUDA** reproducibly.
- Validation at fixed intervals; records loss at the **predeclared fixed token budget**.
- Lightweight checkpoint for smoke runs and selected real runs.
- **Fail loudly on NaN/Inf**, mark the run `failed`, record the reason.
- Emits a full run directory (Section 8) and appends to the manifest index.

### eval.py
- `estimate_loss(model, split, iters)` averaged over random batches.
- `bits_per_byte(loss, compression)` converts nats/token to bits/byte via the measured
  compression ratio, so runs with different tokenizers compare fairly.
- Stability probes surfaced during training: gradient norm, max logit, divergence flag.

### experiments.py
- Allocates `experiments/runs/<run_id>/`, writes `resolved_config.yaml`,
  `environment.json` (Section 8), streams `metrics.jsonl`, finalizes `summary.json`.
- Captures git SHA + dirty flag, dataset/tokenizer hashes, seed, torch/CUDA/device info,
  param counts, estimated FLOPs.
- Appends a one-line index entry to `manifest.jsonl` on completion.

## 5. Config schema

```yaml
# --- identity ---
name: base
group: null            # experiment group tag: "ablation", "scaling", "interaction"
seed: 1337

# --- model ---
vocab_size: 16384      # real study; cpu_smoke uses bytes (256)
block_size: 512        # context length, held constant across sizes
n_layer: 6
n_head: 8
n_embd: 512
dropout: 0.0
bias: false

# --- recipe surface (the interventions) ---
pos: rope              # rope | learned
norm: rms              # rms | layer
activation: swiglu     # swiglu | gelu
qk_norm: true          # RMSNorm on q,k pre-attention
z_loss: 1.0e-4         # coefficient; 0 disables
tie_weights: true      # share token embedding with LM head (capacity/memory: reported apart)
attention_backend: sdpa  # reference | sdpa

# --- controlled training budget ---
tokens_per_param: 20   # D/N; total_tokens = tokens_per_param * n_params (configurable)
batch_size: 32
grad_accum: 1
warmup_frac: 0.05      # fraction of total steps
lr: 3.0e-4
min_lr_frac: 0.1       # min_lr = lr * min_lr_frac
weight_decay: 0.1
grad_clip: 1.0
eval_interval: 250
eval_iters: 100

# --- data / system ---
dataset: tinyshakespeare
tokenizer_path: null   # frozen tokenizer for the real study
device: auto           # auto | cpu | cuda
dtype: auto            # auto -> bf16 on cuda, fp32 on cpu
compile: false         # torch.compile
```

`max_iters` is not hard-coded: it is derived from `tokens_per_param * n_params` and the
tokens-per-step, so the token budget is held constant across sizes by construction.

## 6. The intervention surface, grouped

The **baseline** (`configs/base.yaml`) has all modern components on. Each ablation flips
exactly one, so its record measures that intervention's **conditional** effect given the
rest of the baseline. Grouping matters because the interventions do different jobs:

**Architecture / quality substitutions**
| flag | baseline | ablation | isolates |
|---|---|---|---|
| `pos` | rope | learned | rotary vs learned absolute positions |
| `norm` | rms | layer | RMSNorm vs LayerNorm |
| `activation` | swiglu | gelu | gated MLP vs standard GeLU MLP |

**Stability additions**
| flag | baseline | ablation | isolates |
|---|---|---|---|
| `qk_norm` | true | false | query/key normalization for stability |
| `z_loss` | 1e-4 | 0 | output-logit regularization |

**Capacity / efficiency**
| flag | baseline | ablation | isolates |
|---|---|---|---|
| `tie_weights` | true | false | embedding/output sharing (changes capacity + memory) |

**Fairness rules.**
- SwiGLU uses FFN hidden `~ 8/3 * n_embd` rounded to a multiple of 64 to parameter-match
  the `4 * n_embd` GeLU MLP. Actual params are logged so residual mismatch is visible.
- Total and non-embedding parameter counts logged for every run.
- Exact estimated training FLOPs logged.
- Weight tying reported **separately**, since it materially changes capacity/memory.
- All comparisons at the **fixed token budget**; report fixed-budget loss, not best-ever
  checkpoint.

## 6.5 Tokenizer study (M1.5)

Before the architecture experiments, one tokenizer is chosen and frozen. We compare a
bytes-only tokenizer against byte-level BPE at vocab 1K / 4K / 8K / 16K on one corpus.

**Tokenizer training** uses GPT-2-style regex pre-tokenization, then learns merges over
the set of *unique chunks weighted by frequency* (not the raw byte stream), which is the
standard efficient BPE and keeps merges from crossing word boundaries. The training
sample is bounded (`max_bytes`) so we never scan the whole corpus.

**Reported per tokenizer:**
- compression (bytes/token), fertility (tokens/word), vocabulary utilization (fraction of
  the vocab used on held-out text)
- tokenizer training time and encoding speed (bytes/sec)
- qualitative segmentations of sample strings
- **equal-budget model quality in bits/byte** — the deciding metric

**Why bits/byte decides.** It is tokenizer-independent, so models trained on different
tokenizers compare fairly. "Equal budget" means the same architecture trained for the
same token budget; note a denser tokenizer then sees more underlying text per token,
which is a genuine advantage of compression, not a confound. Compression/fertility/util
are informative but not decisive on their own.

The winning tokenizer is frozen and hashed; every architecture run records that hash, so
the tokenizer is provably held constant across the transfer study.

## 7. Experiment methodology

**Controlled variables held constant across sizes:** tokenizer (frozen), corpus mixture,
context length, optimizer family, LR-schedule *shape*, and tokens-per-parameter ratio
(`D/N = 20` initial, configurable). Only the recipe flag under test and the model size
change.

**Recipe grid at multiple sizes.** The full grid (baseline + single-flip variants) runs
at enough sizes to compute rank transfer. We do **not** shortlist small-scale winners and
skip credible alternatives at large scale.

**Seeds and uncertainty.**
| size | params (approx) | seeds |
|---|---|---|
| S | 15M | 3 |
| M | 40M | 3 |
| L | 100M | >= 2 |
| XL (optional) | 300M | 2 (confirmation, if compute permits) |

Every reported metric carries a spread across seeds (mean plus a dispersion estimate;
small-n CIs where meaningful). An effect that is within seed noise is reported as
**statistically unresolved**, not as a win.

**Preregistered interaction checks** (added after the main grid, Section documented so
they are not fished for):
- `qk_norm` x `z_loss`
- `norm` x `activation`

**Transfer analysis.**
- Rank correlation of recipe ordering between each pair of sizes.
- Selection regret at each large size for the recipe chosen at each smaller size.
- Per-intervention effect trajectory across sizes: grows / holds / reverses / unresolved.

**Held-out transfer (later, optional).** An AI-deployment corpus may be used as a held-out
evaluation of the selected recipe, never as the primary training corpus.

## 8. Experiment records

Per-run directory `experiments/runs/<run_id>/` is authoritative:

```
resolved_config.yaml   # the exact config used
environment.json       # git sha + dirty, torch/CUDA versions, device, hostname, deps
metrics.jsonl          # streamed: step, train_loss, val_loss, grad_norm, max_logit, lr, tokens
summary.json           # final record (below)
checkpoint.pt          # optional
```

`summary.json` fields:
- run id and **status** (`completed` | `failed` | `running`)
- timestamp
- resolved config
- git SHA and dirty flag
- dataset hash and tokenizer hash
- seed
- torch / CUDA / device info
- total and non-embedding parameter counts
- estimated training FLOPs
- tokens seen
- train/val history summary (also in `metrics.jsonl`)
- final **fixed-budget** validation loss
- bits per byte
- wall-clock time
- tokens/sec
- peak memory
- failure reason, if any

`experiments/manifest.jsonl` holds one index line per completed run pointing at its
directory. Delete-and-regenerate safe; a crashed run is recorded as `failed`, not lost.

## 9. Metrics, grouped

**Quality** — validation loss (nats/token, fixed budget); bits per byte
(tokenizer-independent).

**Stability** — divergence flag; gradient-norm trajectory and peak; maximum logit
magnitude; learning-rate tolerance (does the recipe survive a higher LR?). These explain
*why* z-loss / QK-norm earn their place.

**Efficiency** — tokens/sec; MFU vs device peak; peak memory; loss per FLOP.

The baseline-only scaling fit `L(N)=E+A/N^alpha` is computed as a **secondary** diagnostic
for sanity only, explicitly not used to claim compute-optimality.

## 10. Compute tiers

| tier | model | data | purpose |
|---|---|---|---|
| `cpu_smoke` | ~1M params | TinyShakespeare (~1 MB) | prove the pipeline; plumbing only |
| `1gpu` | 15M-100M | FineWeb-Edu shard(s) | the real transfer study |
| `modal` | up to ~300M | more shards | XL confirmation tier, more seeds |

Same code path across tiers; only config numbers, `dataset`, and `tokenizer_path` change.

## 11. Systems approach (no custom kernel in core)

Profiling and resource accounting, mixed precision, `torch.compile`, SDPA, tokens/sec,
peak memory, and MFU are in scope. A custom Triton kernel is a **clearly labeled
post-project stretch goal**, not part of the core milestones.

## 12. Dependencies

`torch`, `numpy`, `pyyaml`, `requests` (corpus download), `pytest`. Optional `matplotlib`
for analysis plots. No experiment-tracking service; no large dependency stack.

## 13. Testing (bar in ROADMAP)

Reference-vs-SDPA agreement, causal-masking leakage, exact parameter accounting across all
toggles, tokenizer round-trips and determinism, document-split leakage, overfit-one-batch,
and same-seed CPU reproducibility are all required. No test downloads external data.

## 14. Resolved decisions (locked 2026-07-21)

1. **Budget basis:** `D/N = 20` with **N = total parameters** (Chinchilla convention).
   Noted trade-off: at ~15M the embedding table is ~37% of N, so small models spend a
   larger share of their budget on the embedding; accepted for convention-compatibility.
2. **Size tiers:** S ~15M, M ~40M, L ~100M (no XL for now). ~36 GPU-h core (~1.5
   GPU-days, ~$54 on A100). Seeds: S 3, M 3, L 2, plus interaction checks (2 x 3 at S,M).
3. **Systems depth:** resource accounting + `torch.compile` + SDPA + tokens/sec + MFU +
   peak memory. No custom Triton kernel in the core (remains a labeled stretch goal).
4. **Corpus:** FineWeb-Edu. Freeze one byte-level BPE tokenizer (vocab 16,384) on a
   bounded sample. AI-deployment corpus deferred as possible held-out transfer later.

### Approved starting geometries (vocab 16,384, block 512, head_dim 64)

| size | n_layer | n_embd | n_head | total | non-emb | tokens (D=20N) |
|---|---|---|---|---|---|---|
| S ~15M | 6 | 384 | 6 | 16.9M | 10.6M | 338M |
| M ~40M | 8 | 512 | 8 | 33.3M | 24.9M | 666M |
| L ~100M | 12 | 768 | 12 | 97.5M | 85.0M | 1.95B |

M sits at ~33M; bump to `n_layer=10, n_embd=576` for ~49M if a wider size spread is
wanted. These are the M6 defaults, adjustable before the GPU gate.

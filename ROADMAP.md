# nanoscale — roadmap and todos

Milestones are ordered. Each has an acceptance check ("done when"). The **CPU acceptance
gate** must pass before any GPU work, and the **GPU spending gate** must be approved
before any large run. Phases map to CS336 lectures/assignments.

Legend: `[ ]` todo, `[~]` in progress, `[x]` done.

**Progress (2026-07-23):** M0-M5 built, CPU gate passed, M6a protocol-validity change
set landed, and Phase 1 of M6 is complete apart from corpus preparation. 206 tests
green. Corpus pinned (FineWeb-Edu `87f09149`, 4 shards), tokenizer frozen (vocab 16,384,
hash `03be9e0e34d77bec`), `protocol_hash` = `31f6c08926236ff0`, Modal runner built.
Remaining before the pilot: tokenize the corpus once, sized for L. No GPU runs yet.

**Cost note:** the estimate rose from ~$59 to ~$95 (H100) after balancing seeds to 3/3/3
and correcting FLOPs to include the output projection. The L tier is ~87% of it. Against
a $30 balance, S+M fits (~$12) and L does not.

Local commit checkpoints (no push yet):
1. `docs: define proxy-validity study`
2. `build: scaffold configuration and experiment records`
3. `build: tokenizer and data pipeline`
4. `build: transformer and training loop`
5. `build: evaluation and CPU ablation harness`

---

## M0 — Scaffold, config, experiment records  (CS336 Lec 2)
- [ ] `pyproject.toml` installable package; `requirements` pinned loosely; `.gitignore`
- [ ] Package skeleton under `nanoscale/` with docstrings + signatures
- [ ] `config.py`: frozen validated `Config`, `from_yaml`, `override`, CLI parsing,
      early validation (head divisibility, RoPE head-dim, vocab vs dtype, enums, positivity)
- [ ] `config.py`: `n_params`, `n_params_non_embedding`, `flops_per_token`; budget from
      `tokens_per_param`
- [ ] `experiments.py`: run directory, `resolved_config.yaml`, `environment.json`,
      `metrics.jsonl`, `summary.json`, manifest index, git/env/hash capture
- [ ] `configs/cpu_smoke.yaml`, `configs/base.yaml`
- [ ] tests: config validation rejects bad configs; CLI override lands in resolved config;
      run record round-trips

**Done when:** `pip install -e .` works, `pytest` green, a dummy run writes a complete
run directory + manifest index.

---

## M1 — Tokenizer  (CS336 Lec 1, Assignment 1a)
- [ ] Byte-level BPE `train(text, vocab_size, max_bytes)` with deterministic tie-breaking
- [ ] `encode`/`decode`, UTF-8 + arbitrary-byte round-trip
- [ ] Special tokens incl `<|endoftext|>` / document boundary
- [ ] `save`/`load` equivalence; `bytes` fallback; bounded training sample
- [ ] Fertility + compression reporting
- [ ] tests: unicode round-trip, arbitrary-byte round-trip, empty input, special tokens,
      deterministic training/tie-breaking, save/load equivalence, vocab-size constraint

**Done when:** round-trip holds on TinyShakespeare and a frozen tokenizer saves/loads.

---

## M2 — Data pipeline  (CS336 Lec 1, Assignment 1b)
- [ ] Local TXT + JSONL ingestion (configurable text field)
- [ ] Document-level train/val split **before** concatenation, deterministic under seed
- [ ] Memmapped uint16 token storage + `meta.json`
- [ ] Record dataset hash, tokenizer hash, doc counts, token counts, split seed
- [ ] `get_batch`: shifted targets, reproducible under a generator seed
- [ ] `scripts/prepare_data.py` (TinyShakespeare download for smoke)
- [ ] tests: correct shifted targets, deterministic split, no document overlap,
      memmap shape/dtype, reproducible batches under seed

**Done when:** `prepare_data.py --dataset tinyshakespeare` writes bins + meta with hashes
and a seeded batch is reproducible.

---

## M1.5 — Tokenizer study  (CS336 Lec 1)
Compare bytes-only against byte-level BPE and choose the tokenizer to **freeze** before
any architecture experiment. The model-quality arm depends on M3/M4, so this milestone
*executes after the CPU gate* but is numbered here because it fixes an input the recipe
grid holds constant. TinyShakespeare/CPU is plumbing; the real decision is on FineWeb-Edu.

- [x] Efficient BPE trainer: regex pre-tokenization + frequency-weighted unique chunks
- [x] `tokenizer.py`: `utilization`, `segment`, `piece_repr`
- [x] `scripts/tokenizer_study.py`: bytes + BPE at 1K/4K/8K/16K
- [x] Report: compression, fertility, utilization, train time, encode speed, qualitative
      segmentation
- [x] Equal-budget model quality in bits/byte (`--with-model`); decision metric
- [ ] Run on FineWeb-Edu at M6 and **freeze** the winner (`--freeze data/tokenizer.json`)
- [x] tests: utilization/segment, intrinsic metrics, tiling, equal-budget model path

**Done when:** the study runs end to end and, on the real corpus, a single tokenizer is
frozen and hashed for all later experiments. (Intrinsic + CPU model arm validated on
TinyShakespeare; freeze happens at M6.)

Harness validated on TinyShakespeare (800 KB tokenizer sample, 150 KB held out, 50-step
equal-budget models). Plumbing only, single seed, not a finding:

| tokenizer | compression | fertility | util% | train_s | bits/byte | model tok/s |
|---|---|---|---|---|---|---|
| bytes | 1.00 | 5.50 | 24.1 | 0 | 4.133 | 4,770 |
| bpe_1024 | 2.25 | 2.45 | 72.0 | 25 | 3.332 | 3,353 |
| bpe_4096 | 2.79 | 1.97 | 66.5 | 111 | 2.815 | 3,676 |
| bpe_8192 | 2.97 | 1.85 | 47.6 | 154 | 2.696 | 2,986 |
| bpe_16384 | 3.08 | 1.79 | 27.8 | 289 | 2.647 | 1,595 |

Observations to carry into the FineWeb-Edu run: BPE beats bytes decisively, but gains
saturate (8K to 16K buys 0.05 bits/byte for 2x the tokenizer training time and ~2x
slower model throughput); utilization falls off at large vocab, though the 150 KB
held-out slice understates it (rare tokens cannot appear in so little text), so the real
run needs a much larger eval slice before utilization is trusted.

**Decision metric:** lowest equal-budget bits/byte (tokenizer-independent). Compression,
fertility, and utilization are informative but not decisive. "Equal budget" = same model
and same token budget; a denser tokenizer therefore sees more underlying text per token,
which is a real advantage, not a confound.

---

## M3 — Transformer and training loop  (CS336 Lec 2-4, Assignment 1c) — CORE
- [ ] `model.py`: decoder-only pre-norm Transformer; RoPE, RMSNorm, SwiGLU, QK-norm,
      weight tying, z-loss
- [ ] `attention_backend: reference | sdpa`; reference is simple + testable
- [ ] `config` accounting matches actual module params for every toggle combination
- [ ] `train.py`: AdamW, warmup+cosine, grad clip; CPU fp32 / CUDA bf16 autocast
- [ ] Reproducible seeding (Python/NumPy/torch/CUDA); fixed-budget final loss
- [ ] Lightweight checkpoint; NaN/Inf fail-loud + `failed` record
- [ ] tests: output shapes; finite fwd/bwd; causal masking blocks future tokens; tied
      weights share storage; exact param accounting; RoPE + RMSNorm vs small reference;
      reference vs SDPA agreement; z-loss formula; overfit one batch; same-seed CPU
      determinism; short run decreases loss; interrupted run recorded; CLI override in
      resolved config; no external downloads in tests

**Done when:** `python -m nanoscale.train --config configs/cpu_smoke.yaml` trains the
~1M model in minutes, loss decreases, a full run directory is written.

---

## M4 — Evaluation spine  (CS336 Lec 12)
- [ ] `eval.py`: `estimate_loss`, `bits_per_byte`, `generate`
- [ ] Stability probes (grad norm, max logit, divergence flag) streamed to `metrics.jsonl`
- [ ] Bits/byte in `summary.json` for cross-tokenizer fairness

**Done when:** every run logs val loss, bits/byte, and stability probes; a sample prints.

---

## M5 — CPU ablation harness (plumbing)  (Phase 1 wiring only)
- [ ] `configs/base.yaml` = full-stack baseline
- [ ] `scripts/run_ablation.py`: baseline + 6 single-flip variants via `Config.override`,
      grouped quality/stability/efficiency, tagged `ablation`
- [ ] `scripts/make_table.py`: grouped report, **clearly labeled smoke-test plumbing, not
      findings**
- [ ] tests: the 7-config grid runs end to end on CPU

**Done when:** `run_ablation.py` on CPU produces 7 run directories and `make_table.py`
prints a grouped table stamped as smoke-test output.

---

## CPU ACCEPTANCE GATE (must pass before any GPU work)
1. [ ] All tests pass
2. [ ] TinyShakespeare preparation works
3. [ ] ~1M model trains and generates text
4. [ ] Validation loss decreases
5. [ ] Manifest entry + complete run directory produced
6. [ ] 7-config ablation plumbing grid runs on CPU
7. [ ] Table labels the numbers as smoke-test results, not findings

**Then stop and report** (files, tests, smoke metrics, deviations, estimated GPU
hours/cost, proposed exact 15M/40M/100M configs, unresolved decisions).

---

## M6a — Protocol validity change set  (done; protocol still a DRAFT)
Fixes found while treating the first protocol as a draft rather than a plan.

- [x] `target_train_tokens` per scale from the baseline geometry; identical
      tokens/max_steps for every recipe at a scale (tests prove learned positions and
      untied weights no longer receive extra data)
- [x] FLOPs include the vocabulary output projection regardless of weight tying;
      `estimated_*` names keep estimates separate from measured throughput
- [x] Iterable document ingestion, content-hash train/val assignment stable under
      corpus growth, incremental token and hash writing
- [x] Corpora pinned by immutable revision and explicit shard order; unpinned
      FineWeb-Edu config is refused by the loader on purpose
- [x] Deterministic packed stream replacing random-with-replacement sampling; S, M and
      L consume nested prefixes per shared `data_seed`
- [x] Frozen evaluation set keyed to `eval_seed`, independent of the training seed
- [x] `study_id`, `protocol_hash`, `scale_id`, `recipe_id`, `init_seed`, `data_seed`,
      `eval_set_hash` on every run; atomic summary writes
- [x] Resumable 63-run runner (skip completed identities, periodic checkpoints);
      dry-run prints the full matrix and estimated cost
- [x] Analysis written before execution: paired effects, selection regret, selection
      probabilities, descriptive Spearman/Kendall, equivalence-aware classification,
      Markdown/HTML report
- [x] Five-seed S pilot configured (baseline + `no_swiglu`), **not run**
- [x] Interaction cells deferred until the balanced three-seed L grid is funded

---

## M6 — Experimental protocol  (before any GPU spend)
- [x] Pin FineWeb-Edu: revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, 4 shards
- [x] Run M1.5 tokenizer study on FineWeb-Edu; freeze the winner (vocab 16,384,
      hash `03be9e0e34d77bec`) and wire it into `base.yaml`
- [x] Modal runner: CPU prepare into a Volume, batched GPU training, plan-only default
- [ ] **Tokenize the corpus once, sized for L (~2B tokens).** Must cover the largest
      tier that will ever run: `PackedStream` permutes over the block count, so
      appending data later changes the order and breaks nested prefixes
- [ ] Local FineWeb-Edu-format ingestion proven on a small shard, then the streaming adapter
- [ ] `configs/scales/{s,m,l}.yaml` = exact 15M/40M/100M configs; hold constants per DESIGN
- [ ] Derive per-size `max_iters` from `D/N = 20`; verify FLOP-matched budgets
- [ ] Preregister seeds (S:3, M:3, L:>=2, XL:2) and the two interaction checks
- [ ] `analysis/`: rank correlation, selection regret, per-intervention scale trajectory,
      seed dispersion / CIs

**Done when:** the protocol, configs, budgets, seeds, and analysis code exist and are
reviewed. This is the artifact the GPU gate approves.

---

## GPU SPENDING GATE (explicit approval required)
Do not launch large runs until:
- [ ] Estimated GPU hours and cost approved
- [ ] Exact 15M/40M/100M configs approved
- [ ] Corpus shard(s) and token budget approved
- [ ] `D/N` confirmed and XL tier decision made
- [ ] Systems depth chosen (accounting + compile + MFU default)

---

## M7 — Real transfer study  (CS336 Lec 8-9, Assignment 3)
- [ ] Run the full grid at S and M with all seeds; log stability + efficiency
- [ ] Run the grid at L (credible alternatives included, not only S/M winners)
- [ ] Interaction checks at S/M
- [ ] Rank correlation across size pairs; selection regret at L
- [ ] Per-intervention effect trajectory (grows/holds/reverses/unresolved)
- [ ] Secondary baseline scaling fit as a diagnostic only
- [ ] Optional XL confirmation tier if compute permits

**Done when:** rank-transfer, regret, and grouped effect results exist with seed
uncertainty. This is the headline finding.

---

## Follow-on / stretch (out of core; documented, cut first if short on time)

### P8 — Inference  (CS336 Lec 10)
- [ ] KV cache in `generate`; simple quantization; latency vs quality frontier

### P9 — Alignment  (CS336 Lec 15-16, Assignment 5)
- [ ] SFT then DPO on a small preference set; ties to CS329A DPO/GRPO work

### P10 — Custom kernel (stretch, explicitly out of the core project)
- [ ] One Triton kernel (fused GeLU or fused attention) with a before/after tokens/sec

### P11 — Held-out transfer
- [ ] Evaluate the selected recipe on an AI-deployment corpus as held-out transfer only

---

## Suggested order of operations
1. M0 -> M5 in sequence, everything `cpu_smoke` and fast.
2. Pass the CPU acceptance gate; stop and report.
3. M6 protocol; resolve DESIGN Section 14 decisions.
4. Pass the GPU spending gate.
5. M7 real study, then optional follow-ons.

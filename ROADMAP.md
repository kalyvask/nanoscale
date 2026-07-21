# nanoscale — roadmap and todos

Milestones are ordered. Each has an acceptance check ("done when"). The **CPU acceptance
gate** must pass before any GPU work, and the **GPU spending gate** must be approved
before any large run. Phases map to CS336 lectures/assignments.

Legend: `[ ]` todo, `[~]` in progress, `[x]` done.

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

## M6 — Experimental protocol  (before any GPU spend)
- [ ] Freeze the real-study tokenizer (vocab ~16,384) on a bounded sample; record its hash
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

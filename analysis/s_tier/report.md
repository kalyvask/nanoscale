# nanoscale transfer analysis

Metric: `final_val_loss`. Equivalence margin: 0.01 nats/token. Runs: 21.

Protocol, evaluation set and tokenizer are consistent across runs.


## Seed noise (the floor every effect must clear)

| scale | n | baseline mean | sd | range |
|---|---|---|---|---|
| S | 3 | 3.8198 | 0.0047 | 0.0093 |

## Paired effects vs baseline

Positive delta means the recipe is worse than baseline, so the removed component was helping. Pairing is by seed.


### Scale S

| recipe | n | mean delta | 95% CI | verdict |
|---|---|---|---|---|
| no_rope | 3 | 0.0666 | [0.0565, 0.0738] | removing it hurts |
| no_qknorm | 3 | 0.0446 | [0.0415, 0.0497] | removing it hurts |
| no_swiglu | 3 | 0.0259 | [0.0227, 0.0289] | removing it hurts |
| no_rmsnorm | 3 | 0.0020 | [-0.0023, 0.0049] | practically equal |
| untied | 3 | -0.0139 | [-0.0234, -0.0041] | unresolved |
| no_zloss | 3 | -0.0155 | [-0.0171, -0.0146] | removing it helps |

## Selection regret

Pick the best recipe at the small scale, then read its loss at the large scale. Zero regret means the cheap experiment chose correctly.

| small | large | chosen | best | regret | correct |
|---|---|---|---|---|---|

## Selection probability (seed resampling)

| small | large | P(small pick == large best) | mean regret | 95% CI |
|---|---|---|---|---|

## Rank transfer (descriptive)

| small | large | recipes | Spearman | Kendall tau | note |
|---|---|---|---|---|---|

## Effect trajectory across scale

| recipe | trajectory | per-scale mean deltas |
|---|---|---|
| no_qknorm | insufficient_scales | S=0.0446 |
| no_rmsnorm | insufficient_scales | S=0.0020 |
| no_rope | insufficient_scales | S=0.0666 |
| no_swiglu | insufficient_scales | S=0.0259 |
| no_zloss | insufficient_scales | S=-0.0155 |
| untied | insufficient_scales | S=-0.0139 |

---

One-at-a-time flips measure conditional effects given the rest of the baseline, not independent contributions. Rank correlations over a handful of recipes are descriptive, not tests. `unresolved` means the data could not distinguish an effect from none; it is not evidence of no effect.

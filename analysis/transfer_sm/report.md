# nanoscale transfer analysis

Metric: `final_val_loss`. Equivalence margin: 0.01 nats/token. Runs: 35.

Protocol, evaluation set and tokenizer are consistent across runs.


## Seed noise (the floor every effect must clear)

| scale | n | baseline mean | sd | range |
|---|---|---|---|---|
| S | 3 | 3.8198 | 0.0047 | 0.0093 |
| M | 2 | 3.4619 | 0.0021 | 0.0029 |

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

### Scale M

| recipe | n | mean delta | 95% CI | verdict |
|---|---|---|---|---|
| no_rope | 2 | 0.0535 | [0.0533, 0.0537] | removing it hurts |
| no_qknorm | 2 | 0.0225 | [0.0198, 0.0251] | removing it hurts |
| no_swiglu | 2 | 0.0117 | [0.0116, 0.0118] | removing it hurts |
| no_rmsnorm | 2 | -0.0002 | [-0.0012, 0.0009] | practically equal |
| untied | 2 | -0.0061 | [-0.0085, -0.0038] | practically equal |
| no_zloss | 2 | -0.0130 | [-0.0136, -0.0124] | removing it helps |

## Selection regret

Pick the best recipe at the small scale, then read its loss at the large scale. Zero regret means the cheap experiment chose correctly.

| small | large | chosen | best | regret | correct |
|---|---|---|---|---|---|
| S | M | no_zloss | no_zloss | 0.0000 | True |

## Selection probability (seed resampling)

| small | large | P(small pick == large best) | mean regret | 95% CI |
|---|---|---|---|---|
| S | M | 0.662 | 0.0023 | [0.0000, 0.0107] |

## Rank transfer (descriptive)

| small | large | recipes | Spearman | Kendall tau | note |
|---|---|---|---|---|---|
| S | M | 7 | 0.964 | 0.905 | underpowered (few recipes) |

## Effect trajectory across scale

| recipe | trajectory | per-scale mean deltas |
|---|---|---|
| no_qknorm | shrinks | S=0.0446, M=0.0225 |
| no_rmsnorm | reverses | S=0.0020, M=-0.0002 |
| no_rope | shrinks | S=0.0666, M=0.0535 |
| no_swiglu | shrinks | S=0.0259, M=0.0117 |
| no_zloss | holds | S=-0.0155, M=-0.0130 |
| untied | unresolved | S=-0.0139, M=-0.0061 |

---

One-at-a-time flips measure conditional effects given the rest of the baseline, not independent contributions. Rank correlations over a handful of recipes are descriptive, not tests. `unresolved` means the data could not distinguish an effect from none; it is not evidence of no effect.

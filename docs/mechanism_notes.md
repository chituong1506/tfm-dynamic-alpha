# Mechanisms and estimators (Phase 2)

## History format

Estimators consume a mapping from block index to object ID to a sequence of
`(priority, compute)` observations. Blocks are processed in ascending index
order. `DynamicAlpha.params_for_block(b, history)` filters out blocks with an
index greater than or equal to `b`, so the parameters for block `b` use only
history through block `b-1`.

An object with no observations uses the configured prior `pi0` (default 1.0).
When an object list is not supplied to a mechanism, object IDs are inferred
from history; pass `object_ids` to return parameters for known but not-yet-seen
objects as well.

## Priority estimators

- `last_block_weighted_mean` computes `sum(priority * compute) / sum(compute)`
  for the most recent block containing observations for that object.
- `weighted_median_K(K)` returns an estimator that computes the compute-weighted
  median over observations from the last `K` history blocks. The selected
  median is the first priority whose cumulative compute reaches at least half
  of total compute.
- `ema_trimmed_mean(K, trim_frac, lam)` returns an estimator that takes the
  last `K` data-bearing blocks, removes `floor(n * trim_frac)` highest-priority
  observations in each block (but retains at least one), computes each
  remaining block's compute-weighted mean, then applies
  `EMA_b = lam * mean_b + (1-lam) * EMA_(b-1)`. A zero trim fraction or a
  one-observation block leaves that block untrimmed.

All observations require finite priorities of at least 1 and finite positive
compute. `K` must be positive, `trim_frac` must be in `[0, 1)`, and `lam` must
be in `[0, 1]`.

## Mechanisms

`FixedAlpha(alpha, eta)` returns the same pair for each object.
`DynamicAlpha(alpha_max, kappa, eta0, gamma, estimator)` calculates per object:

```text
eta_o   = min(eta0, alpha_max * (1-gamma) / (kappa * pi_hat_o * gamma))
alpha_o = min(alpha_max, kappa * pi_hat_o * eta_o * gamma / (1-gamma))
```

The estimator supplies `pi_hat_o` from past observations; estimates must be
finite and at least 1. `0 < gamma < 1`, `0 < alpha_max <= 1`, and `eta0 >= 0`.
By default, `kappa >= 1`; values `0 < kappa < 1` require the explicit
`allow_kappa_below_one=True` opt-in. The formula ensures `alpha_o <= alpha_max`
and `eta_o <= eta0`. An estimator returning an invalid estimate raises
`ValueError` rather than silently falling back to the prior.

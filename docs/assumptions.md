# Assumptions used in Phase 1

- The default unused-fee convention is `unused_fee_includes_priority_and_compute = True`, matching Section 1.2 of the project brief.
- When `unused_fee_includes_priority_and_compute = False`, the unused-object charge is `alpha * p_o` rather than `alpha * pi * t * p_o`.
- `p_min` defaults to `0.0` unless a caller explicitly provides a different floor.
- The simulator stores object utilization in the `declared` mode unless a caller explicitly requests `used` mode.
- These assumptions are implemented as explicit config-level options for later experiments.
- Workload arrivals are sampled independently per primary object from the configured price-elastic Poisson rates. A scalar base rate is apportioned by the configured Zipf popularity (normalized to preserve the mean rate); a per-object rate tuple overrides this apportionment. Declared object sets, compute, priorities, and execution outcomes are then pre-sampled once and reused across mechanisms.
- The default calibrated priority distribution is a two-component mixture (a point mass at 1.2 and a lognormal component) because a single lognormal is not assumed to meet all three target statistics.

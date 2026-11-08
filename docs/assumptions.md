# Assumptions used in Phase 1

- The default unused-fee convention is `unused_fee_includes_priority_and_compute = True`, matching Section 1.2 of the project brief.
- When `unused_fee_includes_priority_and_compute = False`, the unused-object charge is `alpha * p_o` rather than `alpha * pi * t * p_o`.
- `p_min` defaults to `0.0` unless a caller explicitly provides a different floor.
- The simulator stores object utilization in the `declared` mode unless a caller explicitly requests `used` mode.
- These assumptions are implemented as explicit config-level options so they remain easy to change in later Phase 2 work.

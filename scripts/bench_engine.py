"""Time 2,000-block simulations with fixed and dynamic alpha."""

from __future__ import annotations

import time

import numpy as np

from tfmsim.engine import SimulationConfig, simulate
from tfmsim.estimators import last_block_weighted_mean
from tfmsim.mechanisms import DynamicAlpha, FixedAlpha
from tfmsim.workload import WorkloadConfig, generate_workload


def main() -> None:
    n_blocks = 2000
    workload = generate_workload(
        WorkloadConfig(
            n_blocks=n_blocks,
            n_objects=3,
            lambda_per_object=1.0,
            max_demand_multiplier=2.0,
        ),
        np.random.default_rng(20261108),
    )
    config = SimulationConfig(
        object_ids=(0, 1, 2),
        target_utilization={0: 4.0, 1: 4.0, 2: 4.0},
        initial_prices={0: 1.0, 1: 1.0, 2: 1.0},
        max_compute=6.0,
        cap_factor=2.0,
        gamma=0.5,
        p_min=0.001,
        history_window=None,
    )
    mechanisms = (
        ("FixedAlpha", FixedAlpha(0.5, 0.125)),
        (
            "DynamicAlpha",
            DynamicAlpha(1.0, 1.0, 0.125, 0.5, last_block_weighted_mean),
        ),
    )
    for label, mechanism in mechanisms:
        started = time.perf_counter()
        result = simulate(workload, mechanism, config)
        elapsed = time.perf_counter() - started
        print(f"{label}: {elapsed:.6f} seconds for {len(result.records)} blocks")


if __name__ == "__main__":
    main()

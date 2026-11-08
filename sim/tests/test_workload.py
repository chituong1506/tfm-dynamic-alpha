import numpy as np

from tfmsim.workload import WorkloadConfig, expected_arrivals, generate_workload


def test_workload_is_deterministic_and_samples_used_sets_once():
    config = WorkloadConfig(n_blocks=8, n_objects=3, lambda_per_object=5.0)
    first = generate_workload(config, np.random.default_rng(42))
    second = generate_workload(config, np.random.default_rng(42))
    assert first == second
    for block in first:
        for tx in block.txs:
            assert tx.used is not None
            assert tx.used.issubset(tx.declared)
            assert (tx.declared - tx.contingent).issubset(tx.used)


def test_demand_elasticity_reduces_expected_arrivals_when_price_rises():
    config = WorkloadConfig(n_blocks=1, n_objects=2, lambda_per_object=10, epsilon=1.0)
    low_prices = expected_arrivals(config, {0: 1.0, 1: 1.0})
    high_prices = expected_arrivals(config, {0: 2.0, 1: 2.0})
    assert all(high_prices[obj] < low_prices[obj] for obj in low_prices)
    assert all(high_prices[obj] == low_prices[obj] / 2 for obj in low_prices)


def test_candidate_pools_are_identical_with_equal_seeds():
    config = WorkloadConfig(
        n_blocks=1000,
        n_objects=2,
        lambda_per_object=20.0,
        epsilon=1.0,
        min_declared_objects=1,
        max_declared_objects=1,
    )
    first = generate_workload(config, np.random.default_rng(123))
    second = generate_workload(config, np.random.default_rng(123))
    assert first == second
    assert all(0 <= tx.u <= 1 for block in first for tx in block.txs)


def test_candidate_intensity_is_scaled_by_multiplier():
    config = WorkloadConfig(
        n_blocks=500,
        n_objects=2,
        lambda_per_object=10.0,
        max_demand_multiplier=2.0,
    )
    workload = generate_workload(config, np.random.default_rng(123))
    observed = sum(len(block.txs) for block in workload)
    assert observed > config.n_blocks * sum(expected_arrivals(
        WorkloadConfig(n_blocks=1, n_objects=2, lambda_per_object=10.0),
        {0: 1.0, 1: 1.0},
    ).values())


def test_declared_object_count_distribution_is_configurable():
    config = WorkloadConfig(
        n_blocks=3,
        n_objects=2,
        lambda_per_object=10.0,
        min_declared_objects=1,
        max_declared_objects=2,
        declared_count_probabilities=(1.0, 0.0),
    )
    workload = generate_workload(config, np.random.default_rng(77))
    assert all(len(tx.declared) == 1 for block in workload for tx in block.txs)

import numpy as np
import pytest

from tfmsim.engine import SimulationConfig, simulate
from tfmsim.estimators import last_block_weighted_mean
from tfmsim.mechanisms import DynamicAlpha, FixedAlpha
from tfmsim.workload import WorkloadConfig, generate_workload


def _config():
    return SimulationConfig(
        object_ids=(0, 1),
        target_utilization={0: 3.0, 1: 3.0},
        initial_prices={0: 1.0, 1: 1.0},
        max_compute=4.0,
        cap_factor=1.5,
        gamma=0.5,
        p_min=0.01,
    )


def test_engine_is_deterministic_for_equal_seeds_and_respects_capacity():
    workload_config = WorkloadConfig(n_blocks=10, n_objects=2, lambda_per_object=8.0)
    first_workload = generate_workload(workload_config, np.random.default_rng(5))
    second_workload = generate_workload(workload_config, np.random.default_rng(5))
    result1 = simulate(first_workload, FixedAlpha(0.5, 0.125), _config())
    result2 = simulate(second_workload, FixedAlpha(0.5, 0.125), _config())
    assert result1 == result2
    for block in result1.blocks:
        total_compute = sum(tx.compute for tx in block.txs)
        assert total_compute <= 4.0
        for object_id, target in _config().target_utilization.items():
            used = sum(tx.compute for tx in block.txs if object_id in tx.declared)
            assert used <= 1.5 * target


def test_fees_equal_scheduler_revenue_plus_burned():
    workload = generate_workload(
        WorkloadConfig(n_blocks=5, n_objects=2, lambda_per_object=8.0),
        np.random.default_rng(20),
    )
    result = simulate(workload, FixedAlpha(0.5, 0.125), _config())
    for record in result.records:
        assert record.fees_paid == pytest.approx(record.scheduler_revenue + record.burned)


def test_engine_passes_full_object_ids_and_only_prior_history():
    class ObservingMechanism:
        def __init__(self):
            self.calls = []

        def params_for_block(self, b, history, *, object_ids=None):
            self.calls.append((b, dict(history), object_ids))
            return {obj: (0.5, 0.125) for obj in object_ids}

    workload = generate_workload(
        WorkloadConfig(n_blocks=4, n_objects=2, lambda_per_object=5.0),
        np.random.default_rng(21),
    )
    mechanism = ObservingMechanism()
    result = simulate(workload, mechanism, _config())
    for block_index, history, object_ids in mechanism.calls:
        assert object_ids == (0, 1)
        assert all(recorded_block < block_index for recorded_block in history)
        assert set(history) == set(range(block_index))
        assert all(set(per_object) == {0, 1} for per_object in history.values())
    assert mechanism.calls[0][1] == {}
    assert result.history.keys() == set(range(4))


def test_dynamic_alpha_engine_smoke():
    workload = generate_workload(
        WorkloadConfig(n_blocks=4, n_objects=2, lambda_per_object=5.0),
        np.random.default_rng(22),
    )
    mechanism = DynamicAlpha(1.0, 1.0, 0.125, 0.5, last_block_weighted_mean)
    result = simulate(workload, mechanism, _config())
    assert len(result.records) == 4
    assert all(0 <= record.mean_alpha <= 1 for record in result.records)

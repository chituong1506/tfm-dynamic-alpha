import numpy as np
import pytest

from tfmsim.engine import SimulationConfig, create_initial_state, simulate, step
from tfmsim.estimators import last_block_weighted_mean
from tfmsim.mechanisms import DynamicAlpha, FixedAlpha
from tfmsim.types import Tx
from tfmsim.workload import WorkloadConfig, generate_workload


def _config(**kwargs):
    values = dict(
        object_ids=(0, 1),
        target_utilization={0: 3.0, 1: 3.0},
        initial_prices={0: 1.0, 1: 1.0},
        max_compute=4.0,
        cap_factor=1.5,
        gamma=0.5,
        p_min=0.01,
    )
    values.update(kwargs)
    return SimulationConfig(
        **values,
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
    assert all(set(record.alphas) == {0, 1} and set(record.etas) == {0, 1} for record in result.records)
    assert all(set(record.pi_hat) == {0, 1} for record in result.records if record.pi_hat is not None)


def test_step_matches_simulate_and_snapshot_resume_is_identical():
    workload = generate_workload(
        WorkloadConfig(n_blocks=8, n_objects=2, lambda_per_object=5.0),
        np.random.default_rng(81),
    )
    mechanism = FixedAlpha(0.5, 0.125)
    config = _config()
    uninterrupted = simulate(workload, mechanism, config)

    state = create_initial_state(config)
    for candidate_block in workload:
        state, _, _ = step(state, candidate_block.txs, mechanism, config)
    stepped = simulate([], mechanism, config, initial_state=state)
    assert stepped == uninterrupted

    snapshot = create_initial_state(config)
    for candidate_block in workload[:4]:
        snapshot, _, _ = step(snapshot, candidate_block.txs, mechanism, config)
    continued = simulate(workload[4:], mechanism, config, initial_state=snapshot)
    assert continued == uninterrupted


def test_step_and_initial_state_do_not_share_mutable_history():
    config = _config()
    original = create_initial_state(config)
    tx = Tx(
        id="history",
        declared=frozenset({0}),
        used=None,
        u=0.0,
    )
    advanced, _, _ = step(original, [tx], FixedAlpha(0.5, 0.125), config)
    advanced.history[0][0].append((8.0, 1.0))
    advanced.price_history[0][0] = 50.0
    assert original.history == {}
    assert original.price_history[0][0] == 1.0

    copied = simulate([], FixedAlpha(0.5, 0.125), config, initial_state=advanced)
    copied.history[0][0].append((9.0, 1.0))
    assert (9.0, 1.0) not in advanced.history[0][0]


def test_price_thinning_reduces_included_transactions_and_zero_elasticity_keeps_all():
    workload = generate_workload(
        WorkloadConfig(
            n_blocks=1,
            n_objects=2,
            lambda_per_object=200.0,
            max_demand_multiplier=2.0,
            epsilon=1.0,
            min_declared_objects=1,
            max_declared_objects=1,
        ),
        np.random.default_rng(99),
    )
    low_config = _config(
        object_ids=(0, 1),
        target_utilization={0: 1000.0, 1: 1000.0},
        initial_prices={0: 1.0, 1: 1.0},
        reference_prices={0: 1.0, 1: 1.0},
        max_compute=10000.0,
        cap_factor=10.0,
        max_demand_multiplier=2.0,
        epsilon=1.0,
    )
    high_config = _config(
        object_ids=(0, 1),
        target_utilization={0: 1000.0, 1: 1000.0},
        initial_prices={0: 2.0, 1: 2.0},
        reference_prices={0: 1.0, 1: 1.0},
        max_compute=10000.0,
        cap_factor=10.0,
        max_demand_multiplier=2.0,
        epsilon=1.0,
    )
    low = simulate(workload, FixedAlpha(0.5, 0.0), low_config)
    high = simulate(workload, FixedAlpha(0.5, 0.0), high_config)
    assert high.records[0].tx_count < low.records[0].tx_count

    no_elasticity_workload = generate_workload(
        WorkloadConfig(
            n_blocks=1,
            n_objects=2,
            lambda_per_object=50.0,
            epsilon=0.0,
            max_demand_multiplier=1.0,
            min_declared_objects=1,
            max_declared_objects=1,
        ),
        np.random.default_rng(1234),
    )
    all_config = _config(
        object_ids=(0, 1),
        target_utilization={0: 10000.0, 1: 10000.0},
        initial_prices={0: 10.0, 1: 10.0},
        reference_prices={0: 1.0, 1: 1.0},
        max_compute=100000.0,
        cap_factor=10.0,
        epsilon=0.0,
        max_demand_multiplier=1.0,
    )
    all_kept = simulate(no_elasticity_workload, FixedAlpha(0.5, 0.0), all_config)
    assert all_kept.records[0].tx_count == len(no_elasticity_workload[0].txs)


def test_mechanism_gamma_must_match_config():
    workload = generate_workload(
        WorkloadConfig(
            n_blocks=1,
            n_objects=1,
            lambda_per_object=1.0,
            max_declared_objects=1,
        ),
        np.random.default_rng(1),
    )
    mechanism = DynamicAlpha(1.0, 1.0, 0.125, 0.4, last_block_weighted_mean)
    with pytest.raises(ValueError, match="gamma"):
        simulate(workload, mechanism, _config(object_ids=(0,), target_utilization={0: 3.0}, initial_prices={0: 1.0}))


def test_p_min_defaults_to_positive_fraction_and_rejects_zero_unless_opted_in():
    defaulted = SimulationConfig(
        object_ids=(0,),
        target_utilization={0: 1.0},
        initial_prices={0: 2.0},
        max_compute=1.0,
        cap_factor=1.0,
    )
    assert defaulted.p_min == {0: pytest.approx(2e-6)}
    with pytest.raises(ValueError, match="p_min"):
        _config(p_min=0.0)
    explicit = _config(p_min=0.0, allow_zero_p_min=True)
    assert explicit.p_min == 0.0

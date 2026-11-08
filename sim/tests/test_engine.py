import numpy as np
import pytest

from tfmsim.block import build_block
from tfmsim.engine import SimulationConfig, create_initial_state, simulate, step
from tfmsim.estimators import last_block_weighted_mean
from tfmsim.fees import burn_split, f_act, f_att, f_ui
from tfmsim.mechanisms import DynamicAlpha, FixedAlpha
from tfmsim.pricing import update_prices, utilization
from tfmsim.types import ObjectState, Tx
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
    step_records = []
    for candidate_block in workload:
        state, record, _ = step(state, candidate_block.txs, mechanism, config)
        step_records.append(record)
    assert tuple(step_records) == uninterrupted.records
    assert state.prices == uninterrupted.final_state.prices
    assert state.next_block_index == uninterrupted.final_state.next_block_index

    snapshot = create_initial_state(config)
    prefix_records = []
    for candidate_block in workload[:4]:
        snapshot, record, _ = step(snapshot, candidate_block.txs, mechanism, config)
        prefix_records.append(record)
    continued = simulate(workload[4:], mechanism, config, initial_state=snapshot)
    assert continued.records == uninterrupted.records[4:]
    assert continued.price_history[0] == uninterrupted.price_history[4]
    assert continued.price_history[-1] == uninterrupted.price_history[-1]
    assert continued.final_state.prices == uninterrupted.final_state.prices


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
    advanced.history[99] = {0: ()}
    advanced.prices = (50.0, 1.0)
    assert original.history == {}
    assert original.prices[0] == 1.0

    copied = simulate([], FixedAlpha(0.5, 0.125), config, initial_state=advanced)
    copied.final_state.history[100] = {0: ()}
    assert 99 in advanced.history
    assert 100 not in advanced.history


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


def test_history_window_bounds_state_and_preserves_recent_estimates():
    workload = generate_workload(
        WorkloadConfig(n_blocks=7, n_objects=2, lambda_per_object=5.0),
        np.random.default_rng(811),
    )
    config = _config(history_window=3)
    result = simulate(
        workload,
        DynamicAlpha(1.0, 1.0, 0.125, 0.5, last_block_weighted_mean),
        config,
    )
    assert set(result.final_state.history) == {4, 5, 6}
    assert len(result.final_state.history) == 3


@pytest.mark.parametrize(
    "mechanism_factory",
    [
        lambda: FixedAlpha(0.5, 0.125),
        lambda: DynamicAlpha(1.0, 1.0, 0.125, 0.5, last_block_weighted_mean),
    ],
)
def test_results_match_unbounded_history_with_equivalent_candidate_input(mechanism_factory):
    workload = generate_workload(
        WorkloadConfig(
            n_blocks=12,
            n_objects=2,
            lambda_per_object=5.0,
            epsilon=0.0,
            max_demand_multiplier=1.0,
        ),
        np.random.default_rng(812),
    )
    result = simulate(workload, mechanism_factory(), _config())
    assert len(result.records) == len(workload)
    assert [record.index for record in result.records] == list(range(len(workload)))
    assert len(result.history) == len(workload)


@pytest.mark.parametrize(
    "mechanism_factory",
    [
        lambda: FixedAlpha(0.5, 0.125),
        lambda: DynamicAlpha(1.0, 1.0, 0.125, 0.5, last_block_weighted_mean),
    ],
)
def test_outputs_match_pre_refactor_full_history_reference(mechanism_factory):
    workload = generate_workload(
        WorkloadConfig(
            n_blocks=6,
            n_objects=2,
            lambda_per_object=4.0,
            epsilon=0.0,
            max_demand_multiplier=1.0,
        ),
        np.random.default_rng(813),
    )
    config = _config(epsilon=0.0, max_demand_multiplier=1.0)
    mechanism = mechanism_factory()
    actual = simulate(workload, mechanism, config)

    states = {
        obj: ObjectState(
            obj,
            config.initial_prices[obj],
            config.target_utilization[obj],
            p_min=config.p_min[obj] if isinstance(config.p_min, dict) else config.p_min,
        )
        for obj in config.object_ids
    }
    history = {}
    expected_records = []
    expected_prices = [{obj: states[obj].price for obj in config.object_ids}]
    for block_index, candidate_block in enumerate(workload):
        prices = {obj: states[obj].price for obj in config.object_ids}
        params = mechanism.params_for_block(
            block_index,
            history,
            object_ids=config.object_ids,
        )
        alpha = {obj: params[obj][0] for obj in config.object_ids}
        eta = {obj: params[obj][1] for obj in config.object_ids}
        built = build_block(
            candidate_block.txs,
            index=block_index,
            max_compute=config.max_compute,
            target_utilization=config.target_utilization,
            cap_factor=config.cap_factor,
        )
        fees = [f_act(tx, prices, alpha) for tx in built.txs]
        ur_total = sum(fee - f_ui(tx, prices) for fee, tx in zip(fees, built.txs))
        sr_total = sum(
            f_att(tx, prices) - fee
            for fee, tx in zip(fees, built.txs)
        )
        revenue_burn = [burn_split(fee, config.gamma) for fee in fees]
        used = utilization(built, mode=config.utilization_mode, object_ids=config.object_ids)
        expected_records.append(
            (
                sum(fees),
                ur_total,
                sr_total,
                sum(split[0] for split in revenue_burn),
                sum(split[1] for split in revenue_burn),
                used,
                alpha,
                eta,
            )
        )
        history[block_index] = {obj: [] for obj in config.object_ids}
        for tx in built.txs:
            for obj in tx.declared:
                history[block_index][obj].append((tx.priority, tx.compute))
        states = update_prices(
            {
                obj: ObjectState(
                    obj,
                    states[obj].price,
                    states[obj].target_utilization,
                    utilization=used[obj],
                    p_min=states[obj].p_min,
                )
                for obj in config.object_ids
            },
            eta=eta,
            price_update=config.price_update,
            p_min=config.p_min,
        )
        expected_prices.append({obj: states[obj].price for obj in config.object_ids})

    for expected, observed in zip(expected_records, actual.records, strict=True):
        fees, ur_total, sr_total, revenue, burned, used, alpha, eta = expected
        assert observed.fees_paid == pytest.approx(fees)
        assert observed.user_risk == pytest.approx(ur_total)
        assert observed.scheduler_risk == pytest.approx(sr_total)
        assert observed.scheduler_revenue == pytest.approx(revenue)
        assert observed.burned == pytest.approx(burned)
        assert observed.utilization == used
        assert observed.alphas == alpha
        assert observed.etas == eta
    assert actual.price_history == tuple(expected_prices)
    expected_history = {
        block_index: {
            object_id: tuple(observations)
            for object_id, observations in objects.items()
        }
        for block_index, objects in history.items()
    }
    assert actual.history == expected_history


def test_forced_tx_bypasses_filter_and_capacity_and_is_accounted():
    config = SimulationConfig(
        object_ids=(0,),
        target_utilization={0: 1.0},
        initial_prices={0: 2.0},
        reference_prices={0: 1.0},
        max_compute=1.0,
        cap_factor=1.0,
        gamma=0.5,
        epsilon=1.0,
        max_demand_multiplier=2.0,
        p_min=0.01,
    )
    candidate = Tx(
        id="candidate",
        compute=1.0,
        priority=2.0,
        declared=frozenset({0}),
        used=None,
        u=0.0,
    )
    forced = Tx(
        id="shill",
        sender_type="shill",
        compute=1.0,
        priority=1.0,
        declared=frozenset({0}),
        contingent=frozenset({0}),
        used=frozenset(),
        u=1.0,
    )
    mechanism = FixedAlpha(0.5, 0.125)
    state = create_initial_state(config)
    new_state, record, tx_records = step(
        state,
        [candidate],
        mechanism,
        config,
        forced_txs=[forced],
    )
    assert [tx.id for tx in record.included_txs] == ["candidate", "shill"]
    assert record.tx_count == 2
    assert record.fees_paid == pytest.approx(2.0 * 1.0 * 2.0 + 0.5 * 1.0 * 2.0)
    assert record.utilization[0] == pytest.approx(2.0)
    assert new_state.history[0][0] == ((2.0, 1.0), (1.0, 1.0))
    shill_record = next(item for item in tx_records if item.tx_id == "shill")
    assert shill_record.sender_type == "shill"
    assert shill_record.f_act == pytest.approx(1.0)
    assert shill_record.n_unused == 1


def test_forced_txs_can_displace_only_when_configured():
    candidate = Tx(
        id="candidate",
        compute=1.0,
        priority=2.0,
        declared=frozenset({0}),
        used=None,
        u=0.0,
    )
    forced = Tx(
        id="forced",
        sender_type="shill",
        compute=1.0,
        priority=3.0,
        declared=frozenset({0}),
        used=None,
        u=1.0,
    )
    config = SimulationConfig(
        object_ids=(0,),
        target_utilization={0: 1.0},
        initial_prices={0: 1.0},
        max_compute=1.0,
        cap_factor=1.0,
        shill_displaces=True,
    )
    _, record, _ = step(
        create_initial_state(config),
        [candidate],
        FixedAlpha(0.5, 0.125),
        config,
        forced_txs=[forced],
    )
    assert tuple(tx.id for tx in record.included_txs) == ("forced",)


def test_simulate_forced_by_block_and_empty_forced_is_bit_identical():
    workload = generate_workload(
        WorkloadConfig(
            n_blocks=3,
            n_objects=2,
            lambda_per_object=5.0,
            epsilon=0.0,
            max_demand_multiplier=1.0,
        ),
        np.random.default_rng(814),
    )
    config = _config(epsilon=0.0, max_demand_multiplier=1.0)
    mechanism = FixedAlpha(0.5, 0.125)
    baseline = simulate(workload, mechanism, config)
    explicitly_empty = simulate(workload, mechanism, config, forced_by_block={})
    assert explicitly_empty == baseline

    forced = Tx(
        id="forced-scheduler",
        sender_type="shill",
        declared=frozenset({0}),
        used=None,
        u=1.0,
    )
    with_forced = simulate(
        workload[:1],
        mechanism,
        config,
        forced_by_block={0: [forced]},
    )
    assert any(tx.id == "forced-scheduler" for tx in with_forced.blocks[0].txs)
    assert any(tx_record.sender_type == "shill" for tx_record in with_forced.tx_records)

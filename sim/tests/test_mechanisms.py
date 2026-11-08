import random

import pytest

from tfmsim.estimators import last_block_weighted_mean
from tfmsim.mechanisms import DynamicAlpha, FixedAlpha


@pytest.mark.parametrize(
    ("pi_hat", "expected"),
    [
        (1.2, (0.15, 1 / 8)),
        (5.0, (0.625, 1 / 8)),
        (16.0, (1.0, 1 / 16)),
    ],
)
def test_dynamic_alpha_expected_values(pi_hat, expected):
    history = {0: {0: [(pi_hat, 1.0)]}}
    mechanism = DynamicAlpha(1.0, 1.0, 1 / 8, 0.5, last_block_weighted_mean)
    alpha, eta = mechanism.params_for_block(1, history)[0]
    assert (alpha, eta) == pytest.approx(expected)


def test_dynamic_alpha_bounds_over_random_inputs():
    rng = random.Random(573)
    for _ in range(250):
        alpha_max = rng.uniform(0.01, 1.0)
        kappa = rng.uniform(1.0, 5.0)
        eta0 = rng.uniform(0.0, 1.0)
        gamma = rng.uniform(0.01, 0.99)
        pi_hat = rng.uniform(1.0, 100.0)
        history = {0: {0: [(pi_hat, 1.0)]}}
        mechanism = DynamicAlpha(
            alpha_max,
            kappa,
            eta0,
            gamma,
            last_block_weighted_mean,
        )
        alpha, eta = mechanism.params_for_block(1, history)[0]
        assert 0.0 <= alpha <= alpha_max
        assert 0.0 <= eta <= eta0


def test_dynamic_alpha_validates_parameters_and_estimate():
    with pytest.raises(ValueError):
        DynamicAlpha(1.1, 1.0, 0.1, 0.5, last_block_weighted_mean)
    with pytest.raises(ValueError):
        DynamicAlpha(1.0, 0.5, 0.1, 0.5, last_block_weighted_mean)
    opted_in = DynamicAlpha(
        1.0,
        0.5,
        0.1,
        0.5,
        last_block_weighted_mean,
        allow_kappa_below_one=True,
    )
    assert opted_in.kappa == pytest.approx(0.5)
    with pytest.raises(ValueError):
        DynamicAlpha(1.0, 1.0, 0.1, 1.0, last_block_weighted_mean)
    with pytest.raises(ValueError):
        DynamicAlpha(1.0, 1.0, 0.1, 0.5, lambda history, obj, pi0: 0.9).params_for_block(1, {0: {0: [(2.0, 1.0)]}})


def test_fixed_alpha_parameters_and_empty_configured_objects():
    mechanism = FixedAlpha(0.5, 0.125, object_ids=(0, 1))
    assert mechanism.params_for_block(4, {}) == {
        0: (0.5, 0.125),
        1: (0.5, 0.125),
    }


def test_dynamic_alpha_uses_history_before_current_block_only():
    history = {
        0: {0: [(2.0, 1.0)]},
        1: {0: [(16.0, 1.0)]},
    }
    mechanism = DynamicAlpha(1.0, 1.0, 1 / 8, 0.5, last_block_weighted_mean)
    assert mechanism.params_for_block(1, history)[0] == pytest.approx((0.25, 1 / 8))

import pytest

from tfmsim.estimators import ema_trimmed_mean, last_block_weighted_mean, weighted_median_K


def test_last_block_compute_weighted_mean():
    history = {
        0: {0: [(2.0, 1.0), (4.0, 3.0)]},
        1: {0: [(3.0, 1.0), (5.0, 1.0)]},
    }
    assert last_block_weighted_mean(history, 0) == pytest.approx(4.0)


def test_weighted_median_over_last_k_is_robust_to_outlier():
    history = {
        0: {0: [(2.0, 1.0), (2.0, 1.0), (3.0, 1.0)]},
        1: {0: [(100.0, 1.0)]},
    }
    assert weighted_median_K(2)(history, 0) == pytest.approx(2.0)
    assert last_block_weighted_mean(history, 0) == pytest.approx(100.0)


def test_trimmed_ema_is_robust_to_top_outlier():
    history = {
        0: {0: [(2.0, 1.0), (3.0, 1.0), (100.0, 1.0)]},
        1: {0: [(4.0, 1.0), (5.0, 1.0), (200.0, 1.0)]},
    }
    estimate = ema_trimmed_mean(K=2, trim_frac=1 / 3, lam=0.5)
    assert estimate(history, 0) == pytest.approx(3.5)


def test_estimators_return_prior_when_no_object_data():
    history = {0: {1: [(2.0, 1.0)]}}
    assert last_block_weighted_mean(history, 0, pi0=1.2) == pytest.approx(1.2)
    assert weighted_median_K(3)(history, 0, pi0=1.2) == pytest.approx(1.2)
    assert ema_trimmed_mean(2, 0.2, 0.5)(history, 0, pi0=1.2) == pytest.approx(1.2)

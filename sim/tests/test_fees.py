import pytest

from tfmsim.fees import alpha_min, burn_split, f_act, f_att, f_base, f_ui, sr, ur
from tfmsim.types import Tx


def test_alice_swap_numbers():
    prices = {0: 1.0, 1: 3.0, 2: 1.0}
    tx = Tx(
        id="alice-swap",
        sender_type="honest",
        compute=1.0,
        priority=1.0,
        declared=frozenset({0, 1, 2}),
        contingent=frozenset({1, 2}),
        used=frozenset({0}),
    )

    assert f_att(tx, prices) == pytest.approx(5.0)
    assert f_ui(tx, prices) == pytest.approx(1.0)
    assert f_base(tx, prices) == pytest.approx(1.0)

    for alpha in (0.0, 0.5, 1.0):
        assert f_act(tx, prices, alpha) == pytest.approx(1.0 + 4.0 * alpha)

    assert ur(tx, prices, 0.5) == pytest.approx(2.0)
    assert sr(tx, prices, 0.5) == pytest.approx(2.0)
    assert ur(tx, prices, 0.5) + sr(tx, prices, 0.5) == pytest.approx(4.0)


def test_used_set_semantics_and_alpha_formulas():
    prices = {0: 1.0, 1: 3.0, 2: 1.0}

    tx_empty = Tx(
        id="empty-used",
        compute=2.0,
        priority=1.5,
        declared=frozenset({0}),
        contingent=frozenset({0}),
        used=frozenset(),
    )
    assert f_ui(tx_empty, prices) == pytest.approx(0.0)
    assert f_act(tx_empty, prices, 0.5) == pytest.approx(0.5 * f_att(tx_empty, prices))

    tx_full = Tx(
        id="full-used",
        compute=2.0,
        priority=1.5,
        declared=frozenset({0, 1, 2}),
        contingent=frozenset({1}),
        used=None,
    )
    assert f_act(tx_full, prices, 0.25) == pytest.approx(f_att(tx_full, prices))
    assert f_ui(tx_full, prices) == pytest.approx(f_att(tx_full, prices))

    tx_mixed = Tx(
        id="mixed-used",
        compute=1.0,
        priority=1.0,
        declared=frozenset({0, 1, 2}),
        contingent=frozenset({1, 2}),
        used=frozenset({0}),
    )
    alpha = 0.4
    assert f_act(tx_mixed, prices, alpha) == pytest.approx(alpha * f_att(tx_mixed, prices) + (1.0 - alpha) * f_ui(tx_mixed, prices))
    assert ur(tx_mixed, prices, alpha) + sr(tx_mixed, prices, alpha) == pytest.approx(f_att(tx_mixed, prices) - f_ui(tx_mixed, prices))

    alpha_by_object = {0: 0.2, 1: 0.5, 2: 0.7}
    expected = 1.0 * 1.0 * 1.0 + 0.5 * 1.0 * 1.0 * 3.0 + 0.7 * 1.0 * 1.0 * 1.0
    assert f_act(tx_mixed, prices, alpha_by_object) == pytest.approx(expected)

    with pytest.raises(KeyError):
        f_act(tx_mixed, {0: 1.0}, 0.5)
    with pytest.raises(ValueError):
        f_act(tx_mixed, prices, 1.5)


def test_alpha_min_closed_form():
    assert alpha_min(1.2, 1.0 / 8.0, 0.5) == pytest.approx(0.15)
    assert alpha_min(5.0, 1.0 / 8.0, 0.5) == pytest.approx(0.625)
    assert alpha_min(16.0, 1.0 / 8.0, 0.5) == pytest.approx(2.0)


def test_invalid_tx_raises():
    with pytest.raises(ValueError):
        Tx(
            id="bad-contingent",
            declared=frozenset({0, 1}),
            contingent=frozenset({2}),
        )
    with pytest.raises(ValueError):
        Tx(
            id="bad-used",
            declared=frozenset({0, 1}),
            contingent=frozenset(),
            used=frozenset({2}),
        )
    with pytest.raises(ValueError):
        Tx(
            id="missing-noncontingent",
            declared=frozenset({0, 1}),
            contingent=frozenset({1}),
            used=frozenset({1}),
        )


def test_burn_split():
    scheduler, burned = burn_split(10.0, 0.5)
    assert scheduler == pytest.approx(5.0)
    assert burned == pytest.approx(5.0)

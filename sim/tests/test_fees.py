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


def test_alpha_min_closed_form():
    assert alpha_min(1.2, 1.0 / 8.0, 0.5) == pytest.approx(0.15)
    assert alpha_min(5.0, 1.0 / 8.0, 0.5) == pytest.approx(0.625)
    assert alpha_min(16.0, 1.0 / 8.0, 0.5) == pytest.approx(2.0)


def test_burn_split():
    scheduler, burned = burn_split(10.0, 0.5)
    assert scheduler == pytest.approx(5.0)
    assert burned == pytest.approx(5.0)

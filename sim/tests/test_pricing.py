import pytest

from tfmsim.pricing import update_object_price, update_price, utilization
from tfmsim.types import Block, ObjectState, Tx


def test_utilization_modes():
    txs = [
        Tx(id="t1", compute=1.0, priority=1.0, declared=frozenset({0, 1}), used=frozenset({0})),
        Tx(id="t2", compute=2.0, priority=1.0, declared=frozenset({1}), used=frozenset({1})),
    ]
    block = Block(index=0, txs=txs)

    assert utilization(block, mode="declared") == {0: 1.0, 1: 3.0}
    assert utilization(block, mode="used") == {0: 1.0, 1: 2.0}


def test_price_update_rules():
    assert update_price(100.0, 10.0, 10.0, 0.1, price_update="exp") == pytest.approx(100.0)
    assert update_price(100.0, 10.0, 15.0, 0.1, price_update="exp") > 100.0
    assert update_price(100.0, 10.0, 5.0, 0.1, price_update="exp") < 100.0

    eta = 0.01
    exp_price = update_price(100.0, 10.0, 10.1, eta, price_update="exp")
    linear_price = update_price(100.0, 10.0, 10.1, eta, price_update="linear")
    assert exp_price == pytest.approx(linear_price, rel=1e-8)

    assert update_price(0.1, 10.0, 20.0, 0.5, price_update="exp", p_min=0.2) == pytest.approx(0.2)


def test_object_state_update_floor():
    state = ObjectState(object_id=0, price=10.0, target_utilization=100.0, utilization=200.0, eta=0.1)
    updated = update_object_price(state, eta=0.1, price_update="exp", p_min=1.0)
    assert updated.price > 10.0
    assert updated.p_min == 1.0
    assert updated.eta == 0.1

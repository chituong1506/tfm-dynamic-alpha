from __future__ import annotations

from collections.abc import Mapping


def _price_for_object(prices: Mapping[int, float] | dict[int, float], obj: int) -> float:
    if isinstance(prices, Mapping):
        return float(prices.get(obj, 0.0))
    return float(prices[obj])


def f_att(tx, prices: Mapping[int, float]) -> float:
    """Attainable fee: all declared objects are used."""
    return sum(tx.priority * tx.compute * _price_for_object(prices, obj) for obj in tx.declared)


def f_ui(tx, prices: Mapping[int, float]) -> float:
    """User-ideal fee: actual used set."""
    used = tx.used if tx.used else tx.declared
    return sum(tx.priority * tx.compute * _price_for_object(prices, obj) for obj in used)


def f_base(tx, prices: Mapping[int, float]) -> float:
    """Base fee for the non-contingent subset of declared objects."""
    base_set = tx.declared - tx.contingent
    return sum(tx.priority * tx.compute * _price_for_object(prices, obj) for obj in base_set)


def f_act(
    tx,
    prices: Mapping[int, float],
    alpha: float,
    *,
    unused_fee_includes_priority_and_compute: bool = True,
) -> float:
    """Actual fee under a single alpha value for all objects."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    used = tx.used if tx.used else tx.declared
    total = 0.0
    for obj in tx.declared:
        price = _price_for_object(prices, obj)
        if obj in used:
            total += tx.priority * tx.compute * price
        else:
            if unused_fee_includes_priority_and_compute:
                total += alpha * tx.priority * tx.compute * price
            else:
                total += alpha * price
    return total


def ur(tx, prices: Mapping[int, float], alpha: float, **kwargs) -> float:
    return f_act(tx, prices, alpha, **kwargs) - f_ui(tx, prices)


def sr(tx, prices: Mapping[int, float], alpha: float, **kwargs) -> float:
    return f_att(tx, prices) - f_act(tx, prices, alpha, **kwargs)


def burn_split(actual_fee: float, gamma: float) -> tuple[float, float]:
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be strictly between 0 and 1")
    scheduler_revenue = gamma * actual_fee
    burned = (1.0 - gamma) * actual_fee
    return scheduler_revenue, burned


def alpha_min(pi: float, eta: float, gamma: float) -> float:
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be strictly between 0 and 1")
    return pi * eta * gamma / (1.0 - gamma)

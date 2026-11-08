from __future__ import annotations

from collections.abc import Mapping

# alpha: one value for all objects, or one value per object (needed by dynamic alpha)
Alpha = float | Mapping[int, float]


def _price(prices: Mapping[int, float], obj: int) -> float:
    """Price of an object. A missing price raises KeyError (never silently 0)."""
    return float(prices[obj])


def _used(tx):
    """used=None means every declared object was used; used=set() means nothing was used."""
    return tx.declared if tx.used is None else tx.used


def _alpha_for(alpha: Alpha, obj: int) -> float:
    a = float(alpha[obj]) if isinstance(alpha, Mapping) else float(alpha)
    if not 0.0 <= a <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    return a


def f_att(tx, prices: Mapping[int, float]) -> float:
    """Attainable fee: all declared objects are used."""
    return sum(tx.priority * tx.compute * _price(prices, o) for o in sorted(tx.declared))


def f_ui(tx, prices: Mapping[int, float]) -> float:
    """User-ideal fee: only the objects actually used."""
    return sum(tx.priority * tx.compute * _price(prices, o) for o in sorted(_used(tx)))


def f_base(tx, prices: Mapping[int, float]) -> float:
    """Base fee: only the non-contingent (always used) declared objects."""
    base_set = tx.declared - tx.contingent
    return sum(tx.priority * tx.compute * _price(prices, o) for o in sorted(base_set))


def f_act(
    tx,
    prices: Mapping[int, float],
    alpha: Alpha,
    *,
    unused_fee_includes_priority_and_compute: bool = True,
) -> float:
    """Actual fee. Used objects pay in full; declared-but-unused objects pay an alpha share."""
    used = _used(tx)
    total = 0.0
    for obj in sorted(tx.declared):
        a = _alpha_for(alpha, obj)          # validated for every declared object
        price = _price(prices, obj)
        if obj in used:
            total += tx.priority * tx.compute * price
        elif unused_fee_includes_priority_and_compute:
            total += a * tx.priority * tx.compute * price
        else:
            total += a * price
    return total


def ur(tx, prices: Mapping[int, float], alpha: Alpha, **kwargs) -> float:
    return f_act(tx, prices, alpha, **kwargs) - f_ui(tx, prices)


def sr(tx, prices: Mapping[int, float], alpha: Alpha, **kwargs) -> float:
    return f_att(tx, prices) - f_act(tx, prices, alpha, **kwargs)


def burn_split(actual_fee: float, gamma: float) -> tuple[float, float]:
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be strictly between 0 and 1")
    return gamma * actual_fee, (1.0 - gamma) * actual_fee


def alpha_min(pi: float, eta: float, gamma: float) -> float:
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be strictly between 0 and 1")
    return pi * eta * gamma / (1.0 - gamma)
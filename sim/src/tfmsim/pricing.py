from __future__ import annotations

import math
from collections.abc import Mapping
from collections import defaultdict
from dataclasses import replace

from .types import Block, ObjectState


def utilization(block: Block, *, mode: str = "declared", object_ids=None) -> dict[int, float]:
    """Compute per-object utilization over a block in either declared or used mode."""
    if mode not in {"declared", "used"}:
        raise ValueError("mode must be 'declared' or 'used'")

    totals: dict[int, float] = defaultdict(float)
    if object_ids is not None:
        for obj in object_ids:
            totals[obj] = 0.0

    for tx in block.txs:
        objects = (
            tx.declared
            if mode == "declared"
            else tx.declared if tx.used is None else tx.used
        )
        for obj in objects:
            totals[obj] += tx.compute

    if object_ids is not None:
        return {obj: totals.get(obj, 0.0) for obj in object_ids}
    return dict(totals)


def update_price(
    price: float,
    target_utilization: float,
    utilization_value: float,
    eta: float,
    *,
    price_update: str = "exp",
    p_min: float = 0.0,
) -> float:
    """Apply the price update rule for a single object."""
    if target_utilization <= 0:
        raise ValueError("target_utilization must be positive")
    if price_update not in {"exp", "linear"}:
        raise ValueError("price_update must be 'exp' or 'linear'")

    delta = (utilization_value - target_utilization) / target_utilization
    if price_update == "exp":
        new_price = price * math.exp(eta * delta)
    else:
        new_price = price * (1.0 + eta * delta)
    return max(p_min, new_price)


def update_object_price(
    state: ObjectState,
    *,
    eta: float | None = None,
    price_update: str = "exp",
    p_min: float | None = None,
) -> ObjectState:
    """Update an ObjectState in-place via the configured price rule."""
    next_eta = state.eta if eta is None else eta
    next_p_min = state.p_min if p_min is None else p_min
    next_price = update_price(
        state.price,
        state.target_utilization,
        state.utilization,
        next_eta,
        price_update=price_update,
        p_min=next_p_min,
    )
    return replace(state, price=next_price, eta=next_eta, p_min=next_p_min)


def update_prices(
    states: Mapping[int, ObjectState],
    *,
    eta: float | Mapping[int, float] | None = None,
    price_update: str = "exp",
    p_min: float | Mapping[int, float] | None = None,
) -> dict[int, ObjectState]:
    """Return updated states using a scalar or per-object eta and price floor."""
    updated: dict[int, ObjectState] = {}
    for object_id, state in states.items():
        object_eta = eta[object_id] if isinstance(eta, Mapping) else eta
        object_floor = p_min[object_id] if isinstance(p_min, Mapping) else p_min
        updated[object_id] = update_object_price(
            state,
            eta=object_eta,
            price_update=price_update,
            p_min=object_floor,
        )
    return updated

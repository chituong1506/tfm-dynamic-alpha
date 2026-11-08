"""Aggregate simulator metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .engine import SimulationResult


def summarize(
    result: SimulationResult,
    target_utilization: Mapping[int, float],
    *,
    convergence_tolerance: float = 0.05,
) -> dict[str, float | int | None]:
    """Compute accounting totals, price volatility, and first convergence block."""
    if not 0.0 < convergence_tolerance < 1.0:
        raise ValueError("convergence_tolerance must lie strictly between 0 and 1")
    if not result.records:
        return {
            "blocks": 0,
            "fees_paid": 0.0,
            "user_risk": 0.0,
            "scheduler_risk": 0.0,
            "scheduler_revenue": 0.0,
            "burned": 0.0,
            "mean_alpha": 0.0,
            "mean_eta": 0.0,
            "price_volatility": 0.0,
            "convergence_time": None,
        }

    log_price_changes: list[float] = []
    for previous, current in zip(result.price_history, result.price_history[1:]):
        log_price_changes.extend(
            math.log(current[obj] / previous[obj])
            for obj in target_utilization
            if current[obj] > 0 and previous[obj] > 0
        )
    convergence_time = next(
        (
            record.index
            for record in result.records
            if all(
                abs(record.utilization[obj] - target_utilization[obj])
                / target_utilization[obj]
                < convergence_tolerance
                for obj in target_utilization
            )
        ),
        None,
    )
    return {
        "blocks": len(result.records),
        "fees_paid": sum(record.fees_paid for record in result.records),
        "user_risk": sum(record.user_risk for record in result.records),
        "scheduler_risk": sum(record.scheduler_risk for record in result.records),
        "scheduler_revenue": sum(record.scheduler_revenue for record in result.records),
        "burned": sum(record.burned for record in result.records),
        "mean_alpha": float(np.mean([record.mean_alpha for record in result.records])),
        "mean_eta": float(np.mean([record.mean_eta for record in result.records])),
        "price_volatility": float(np.std(log_price_changes)) if log_price_changes else 0.0,
        "convergence_time": convergence_time,
    }

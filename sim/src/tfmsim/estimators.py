"""Priority estimators over per-block, per-object transaction observations."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

Observation = tuple[float, float]
History = Mapping[int, Mapping[int, Sequence[Observation]]]
Estimator = Callable[[History, int, float], float]


def _observations(history: History, object_id: int) -> list[tuple[int, list[Observation]]]:
    """Return valid observations for an object, ordered by block index."""
    blocks: list[tuple[int, list[Observation]]] = []
    for block_id, by_object in sorted(history.items()):
        observations = list(by_object.get(object_id, ()))
        for priority, compute in observations:
            if not math.isfinite(priority) or priority < 1:
                raise ValueError("observed priorities must be finite and at least 1")
            if not math.isfinite(compute) or compute <= 0:
                raise ValueError("observed compute must be finite and positive")
        blocks.append((block_id, observations))
    return blocks


def _weighted_mean(observations: Sequence[Observation]) -> float:
    total_compute = sum(compute for _, compute in observations)
    if total_compute <= 0:
        raise ValueError("observations must have positive total compute")
    return sum(priority * compute for priority, compute in observations) / total_compute


def _weighted_median(observations: Sequence[Observation]) -> float:
    ordered = sorted(observations, key=lambda pair: pair[0])
    halfway = sum(compute for _, compute in ordered) / 2.0
    cumulative = 0.0
    for priority, compute in ordered:
        cumulative += compute
        if cumulative >= halfway:
            return priority
    raise ValueError("cannot compute weighted median from empty observations")


def _trim_top(observations: Sequence[Observation], trim_frac: float) -> list[Observation]:
    if not 0.0 <= trim_frac < 1.0:
        raise ValueError("trim_frac must lie in [0, 1)")
    ordered = sorted(observations, key=lambda pair: pair[0])
    trim_count = min(int(len(ordered) * trim_frac), len(ordered) - 1)
    return ordered[: len(ordered) - trim_count]


def last_block_weighted_mean(
    history: History,
    object_id: int,
    pi0: float = 1.0,
) -> float:
    """Compute the last observed block's compute-weighted priority mean."""
    blocks = _observations(history, object_id)
    for _, observations in reversed(blocks):
        if observations:
            return _weighted_mean(observations)
    return pi0


def weighted_median_K(K: int) -> Estimator:
    """Build an estimator using a compute-weighted median over the last K data blocks."""
    if K <= 0:
        raise ValueError("K must be positive")

    def estimate(history: History, object_id: int, pi0: float = 1.0) -> float:
        blocks = _observations(history, object_id)[-K:]
        observations = [item for _, block in blocks for item in block]
        if not observations:
            return pi0
        return _weighted_median(observations)

    return estimate


def ema_trimmed_mean(K: int, trim_frac: float, lam: float) -> Estimator:
    """Build a block-wise EMA of compute-weighted means after removing top priorities."""
    if K <= 0:
        raise ValueError("K must be positive")
    if not 0.0 <= trim_frac < 1.0:
        raise ValueError("trim_frac must lie in [0, 1)")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must lie in [0, 1]")

    def estimate(history: History, object_id: int, pi0: float = 1.0) -> float:
        blocks = _observations(history, object_id)[-K:]
        blocks = [(block_id, items) for block_id, items in blocks if items]
        if not blocks:
            return pi0

        value: float | None = None
        for _, observations in blocks:
            block_mean = _weighted_mean(_trim_top(observations, trim_frac))
            value = block_mean if value is None else lam * block_mean + (1.0 - lam) * value
        return value

    return estimate

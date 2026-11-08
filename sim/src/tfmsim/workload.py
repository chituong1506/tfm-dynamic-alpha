"""Seeded, config-driven generation of common-random-number workloads."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .types import Block, Tx


@dataclass(frozen=True)
class WorkloadConfig:
    n_blocks: int = 200
    n_objects: int = 3
    lambda_per_object: float | tuple[float, ...] = 8.0
    max_demand_multiplier: float = 2.0
    epsilon: float = 1.0
    p_ref: float = 1.0
    popularity_zipf_s: float = 1.0
    min_declared_objects: int = 1
    max_declared_objects: int = 2
    declared_count_probabilities: tuple[float, ...] | None = None
    compute_log_mean: float = -1.0
    compute_log_sigma: float = 0.4
    t_min: float = 0.1
    t_max: float = 1.0
    contingent_fraction: float = 0.426
    p_under: float = 0.196
    priority_low_probability: float = 0.5
    priority_low_value: float = 1.2
    priority_high_mean: float = 8.8
    priority_high_p90: float = 16.0
    initial_price: float = 1.0

    def __post_init__(self) -> None:
        if self.n_blocks <= 0 or self.n_objects <= 0:
            raise ValueError("n_blocks and n_objects must be positive")
        if self.min_declared_objects <= 0 or self.max_declared_objects < self.min_declared_objects:
            raise ValueError("declared-object bounds must be positive and ordered")
        if self.max_declared_objects > self.n_objects:
            raise ValueError("max_declared_objects cannot exceed n_objects")
        if self.declared_count_probabilities is not None:
            probabilities = self.declared_count_probabilities
            expected_count = self.max_declared_objects - self.min_declared_objects + 1
            if len(probabilities) != expected_count:
                raise ValueError("declared_count_probabilities must match the configured object-count range")
            if any(not math.isfinite(probability) or probability < 0 for probability in probabilities):
                raise ValueError("declared-object probabilities must be finite and non-negative")
            if not math.isclose(sum(probabilities), 1.0):
                raise ValueError("declared-object probabilities must sum to 1")
        if not math.isfinite(self.epsilon) or self.epsilon < 0 or not math.isfinite(self.p_ref) or self.p_ref <= 0:
            raise ValueError("epsilon must be non-negative and p_ref positive")
        if not math.isfinite(self.max_demand_multiplier) or self.max_demand_multiplier < 1:
            raise ValueError("max_demand_multiplier must be finite and at least 1")
        if not math.isfinite(self.popularity_zipf_s) or self.popularity_zipf_s <= 0:
            raise ValueError("popularity_zipf_s must be positive")
        if (
            not math.isfinite(self.compute_log_mean)
            or not math.isfinite(self.compute_log_sigma)
            or self.compute_log_sigma < 0
            or not math.isfinite(self.t_min)
            or not math.isfinite(self.t_max)
            or self.t_min <= 0
            or self.t_max < self.t_min
        ):
            raise ValueError("compute distribution and clipping bounds are invalid")
        if (
            not math.isfinite(self.contingent_fraction)
            or not math.isfinite(self.p_under)
            or not 0 <= self.contingent_fraction <= 1
            or not 0 <= self.p_under <= 1
        ):
            raise ValueError("contingent_fraction and p_under must lie in [0, 1]")
        if not math.isfinite(self.priority_low_probability) or not 0 < self.priority_low_probability < 1:
            raise ValueError("priority_low_probability must lie strictly between 0 and 1")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.priority_low_value, self.priority_high_mean, self.priority_high_p90)
        ) or self.priority_low_value < 1:
            raise ValueError("priority distribution parameters must be positive and valid")
        if not math.isfinite(self.initial_price) or self.initial_price <= 0:
            raise ValueError("initial_price must be positive")
        if isinstance(self.lambda_per_object, tuple):
            if len(self.lambda_per_object) != self.n_objects:
                raise ValueError("lambda_per_object tuple must have one value per object")
            if any(not math.isfinite(rate) or rate < 0 for rate in self.lambda_per_object):
                raise ValueError("arrival rates must be finite and non-negative")
        elif not math.isfinite(self.lambda_per_object) or self.lambda_per_object < 0:
            raise ValueError("arrival rate must be finite and non-negative")

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> WorkloadConfig:
        values = dict(data)
        if isinstance(values.get("lambda_per_object"), list):
            values["lambda_per_object"] = tuple(values["lambda_per_object"])
        if isinstance(values.get("declared_count_probabilities"), list):
            values["declared_count_probabilities"] = tuple(values["declared_count_probabilities"])
        return cls(**values)


def _priority_lognormal_parameters(config: WorkloadConfig) -> tuple[float, float]:
    """Derive high-component lognormal parameters from its mean and p90."""
    z90 = 1.2815515655446004
    ratio_log = math.log(config.priority_high_p90 / config.priority_high_mean)
    discriminant = z90 * z90 - 2.0 * ratio_log
    if discriminant < 0:
        raise ValueError("priority high-component mean/p90 pair cannot define a lognormal")
    sigma = z90 - math.sqrt(discriminant)
    if sigma <= 0:
        raise ValueError("priority high-component parameters require positive variance")
    mu = math.log(config.priority_high_mean) - 0.5 * sigma * sigma
    return mu, sigma


def sample_priorities(
    size: int,
    rng: np.random.Generator,
    config: WorkloadConfig,
) -> np.ndarray:
    """Sample priorities from a calibrated point-mass/lognormal mixture."""
    if size < 0:
        raise ValueError("size must be non-negative")
    mu, sigma = _priority_lognormal_parameters(config)
    low_component = rng.random(size) < config.priority_low_probability
    priorities = np.empty(size, dtype=float)
    priorities[low_component] = config.priority_low_value
    priorities[~low_component] = np.maximum(1.0, rng.lognormal(mu, sigma, size=int((~low_component).sum())))
    return priorities


def _arrival_rates(config: WorkloadConfig) -> tuple[float, ...]:
    if isinstance(config.lambda_per_object, tuple):
        return config.lambda_per_object
    popularity = np.arange(1, config.n_objects + 1, dtype=float) ** (-config.popularity_zipf_s)
    popularity *= config.n_objects / popularity.sum()
    return tuple(config.lambda_per_object * rate for rate in popularity)


def expected_arrivals(
    config: WorkloadConfig,
    prices: Mapping[int, float],
) -> dict[int, float]:
    """Return price-elastic Poisson means for each primary object."""
    result: dict[int, float] = {}
    for object_id, base_rate in enumerate(_arrival_rates(config)):
        price = float(prices[object_id])
        if not math.isfinite(price) or price <= 0:
            raise ValueError("object prices must be finite and positive")
        result[object_id] = base_rate * (price / config.p_ref) ** (-config.epsilon)
    return result


def generate_workload(
    config: WorkloadConfig,
    rng: np.random.Generator,
) -> list[Block]:
    """Pre-generate candidate pools and execution outcomes from the supplied RNG."""

    popularity = np.arange(1, config.n_objects + 1, dtype=float) ** (-config.popularity_zipf_s)
    popularity /= popularity.sum()
    blocks: list[Block] = []

    for block_index in range(config.n_blocks):
        arrivals = {
            object_id: config.max_demand_multiplier * base_rate
            for object_id, base_rate in enumerate(_arrival_rates(config))
        }
        tx_specs: list[tuple[int, int]] = []
        for object_id in range(config.n_objects):
            count = int(rng.poisson(arrivals[object_id]))
            tx_specs.extend((object_id, sequence) for sequence in range(count))

        priorities = sample_priorities(len(tx_specs), rng, config)
        txs: list[Tx] = []
        for sequence, ((primary_object, _), priority) in enumerate(zip(tx_specs, priorities, strict=True)):
            count_values = np.arange(
                config.min_declared_objects,
                config.max_declared_objects + 1,
            )
            n_declared = int(
                rng.choice(
                    count_values,
                    p=config.declared_count_probabilities,
                )
                if config.declared_count_probabilities is not None
                else rng.choice(count_values)
            )
            n_declared = min(n_declared, config.n_objects)
            remaining = [obj for obj in range(config.n_objects) if obj != primary_object]
            remaining_weights = np.asarray([popularity[obj] for obj in remaining], dtype=float)
            if remaining:
                remaining_weights /= remaining_weights.sum()
                extras = rng.choice(
                    remaining,
                    size=n_declared - 1,
                    replace=False,
                    p=remaining_weights,
                )
                declared = frozenset((primary_object, *(int(obj) for obj in extras)))
            else:
                declared = frozenset({primary_object})

            compute = float(
                np.clip(
                    rng.lognormal(config.compute_log_mean, config.compute_log_sigma),
                    config.t_min,
                    config.t_max,
                )
            )
            contingent: frozenset[int] = frozenset()
            used = declared
            if declared and rng.random() < config.contingent_fraction:
                contingent_count = int(rng.integers(1, len(declared) + 1))
                contingent_items = rng.choice(tuple(sorted(declared)), size=contingent_count, replace=False)
                contingent = frozenset(int(obj) for obj in contingent_items)
                if rng.random() < config.p_under:
                    underused = int(rng.choice(tuple(sorted(contingent))))
                    used = declared - {underused}

            txs.append(
                Tx(
                    id=f"b{block_index}-tx{sequence}",
                    sender_type="honest",
                    compute=compute,
                    priority=float(priority),
                    declared=declared,
                    contingent=contingent,
                    used=used,
                    u=float(rng.random()),
                )
            )
        blocks.append(Block(index=block_index, txs=txs))
    return blocks

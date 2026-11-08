"""Fixed and dynamic-alpha mechanism parameter policies."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol

from .estimators import Estimator, History


class Mechanism(Protocol):
    def params_for_block(
        self,
        b: int,
        history: History,
        *,
        object_ids: tuple[int, ...] | None = None,
    ) -> dict[int, tuple[float, float]]:
        """Return per-object (alpha, eta) parameters for block b."""


def _object_ids(history: History, configured: tuple[int, ...] | None) -> tuple[int, ...]:
    if configured is not None:
        return configured
    return tuple(sorted({obj for block in history.values() for obj in block}))


class FixedAlpha:
    def __init__(
        self,
        alpha: float,
        eta: float,
        *,
        object_ids: tuple[int, ...] | None = None,
    ) -> None:
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if not math.isfinite(eta) or eta < 0:
            raise ValueError("eta must be finite and non-negative")
        self.alpha = alpha
        self.eta = eta
        self.object_ids = None if object_ids is None else tuple(object_ids)

    def params_for_block(
        self,
        b: int,
        history: History,
        *,
        object_ids: tuple[int, ...] | None = None,
    ) -> dict[int, tuple[float, float]]:
        del b
        return {
            obj: (self.alpha, self.eta)
            for obj in _object_ids(history, object_ids or self.object_ids)
        }


class DynamicAlpha:
    def __init__(
        self,
        alpha_max: float,
        kappa: float,
        eta0: float,
        gamma: float,
        estimator: Estimator,
        *,
        allow_kappa_below_one: bool = False,
        pi0: float = 1.0,
        object_ids: tuple[int, ...] | None = None,
    ) -> None:
        if not math.isfinite(alpha_max) or not 0.0 < alpha_max <= 1.0:
            raise ValueError("alpha_max must lie in (0, 1]")
        if not math.isfinite(kappa) or kappa <= 0:
            raise ValueError("kappa must be finite and positive")
        if kappa < 1.0 and not allow_kappa_below_one:
            raise ValueError("kappa below 1 requires allow_kappa_below_one=True")
        if not math.isfinite(eta0) or eta0 < 0:
            raise ValueError("eta0 must be finite and non-negative")
        if not math.isfinite(gamma) or not 0.0 < gamma < 1.0:
            raise ValueError("gamma must lie strictly between 0 and 1")
        if not math.isfinite(pi0) or pi0 < 1.0:
            raise ValueError("pi0 must be finite and at least 1")
        if not callable(estimator):
            raise TypeError("estimator must be callable")

        self.alpha_max = alpha_max
        self.kappa = kappa
        self.eta0 = eta0
        self.gamma = gamma
        self.estimator = estimator
        self.pi0 = pi0
        self.object_ids = None if object_ids is None else tuple(object_ids)

    def params_for_block(
        self,
        b: int,
        history: History,
        *,
        object_ids: tuple[int, ...] | None = None,
    ) -> dict[int, tuple[float, float]]:
        prior_history = {block: data for block, data in history.items() if block < b}
        params: dict[int, tuple[float, float]] = {}
        for obj in _object_ids(history, object_ids or self.object_ids):
            pi_hat = self.estimator(prior_history, obj, self.pi0)
            if not math.isfinite(pi_hat) or pi_hat < 1.0:
                raise ValueError("estimator must return a finite priority at least 1")
            eta = min(
                self.eta0,
                self.alpha_max * (1.0 - self.gamma) / (self.kappa * pi_hat * self.gamma),
            )
            alpha = min(
                self.alpha_max,
                self.kappa * pi_hat * eta * self.gamma / (1.0 - self.gamma),
            )
            params[obj] = (alpha, eta)
        return params

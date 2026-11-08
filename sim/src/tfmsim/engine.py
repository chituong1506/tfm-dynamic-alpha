"""Multi-block simulator loop for mechanisms, fees, utilization, and prices."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .block import build_block
from .estimators import History
from .fees import burn_split, f_act, sr, ur
from .mechanisms import Mechanism
from .pricing import update_prices, utilization
from .types import Block, ObjectState, Tx


@dataclass(frozen=True)
class SimulationConfig:
    object_ids: tuple[int, ...]
    target_utilization: Mapping[int, float]
    initial_prices: Mapping[int, float]
    max_compute: float
    cap_factor: float
    gamma: float = 0.5
    utilization_mode: str = "declared"
    price_update: str = "exp"
    p_min: float | Mapping[int, float] = 0.0
    unused_fee_includes_priority_and_compute: bool = True

    def __post_init__(self) -> None:
        if not self.object_ids or len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("object_ids must be non-empty and unique")
        objects = set(self.object_ids)
        if set(self.target_utilization) != objects or set(self.initial_prices) != objects:
            raise ValueError("target utilization and initial prices must cover exactly object_ids")
        if any(not math.isfinite(self.target_utilization[o]) or self.target_utilization[o] <= 0 for o in objects):
            raise ValueError("target utilizations must be finite and positive")
        if any(not math.isfinite(self.initial_prices[o]) or self.initial_prices[o] <= 0 for o in objects):
            raise ValueError("initial prices must be finite and positive")
        if not math.isfinite(self.max_compute) or self.max_compute <= 0:
            raise ValueError("max_compute must be finite and positive")
        if not math.isfinite(self.cap_factor) or self.cap_factor <= 0:
            raise ValueError("cap_factor must be finite and positive")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie strictly between 0 and 1")
        if self.utilization_mode not in {"declared", "used"}:
            raise ValueError("utilization_mode must be 'declared' or 'used'")
        if self.price_update not in {"exp", "linear"}:
            raise ValueError("price_update must be 'exp' or 'linear'")
        if isinstance(self.p_min, Mapping) and set(self.p_min) != objects:
            raise ValueError("per-object p_min must cover exactly object_ids")
        floors = self.p_min.values() if isinstance(self.p_min, Mapping) else (self.p_min,)
        if any(not math.isfinite(floor) or floor < 0 for floor in floors):
            raise ValueError("p_min values must be finite and non-negative")

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> SimulationConfig:
        values = dict(data)
        for key in ("target_utilization", "initial_prices"):
            if key in values:
                values[key] = {int(obj): float(value) for obj, value in values[key].items()}
        if "object_ids" in values:
            values["object_ids"] = tuple(int(obj) for obj in values["object_ids"])
        if isinstance(values.get("p_min"), dict):
            values["p_min"] = {int(obj): float(value) for obj, value in values["p_min"].items()}
        return cls(**values)


@dataclass(frozen=True)
class BlockRecord:
    index: int
    fees_paid: float
    user_risk: float
    scheduler_risk: float
    scheduler_revenue: float
    burned: float
    mean_alpha: float
    mean_eta: float
    utilization: Mapping[int, float]
    prices: Mapping[int, float]
    tx_count: int


@dataclass(frozen=True)
class SimulationResult:
    blocks: tuple[Block, ...]
    records: tuple[BlockRecord, ...]
    price_history: tuple[Mapping[int, float], ...]
    history: History


def simulate(
    workload: Sequence[Block],
    mechanism: Mechanism,
    config: SimulationConfig,
) -> SimulationResult:
    """Run a pre-generated workload through the configured fee mechanism."""
    object_ids = config.object_ids
    states = {
        obj: ObjectState(
            object_id=obj,
            price=config.initial_prices[obj],
            target_utilization=config.target_utilization[obj],
            p_min=config.p_min.get(obj, 0.0) if isinstance(config.p_min, Mapping) else config.p_min,
        )
        for obj in object_ids
    }
    history: dict[int, dict[int, list[tuple[float, float]]]] = {}
    included_blocks: list[Block] = []
    records: list[BlockRecord] = []
    price_history: list[Mapping[int, float]] = [
        {obj: states[obj].price for obj in object_ids}
    ]

    for block_index, candidate_block in enumerate(workload):
        params = mechanism.params_for_block(
            block_index,
            history,
            object_ids=object_ids,
        )
        if set(params) != set(object_ids):
            raise ValueError("mechanism must return (alpha, eta) for every configured object")
        alphas = {obj: params[obj][0] for obj in object_ids}
        etas = {obj: params[obj][1] for obj in object_ids}
        if any(not math.isfinite(alpha) or not 0 <= alpha <= 1 for alpha in alphas.values()):
            raise ValueError("mechanism returned alpha outside [0, 1]")
        if any(not math.isfinite(eta) or eta < 0 for eta in etas.values()):
            raise ValueError("mechanism returned invalid eta")

        built = build_block(
            candidate_block.txs,
            index=block_index,
            max_compute=config.max_compute,
            target_utilization=config.target_utilization,
            cap_factor=config.cap_factor,
        )
        prices = {obj: states[obj].price for obj in object_ids}
        paid = total_ur = total_sr = revenue = burned = 0.0
        for tx in built.txs:
            actual = f_act(
                tx,
                prices,
                alphas,
                unused_fee_includes_priority_and_compute=config.unused_fee_includes_priority_and_compute,
            )
            paid += actual
            total_ur += ur(
                tx,
                prices,
                alphas,
                unused_fee_includes_priority_and_compute=config.unused_fee_includes_priority_and_compute,
            )
            total_sr += sr(
                tx,
                prices,
                alphas,
                unused_fee_includes_priority_and_compute=config.unused_fee_includes_priority_and_compute,
            )
            tx_revenue, tx_burned = burn_split(actual, config.gamma)
            revenue += tx_revenue
            burned += tx_burned

        used = utilization(built, mode=config.utilization_mode, object_ids=object_ids)
        states = {
            obj: ObjectState(
                object_id=obj,
                price=states[obj].price,
                target_utilization=states[obj].target_utilization,
                utilization=used[obj],
                alpha=alphas[obj],
                eta=etas[obj],
                p_min=states[obj].p_min,
            )
            for obj in object_ids
        }
        records.append(
            BlockRecord(
                index=block_index,
                fees_paid=paid,
                user_risk=total_ur,
                scheduler_risk=total_sr,
                scheduler_revenue=revenue,
                burned=burned,
                mean_alpha=sum(alphas.values()) / len(object_ids),
                mean_eta=sum(etas.values()) / len(object_ids),
                utilization=used,
                prices=prices,
                tx_count=len(built.txs),
            )
        )
        included_blocks.append(built)

        history[block_index] = {obj: [] for obj in object_ids}
        for tx in built.txs:
            observed_objects = tx.declared if config.utilization_mode == "declared" else (
                tx.declared if tx.used is None else tx.used
            )
            for obj in observed_objects:
                history[block_index][obj].append((tx.priority, tx.compute))

        states = update_prices(
            states,
            eta=etas,
            price_update=config.price_update,
            p_min=config.p_min,
        )
        price_history.append({obj: states[obj].price for obj in object_ids})

    return SimulationResult(
        blocks=tuple(included_blocks),
        records=tuple(records),
        price_history=tuple(price_history),
        history=history,
    )

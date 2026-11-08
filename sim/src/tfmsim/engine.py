"""Copyable simulator state and single-block transition."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .block import build_block
from .estimators import History
from .fees import burn_split, f_act, f_att, f_ui
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
    p_min: float | Mapping[int, float] | None = None
    allow_zero_p_min: bool = False
    unused_fee_includes_priority_and_compute: bool = True
    reference_prices: Mapping[int, float] | None = None
    epsilon: float = 1.0
    max_demand_multiplier: float = 2.0

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
        if not math.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("epsilon must be finite and non-negative")
        if (
            not math.isfinite(self.max_demand_multiplier)
            or self.max_demand_multiplier < 1
        ):
            raise ValueError("max_demand_multiplier must be finite and at least 1")

        reference_prices = self.initial_prices if self.reference_prices is None else self.reference_prices
        if set(reference_prices) != objects or any(
            not math.isfinite(reference_prices[obj]) or reference_prices[obj] <= 0
            for obj in objects
        ):
            raise ValueError("reference prices must be finite, positive, and cover object_ids")
        object.__setattr__(self, "reference_prices", dict(reference_prices))

        if self.p_min is None:
            floors = {
                obj: self.initial_prices[obj] * 1e-6
                for obj in self.object_ids
            }
            object.__setattr__(self, "p_min", floors)
        elif isinstance(self.p_min, Mapping):
            if set(self.p_min) != objects:
                raise ValueError("per-object p_min must cover exactly object_ids")
            if any(not math.isfinite(floor) for floor in self.p_min.values()):
                raise ValueError("p_min values must be finite")
            if not self.allow_zero_p_min and any(floor <= 0 for floor in self.p_min.values()):
                raise ValueError("p_min must be positive unless allow_zero_p_min=True")
            if self.allow_zero_p_min and any(floor < 0 for floor in self.p_min.values()):
                raise ValueError("p_min values must be non-negative")
            object.__setattr__(self, "p_min", dict(self.p_min))
        else:
            if not math.isfinite(self.p_min):
                raise ValueError("p_min must be finite")
            if self.p_min <= 0 and not self.allow_zero_p_min:
                raise ValueError("p_min must be positive unless allow_zero_p_min=True")
            if self.p_min < 0:
                raise ValueError("p_min must be non-negative")

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> SimulationConfig:
        values = dict(data)
        for key in ("target_utilization", "initial_prices", "reference_prices"):
            if key in values and values[key] is not None:
                values[key] = {int(obj): float(value) for obj, value in values[key].items()}
        if "object_ids" in values:
            values["object_ids"] = tuple(int(obj) for obj in values["object_ids"])
        if isinstance(values.get("p_min"), dict):
            values["p_min"] = {int(obj): float(value) for obj, value in values["p_min"].items()}
        return cls(**values)


@dataclass(frozen=True)
class TxRecord:
    block: int
    tx_id: str
    sender_type: str
    priority: float
    compute: float
    n_declared: int
    n_unused: int
    f_att: float
    f_act: float
    f_ui: float
    ur: float
    sr: float


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
    alphas: Mapping[int, float] = field(default_factory=dict)
    etas: Mapping[int, float] = field(default_factory=dict)
    pi_hat: Mapping[int, float] | None = None


@dataclass
class SimState:
    object_states: dict[int, ObjectState]
    history: dict[int, dict[int, list[tuple[float, float]]]]
    price_history: list[dict[int, float]]
    next_block_index: int = 0
    blocks: list[Block] = field(default_factory=list)
    records: list[BlockRecord] = field(default_factory=list)
    tx_records: list[TxRecord] = field(default_factory=list)


@dataclass(frozen=True)
class SimulationResult:
    blocks: tuple[Block, ...]
    records: tuple[BlockRecord, ...]
    price_history: tuple[Mapping[int, float], ...]
    history: History
    tx_records: tuple[TxRecord, ...] = ()


def create_initial_state(config: SimulationConfig) -> SimState:
    """Create independent initial object states and empty estimator history."""
    object_ids = config.object_ids
    object_states = {
        obj: ObjectState(
            object_id=obj,
            price=config.initial_prices[obj],
            target_utilization=config.target_utilization[obj],
            p_min=config.p_min[obj] if isinstance(config.p_min, Mapping) else config.p_min,
        )
        for obj in object_ids
    }
    return SimState(
        object_states=object_states,
        history={},
        price_history=[{obj: object_states[obj].price for obj in object_ids}],
    )


def _validate_mechanism_gamma(mechanism: Mechanism, config: SimulationConfig) -> None:
    mechanism_gamma = getattr(mechanism, "gamma", None)
    if mechanism_gamma is not None and mechanism_gamma != config.gamma:
        raise ValueError(
            f"mechanism gamma ({mechanism_gamma}) must match config gamma ({config.gamma})"
        )


def _demand_selected(tx: Tx, prices: Mapping[int, float], config: SimulationConfig) -> bool:
    if not tx.declared:
        price_ratio = 1.0
    else:
        current_sum = sum(prices[obj] for obj in tx.declared)
        reference_sum = sum(config.reference_prices[obj] for obj in tx.declared)
        price_ratio = current_sum / reference_sum
    keep_probability = min(
        1.0,
        price_ratio ** (-config.epsilon) / config.max_demand_multiplier,
    )
    return tx.u <= keep_probability


def step(
    state: SimState,
    candidate_txs: Sequence[Tx],
    mechanism: Mechanism,
    config: SimulationConfig,
) -> tuple[SimState, BlockRecord, tuple[TxRecord, ...]]:
    """Advance a copied state through one block and return its records."""
    _validate_mechanism_gamma(mechanism, config)
    new_state = copy.deepcopy(state)
    block_index = new_state.next_block_index
    object_ids = config.object_ids
    prices = {obj: new_state.object_states[obj].price for obj in object_ids}
    eligible = [
        tx for tx in candidate_txs
        if _demand_selected(tx, prices, config)
    ]

    mechanism_history = copy.deepcopy(new_state.history)
    params = mechanism.params_for_block(
        block_index,
        mechanism_history,
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

    diagnostics_method = getattr(mechanism, "diagnostics_for_block", None)
    pi_hat = (
        diagnostics_method(
            block_index,
            copy.deepcopy(mechanism_history),
            object_ids=object_ids,
        )
        if callable(diagnostics_method)
        else None
    )
    built = build_block(
        eligible,
        index=block_index,
        max_compute=config.max_compute,
        target_utilization=config.target_utilization,
        cap_factor=config.cap_factor,
    )

    paid = total_ur = total_sr = revenue = burned = 0.0
    transaction_records: list[TxRecord] = []
    for tx in built.txs:
        actual = f_act(
            tx,
            prices,
            alphas,
            unused_fee_includes_priority_and_compute=config.unused_fee_includes_priority_and_compute,
        )
        attainable = f_att(tx, prices)
        user_ideal = f_ui(tx, prices)
        user_risk = actual - user_ideal
        scheduler_risk = attainable - actual
        paid += actual
        total_ur += user_risk
        total_sr += scheduler_risk
        tx_revenue, tx_burned = burn_split(actual, config.gamma)
        revenue += tx_revenue
        burned += tx_burned
        unused = tx.declared - (tx.declared if tx.used is None else tx.used)
        transaction_records.append(
            TxRecord(
                block=block_index,
                tx_id=tx.id,
                sender_type=tx.sender_type,
                priority=tx.priority,
                compute=tx.compute,
                n_declared=len(tx.declared),
                n_unused=len(unused),
                f_att=attainable,
                f_act=actual,
                f_ui=user_ideal,
                ur=user_risk,
                sr=scheduler_risk,
            )
        )

    built_utilization = utilization(
        built,
        mode=config.utilization_mode,
        object_ids=object_ids,
    )
    record = BlockRecord(
        index=block_index,
        fees_paid=paid,
        user_risk=total_ur,
        scheduler_risk=total_sr,
        scheduler_revenue=revenue,
        burned=burned,
        mean_alpha=sum(alphas.values()) / len(object_ids),
        mean_eta=sum(etas.values()) / len(object_ids),
        utilization=built_utilization,
        prices=prices,
        tx_count=len(built.txs),
        alphas=alphas,
        etas=etas,
        pi_hat=pi_hat,
    )

    new_state.history[block_index] = {obj: [] for obj in object_ids}
    for tx in built.txs:
        observed_objects = (
            tx.declared
            if config.utilization_mode == "declared"
            else tx.declared if tx.used is None else tx.used
        )
        for obj in observed_objects:
            new_state.history[block_index][obj].append((tx.priority, tx.compute))

    new_state.object_states = update_prices(
        {
            obj: ObjectState(
                object_id=obj,
                price=new_state.object_states[obj].price,
                target_utilization=new_state.object_states[obj].target_utilization,
                utilization=built_utilization[obj],
                alpha=alphas[obj],
                eta=etas[obj],
                p_min=new_state.object_states[obj].p_min,
            )
            for obj in object_ids
        },
        eta=etas,
        price_update=config.price_update,
        p_min=config.p_min,
    )
    new_state.price_history.append(
        {obj: new_state.object_states[obj].price for obj in object_ids}
    )
    new_state.blocks.append(built)
    new_state.records.append(record)
    new_state.tx_records.extend(transaction_records)
    new_state.next_block_index += 1
    return new_state, record, tuple(transaction_records)


def simulate(
    workload: Sequence[Block],
    mechanism: Mechanism,
    config: SimulationConfig,
    initial_state: SimState | None = None,
) -> SimulationResult:
    """Run candidate blocks as a plain loop over the copyable step transition."""
    _validate_mechanism_gamma(mechanism, config)
    state = copy.deepcopy(initial_state) if initial_state is not None else create_initial_state(config)
    if set(state.object_states) != set(config.object_ids):
        raise ValueError("initial state object IDs must match configuration")
    for candidate_block in workload:
        state, _, _ = step(state, candidate_block.txs, mechanism, config)
    return SimulationResult(
        blocks=tuple(state.blocks),
        records=tuple(state.records),
        price_history=tuple(copy.deepcopy(state.price_history)),
        history=copy.deepcopy(state.history),
        tx_records=tuple(state.tx_records),
    )

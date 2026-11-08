from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tx:
    id: str
    sender_type: str = "honest"
    compute: float = 1.0
    priority: float = 1.0
    declared: frozenset[int] = field(default_factory=frozenset)
    contingent: frozenset[int] = field(default_factory=frozenset)
    used: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.compute <= 0:
            raise ValueError("compute must be positive")
        if self.priority < 1:
            raise ValueError("priority must be at least 1")
        object.__setattr__(self, "declared", frozenset(self.declared))
        object.__setattr__(self, "contingent", frozenset(self.contingent))
        object.__setattr__(self, "used", frozenset(self.used))

    @property
    def t(self) -> float:
        return self.compute

    @property
    def pi(self) -> float:
        return self.priority


@dataclass
class Block:
    index: int
    txs: list[Tx] = field(default_factory=list)


@dataclass
class ObjectState:
    object_id: int
    price: float
    target_utilization: float
    utilization: float = 0.0
    alpha: float = 0.0
    eta: float = 0.0
    p_min: float = 0.0

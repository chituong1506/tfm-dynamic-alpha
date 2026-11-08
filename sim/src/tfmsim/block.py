"""Priority-ordered v1 block builder with compute and per-object capacity limits."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .types import Block, Tx


def build_block(
    candidates: Sequence[Tx],
    *,
    index: int,
    max_compute: float,
    target_utilization: Mapping[int, float],
    cap_factor: float,
) -> Block:
    """Greedily include highest-priority transactions that fit all capacities."""
    if not math.isfinite(max_compute) or max_compute <= 0:
        raise ValueError("max_compute must be finite and positive")
    if not math.isfinite(cap_factor) or cap_factor <= 0:
        raise ValueError("cap_factor must be finite and positive")
    capacities = {obj: cap_factor * target for obj, target in target_utilization.items()}
    if any(not math.isfinite(capacity) or capacity <= 0 for capacity in capacities.values()):
        raise ValueError("object capacities must be finite and positive")

    usage = {obj: 0.0 for obj in capacities}
    compute_used = 0.0
    selected: list[Tx] = []
    ordered = sorted(candidates, key=lambda tx: (-tx.priority, tx.id))
    for tx in ordered:
        if compute_used + tx.compute > max_compute:
            continue
        if any(obj not in capacities for obj in tx.declared):
            raise ValueError(f"transaction {tx.id} declares an unknown object")
        if any(usage[obj] + tx.compute > capacities[obj] for obj in tx.declared):
            continue

        selected.append(tx)
        compute_used += tx.compute
        for obj in tx.declared:
            usage[obj] += tx.compute
    return Block(index=index, txs=selected)


def block_usage(block: Block) -> tuple[float, dict[int, float]]:
    """Return total compute and declared per-object compute for a built block."""
    total = sum(tx.compute for tx in block.txs)
    usage: dict[int, float] = {}
    for tx in block.txs:
        for obj in tx.declared:
            usage[obj] = usage.get(obj, 0.0) + tx.compute
    return total, usage

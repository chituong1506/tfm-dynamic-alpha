import copy
from tfmsim.types import Block, Tx
from tfmsim.engine import SimulationConfig, create_initial_state, step, simulate
from tfmsim.mechanisms import FixedAlpha

results = []
def check(name, cond, detail=""):
    print(f"{'OK  ' if cond else 'FAIL'} {name} {detail}")
    results.append(bool(cond))

def tx(i, u, compute=1.0):
    return Tx(id=f"t{i}", compute=compute, priority=1.0, declared=frozenset({0}),
              contingent=frozenset(), used=None, u=u)

def cfg(price, **kw):
    base = dict(object_ids=(0,), target_utilization={0: 10.0}, initial_prices={0: price},
                reference_prices={0: 1.0}, max_compute=1000.0, cap_factor=2.0, gamma=0.5,
                epsilon=1.0, max_demand_multiplier=2.0)
    base.update(kw)
    return SimulationConfig(**base)

mech = FixedAlpha(0.5, 0.1, object_ids=(0,))

# 1. Loc cau theo gia (tinh tay): gia 2 -> giu voi xac suat 0.25 ; gia 1 -> 0.5
pool = [tx(0, 0.2), tx(1, 0.3)]
s = create_initial_state(cfg(2.0))
_, rec, _ = step(s, pool, mech, cfg(2.0))
check("gia 2: chi giu tx co u=0.2", rec.tx_count == 1, f"tx_count={rec.tx_count}")
s = create_initial_state(cfg(1.0))
_, rec, _ = step(s, pool, mech, cfg(1.0))
check("gia 1: giu ca hai (u<=0.5)", rec.tx_count == 2, f"tx_count={rec.tx_count}")
c0 = cfg(1.0, epsilon=0.0, max_demand_multiplier=1.0)
_, rec, _ = step(create_initial_state(c0), [tx(2, 0.99)], mech, c0)
check("epsilon=0, m=1: giu tat ca", rec.tx_count == 1)

# 2. step lien tiep == simulate, va chup trang thai roi chay tiep == chay lien mach
c = cfg(1.0)
workload = [Block(index=i, txs=tuple(tx(10 * i + k, 0.01) for k in range(3))) for i in range(6)]
full = simulate(workload, mech, c)

state = create_initial_state(c)
stepped_fees = []
for b in workload:
    state, record, _ = step(state, b.txs, mech, c)
    stepped_fees.append(record.fees_paid)
check("step lien tiep == simulate (phi tung block)",
      stepped_fees == [r.fees_paid for r in full.records])

snap_state = create_initial_state(c)
snapshot_fees = []
for b in workload[:3]:
    snap_state, record, _ = step(snap_state, b.txs, mech, c)
    snapshot_fees.append(record.fees_paid)
snapshot = copy.deepcopy(snap_state)
resumed = simulate(workload[3:], mech, c, initial_state=snapshot)
check("chup o block 3 roi chay tiep == chay lien mach",
      [*snapshot_fees, *[r.fees_paid for r in resumed.records]] == [r.fees_paid for r in full.records]
      and resumed.price_history[-1] == full.price_history[-1])

# 3. Sua ban chup khong anh huong ban goc
n_before, idx_before = len(snap_state.history), snap_state.next_block_index
step(snapshot, workload[3].txs, mech, c)
check("step khong sua trang thai dau vao",
      len(snap_state.history) == n_before and snap_state.next_block_index == idx_before)

print("\nALL PASS" if all(results) else "\nSOME CHECKS FAILED")
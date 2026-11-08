import copy
from tfmsim.types import Block, Tx
from tfmsim.engine import SimulationConfig, create_initial_state, step, simulate
from tfmsim.mechanisms import FixedAlpha

results = []
def check(name, cond, detail=""):
    print(f"{'OK  ' if cond else 'FAIL'} {name} {detail}")
    results.append(bool(cond))

def tx(i, u, compute=1.0, sender="honest"):
    return Tx(id=f"t{i}", sender_type=sender, compute=compute, priority=1.0,
              declared=frozenset({0}), contingent=frozenset(), used=None, u=u)

def cfg(price, **kw):
    base = dict(object_ids=(0,), target_utilization={0: 10.0}, initial_prices={0: price},
                reference_prices={0: 1.0}, max_compute=1000.0, cap_factor=2.0, gamma=0.5,
                epsilon=1.0, max_demand_multiplier=2.0)
    base.update(kw)
    return SimulationConfig(**base)

mech = FixedAlpha(0.5, 0.1, object_ids=(0,))

# 1. Loc cau theo gia (tinh tay): gia 2 -> xac suat giu 0.25 ; gia 1 -> 0.5
pool = [tx(0, 0.2), tx(1, 0.3)]
_, rec, _ = step(create_initial_state(cfg(2.0)), pool, mech, cfg(2.0))
check("gia 2: chi giu tx co u=0.2", rec.tx_count == 1, f"tx_count={rec.tx_count}")
_, rec, _ = step(create_initial_state(cfg(1.0)), pool, mech, cfg(1.0))
check("gia 1: giu ca hai (u<=0.5)", rec.tx_count == 2, f"tx_count={rec.tx_count}")
c0 = cfg(1.0, epsilon=0.0, max_demand_multiplier=1.0)
_, rec, _ = step(create_initial_state(c0), [tx(2, 0.99)], mech, c0)
check("epsilon=0, m=1: giu tat ca", rec.tx_count == 1)

# 2. step lien tiep == simulate ; chup trang thai roi chay tiep == chay lien mach
c = cfg(1.0)
workload = [Block(index=i, txs=tuple(tx(10 * i + k, 0.01) for k in range(3))) for i in range(6)]
full = simulate(workload, mech, c)

state, recs = create_initial_state(c), []
for b in workload:
    state, rec, _ = step(state, b.txs, mech, c)
    recs.append(rec)
check("step lien tiep == simulate (phi tung block)",
      [r.fees_paid for r in recs] == [r.fees_paid for r in full.records])

state3 = create_initial_state(c)
for b in workload[:3]:
    state3, _, _ = step(state3, b.txs, mech, c)
snapshot = copy.deepcopy(state3)
resumed = simulate(workload[3:], mech, c, initial_state=snapshot)
check("chup o block 3 roi chay tiep == chay lien mach",
      [r.fees_paid for r in resumed.records] == [r.fees_paid for r in full.records[3:]]
      and resumed.final_state.prices == full.final_state.prices)

# 3. step khong sua trang thai dau vao
idx, prices_before, hist_blocks = snapshot.next_block_index, snapshot.prices, sorted(snapshot.history)
step(snapshot, workload[3].txs, mech, c)
check("step khong sua trang thai dau vao",
      snapshot.next_block_index == idx and snapshot.prices == prices_before
      and sorted(snapshot.history) == hist_blocks)

# 4. forced transaction hooks
try:
    cf = cfg(1.0)
    honest = [tx(i, 0.01, compute=9.0) for i in range(3)]     # tran 2*10=20 -> chi vua 2 tx
    shill = tx(99, 0.99, compute=5.0, sender="shill")          # u=0.99 se bi loc cau neu khong "forced"
    _, rec0, _ = step(create_initial_state(cf), honest, mech, cf)
    s1, rec1, txrecs = step(create_initial_state(cf), honest, mech, cf, forced_txs=[shill])
    check("khong co forced: 2 tx", rec0.tx_count == 2)
    check("co forced: shill duoc them vao block day (3 tx)", rec1.tx_count == 3, f"got {rec1.tx_count}")
    check("utilization = 18 + 5 = 23 (vuot tran, shill khong chiem cho)",
          abs(rec1.utilization[0] - 23.0) < 1e-9, f"got {rec1.utilization[0]}")
    check("phi = 18 + 5 = 23", abs(rec1.fees_paid - 23.0) < 1e-9, f"got {rec1.fees_paid}")
    check("TxRecord giu sender_type=shill", any(r.sender_type == "shill" for r in txrecs))
except TypeError as e:
    check("forced_txs ton tai trong step()", False, f"-> {e}")

print("\nALL PASS" if all(results) else "\nSOME CHECKS FAILED")
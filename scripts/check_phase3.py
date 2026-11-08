import math
from tfmsim.types import Block, Tx
from tfmsim.engine import SimulationConfig, simulate
from tfmsim.mechanisms import FixedAlpha

def make_block(index, txs):
    return Block(index=index, txs=tuple(txs))     # chinh neu ten truong khac

results = []
def check(name, got, exp, tol=1e-9):
    ok = abs(got - exp) < tol
    print(f"{'OK  ' if ok else 'FAIL'} {name}: got {got}, expected {exp}")
    results.append(ok)

# ---- Kich ban A: 1 object, 2 tx khong contingent, 2 block giong het nhau ----
cfg = SimulationConfig(object_ids=(0,), target_utilization={0: 10.0}, initial_prices={0: 1.0},
                       max_compute=1000.0, cap_factor=2.0, gamma=0.5, p_min=1e-9)
txs = [Tx(id=f"t{i}", compute=6.0, priority=1.0, declared=frozenset({0}),
          contingent=frozenset(), used=None) for i in range(2)]
res = simulate([make_block(0, txs), make_block(1, txs)],
               FixedAlpha(0.5, 0.1, object_ids=(0,)), cfg)
r0, r1 = res.records
p1 = math.exp(0.1 * (12 - 10) / 10)                 # U=12, U*=10, eta=0.1
check("block0 phi tra", r0.fees_paid, 12.0)         # 2 * (1*6*1)
check("block0 doanh thu scheduler", r0.scheduler_revenue, 6.0)
check("block0 phi bi dot", r0.burned, 6.0)
check("block0 UR", r0.user_risk, 0.0)
check("gia sau block0", res.price_history[1][0], p1)
check("block1 phi tra", r1.fees_paid, 12.0 * p1)    # gia moi nhan voi 12

# ---- Kich ban B: 2 object, tx under-execute, alpha = 0.5 ----
cfg2 = SimulationConfig(object_ids=(0, 1), target_utilization={0: 100.0, 1: 100.0},
                        initial_prices={0: 1.0, 1: 3.0}, max_compute=1000.0, cap_factor=2.0,
                        gamma=0.5, p_min=1e-9)
tx = Tx(id="u", compute=1.0, priority=1.0, declared=frozenset({0, 1}),
        contingent=frozenset({1}), used=frozenset({0}))
res2 = simulate([make_block(0, [tx])], FixedAlpha(0.5, 0.1, object_ids=(0, 1)), cfg2)
r = res2.records[0]
check("under-exec phi = 1 + 0.5*3", r.fees_paid, 2.5)
check("under-exec UR = 1.5", r.user_risk, 1.5)
check("under-exec SR = 1.5", r.scheduler_risk, 1.5)
check("UR + SR = f_att - f_ui = 3", r.user_risk + r.scheduler_risk, 3.0)
check("gia object 0 sau block", res2.price_history[1][0], math.exp(0.1 * (1 - 100) / 100))

# ---- Kich ban C: chong vuot dung luong ----
big = [Tx(id=f"b{i}", compute=9.0, priority=1.0 + i, declared=frozenset({0}),
          contingent=frozenset(), used=None) for i in range(5)]   # moi tx 9, tran 20 -> chi 2 tx
res3 = simulate([make_block(0, big)], FixedAlpha(0.0, 0.1, object_ids=(0,)), cfg)
check("so tx vao block (tran 2*U*=20)", res3.records[0].tx_count, 2)
check("utilization <= 20", float(res3.records[0].utilization[0] <= 20.0), 1.0)

print("\nALL PASS" if all(results) else "\nSOME CHECKS FAILED")
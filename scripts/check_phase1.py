from tfmsim.types import Tx
from tfmsim.fees import f_att, f_ui, f_act, ur, sr

def check(name, got, expected):
    ok = abs(got - expected) < 1e-9
    print(f"{'OK  ' if ok else 'FAIL'} {name}: got {got}, expected {expected}")
    return ok

results = []
prices = {0: 1.0, 1: 3.0}

# 1. Tx không dùng object nào (used = tập rỗng) -- lỗi cũ sẽ cho 4.0 và 4.0
tx = Tx(id=1, priority=1.0, compute=1.0, declared={0, 1}, contingent={0, 1}, used=set())
results += [
    check("empty used: f_ui", f_ui(tx, prices), 0.0),
    check("empty used: f_act alpha=0.5", f_act(tx, prices, 0.5), 2.0),
    check("empty used: f_act alpha=0", f_act(tx, prices, 0.0), 0.0),
    check("empty used: f_act alpha=1", f_act(tx, prices, 1.0), 4.0),
]

# 2. used=None nghĩa là dùng hết
tx = Tx(id=2, priority=1.0, compute=1.0, declared={0, 1}, contingent={0, 1}, used=None)
results += [
    check("used=None: f_act == f_att", f_act(tx, prices, 0.3), f_att(tx, prices)),
]

# 3. Alpha theo từng object: object 0 alpha=0, object 1 alpha=1 -> 0*1 + 1*3 = 3
tx = Tx(id=3, priority=1.0, compute=1.0, declared={0, 1}, contingent={0, 1}, used=set())
results += [check("per-object alpha", f_act(tx, prices, {0: 0.0, 1: 1.0}), 3.0)]

# 4. Alice's swap: giá (1,3,1), pi=1, t=1, chỉ object 0 luôn được dùng
alice_prices = {0: 1.0, 1: 3.0, 2: 1.0}
alice = Tx(id=4, priority=1.0, compute=1.0, declared={0, 1, 2}, contingent={1, 2}, used={0})
for a, expected in [(0.0, 1.0), (0.5, 3.0), (1.0, 5.0)]:
    results.append(check(f"Alice alpha={a}", f_act(alice, alice_prices, a), expected))
    results.append(check(f"Alice UR+SR alpha={a}",
                         ur(alice, alice_prices, a) + sr(alice, alice_prices, a), 4.0))
favorable = Tx(id=5, priority=1.0, compute=1.0, declared={0, 1, 2}, contingent={1, 2}, used=None)
results.append(check("Alice favorable", f_act(favorable, alice_prices, 0.5), 5.0))

# 5. Thiếu giá phải báo lỗi, input sai phải báo lỗi
try:
    f_att(alice, {0: 1.0})
    print("FAIL missing price did not raise"); results.append(False)
except KeyError:
    print("OK   missing price raises KeyError"); results.append(True)

try:
    f_act(alice, alice_prices, 1.5)
    print("FAIL alpha=1.5 did not raise"); results.append(False)
except ValueError:
    print("OK   alpha=1.5 raises ValueError"); results.append(True)

try:
    Tx(id=6, priority=1.0, compute=1.0, declared={0}, contingent=set(), used={0, 9})
    print("FAIL invalid Tx did not raise"); results.append(False)
except ValueError:
    print("OK   invalid Tx raises ValueError"); results.append(True)

print("\nALL PASS" if all(results) else "\nSOME CHECKS FAILED")
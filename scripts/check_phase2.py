from tfmsim.mechanisms import DynamicAlpha, FixedAlpha

results = []

def check(name, cond, detail=""):
    print(f"{'OK  ' if cond else 'FAIL'} {name} {detail}")
    results.append(cond)

def params(pi_hat, **kw):
    cfg = dict(alpha_max=1.0, kappa=1.0, eta0=1 / 8, gamma=0.5)
    cfg.update(kw)
    mech = DynamicAlpha(estimator=lambda h, o, p0: pi_hat, object_ids=(0,), **cfg)
    return mech.params_for_block(5, {})[0]   # tra ve (alpha, eta)

def close(got, exp):
    return abs(got - exp) < 1e-12

# 1. Bang Table 2 cua bai goc (gamma=.5, eta0=1/8, alpha_max=1, kappa=1)
for pi, a_exp, e_exp in [(1.2, 0.15, 1 / 8), (5.0, 0.625, 1 / 8), (16.0, 1.0, 1 / 16)]:
    a, e = params(pi)
    check(f"pi_hat={pi}", close(a, a_exp) and close(e, e_exp), f"got alpha={a}, eta={e}")

# 2. alpha_max = 0.5: pi_hat=5 -> eta = 0.5*0.5/(5*0.5) = 0.1, alpha = 0.5
a, e = params(5.0, alpha_max=0.5)
check("alpha_max=0.5, pi_hat=5", close(a, 0.5) and close(e, 0.1), f"got {a}, {e}")

# 3. gamma = 0.25 (gamma/(1-gamma)=1/3): pi_hat=4 -> alpha=1/6 ; pi_hat=40 -> eta=0.075, alpha=1
a, e = params(4.0, gamma=0.25)
check("gamma=.25, pi_hat=4", close(a, 1 / 6) and close(e, 1 / 8), f"got {a}, {e}")
a, e = params(40.0, gamma=0.25)
check("gamma=.25, pi_hat=40", close(a, 1.0) and close(e, 0.075), f"got {a}, {e}")

# 4. kappa = 2: pi_hat=5 -> eta=0.1, alpha=1 ; pi_hat=1.2 -> eta=1/8, alpha=0.3
a, e = params(5.0, kappa=2.0)
check("kappa=2, pi_hat=5", close(a, 1.0) and close(e, 0.1), f"got {a}, {e}")
a, e = params(1.2, kappa=2.0)
check("kappa=2, pi_hat=1.2", close(a, 0.3) and close(e, 1 / 8), f"got {a}, {e}")

# 5. Tinh chat tren luoi tham so: alpha<=alpha_max, eta<=eta0, alpha >= can duoi shill-proof,
#    alpha khong giam theo pi_hat, eta khong tang theo pi_hat
ok_bound = ok_shill = ok_mono = True
for gamma in (0.1, 0.25, 0.5, 0.75, 0.9):
    for kappa in (1.0, 1.5, 3.0):
        for alpha_max in (0.3, 0.5, 1.0):
            prev = None
            for k in range(0, 200):
                pi = 1.0 + 0.5 * k
                a, e = params(pi, gamma=gamma, kappa=kappa, alpha_max=alpha_max)
                if a > alpha_max + 1e-12 or e > 1 / 8 + 1e-12:
                    ok_bound = False
                if a + 1e-9 < pi * e * gamma / (1 - gamma):      # kappa>=1 nen phai thoa
                    ok_shill = False
                if prev and (a < prev[0] - 1e-12 or e > prev[1] + 1e-12):
                    ok_mono = False
                prev = (a, e)
check("alpha<=alpha_max va eta<=eta0 tren luoi", ok_bound)
check("alpha >= can duoi shill-proof pi*eta*gamma/(1-gamma) tren luoi", ok_shill)
check("alpha khong giam, eta khong tang theo pi_hat", ok_mono)

# 6. Input sai phai bi tu choi
for name, fn in [
    ("pi_hat < 1", lambda: params(0.5)),
    ("gamma = 1", lambda: params(2.0, gamma=1.0)),
    ("alpha_max > 1", lambda: params(2.0, alpha_max=1.5)),
    ("kappa < 1 khong co co", lambda: params(2.0, kappa=0.5)),
]:
    try:
        fn(); check(f"raises on {name}", False)
    except ValueError:
        check(f"raises on {name}", True)

# 7. FixedAlpha
fa = FixedAlpha(0.5, 1 / 8, object_ids=(0, 1)).params_for_block(0, {})
check("FixedAlpha", fa == {0: (0.5, 1 / 8), 1: (0.5, 1 / 8)}, f"got {fa}")

print("\nALL PASS" if all(results) else "\nSOME CHECKS FAILED")
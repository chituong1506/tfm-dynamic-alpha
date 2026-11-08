# PROJECT: Dynamic-alpha Object-Weighted Transaction Fee Mechanism (OW-TFM) simulator

## 0. Role and goal
You are a senior research engineer. Build a reproducible Python simulator that
(a) reproduces the baseline OW-TFM from the paper "Perils of Parallelism: Transaction Fee
Mechanisms under Execution Uncertainty" (Wadhwa, Yaish, Zhang, Nayak; USENIX Security 2026),
(b) implements a NEW mechanism, "dynamic alpha", and (c) evaluates both against rational
attackers. The simulator supports a research paper; correctness and honesty matter more than speed.

Non-goals: no real blockchain integration, no networking, no GUI.

## 1. Model (self-contained; do not assume access to the paper)

### 1.1 Objects and transactions
- Objects o in {0..n_o-1}. Each has price p_{o,b} > 0 in block b, target utilization U*_o > 0.
- A transaction tx has: id, sender_type in {honest, shill}, compute t > 0, priority pi >= 1,
  declared object set D, contingent subset C subset of D (objects that may go unused),
  and a pre-sampled execution outcome: used set X, with (D minus C) subset of X subset of D.
  "Under-executed" means X != D.
- Contingent objects are used with probability (1 - p_under); sampled ONCE per tx at workload
  generation (common random numbers across mechanisms).

### 1.2 Fees (per object, then summed)
For tx in block b, with object price p_{o,b} and risk parameter alpha_{o,b} in [0,1]:
- object used:                    fee_o = pi * t * p_{o,b}
- object declared but unused:     fee_o = alpha_{o,b} * pi * t * p_{o,b}
- f_att  = sum over D of  pi*t*p_o                 (attainable fee: all declared used)
- f_ui   = sum over X of  pi*t*p_o                 (user-ideal fee)
- f_base = sum over (D minus C) of pi*t*p_o
- f_act  = sum over X of pi*t*p_o + sum over (D minus X) of alpha*pi*t*p_o
        = alpha*f_att + (1-alpha)*f_ui  (when alpha is the same for all objects)
- User risk UR = f_act - f_ui ; Scheduler risk SR = f_att - f_act ; UR + SR = f_att - f_ui.
- Burning: scheduler receives r_act = gamma * f_act, gamma in (0,1) constant; (1-gamma) burned.
  Assert 0 < gamma < 1.
- Config flag `unused_fee_includes_priority_and_compute` (default True). If False, unused fee is
  alpha * p_o only. Document which is used in every results file.

### 1.3 Utilization and price update (EIP-1559-like, per object)
- U_{o,b} = sum over tx in block b of t * 1[o in D(tx)]   (config `utilization_mode`: "declared"
  (default, matches the paper's lemma where an under-executing shill still adds t_s) or "used").
- p_{o,b+1} = max(p_min, p_{o,b} * exp( eta_{o,b} * (U_{o,b} - U*_o) / U*_o ))
  Config flag `price_update`: "exp" (default) or "linear" (1 + eta*x). 
- Fee used by a tx in block b is computed from p_{o,b} and alpha_{o,b}, both known at the START of
  block b (no dependence on the block's own schedule).

### 1.4 Shill-proofness condition (baseline theory, to be VALIDATED, not trusted)
Scheduler shill-proof (paper's Lemma 4, single-block, linearized): 
    alpha >= pi_o * eta * gamma / (1 - gamma)
where pi_o = average priority of honest txs using object o in block b+1.
Closed form helper: alpha_min(pi, eta, gamma) = pi*eta*gamma/(1-gamma).
With gamma=0.5, eta=1/8: pi=1.2 -> 0.15 ; pi=5 -> 0.625 ; pi=16 -> 2.0 (infeasible since > 1).

## 2. Mechanisms to implement (strategy pattern, common interface)
Interface: `params_for_block(b, history) -> dict[o] -> (alpha_o, eta_o)`.
1. `FixedAlpha(alpha, eta)`: same alpha, eta for all objects/blocks. Used for alpha in {0, 0.5, 1}
   (user-friendly / Even-Steven / scheduler-friendly) and for any alpha.
2. `DynamicAlpha(alpha_max, kappa, eta0, gamma, estimator)` (OUR CONTRIBUTION):
       pi_hat_{o,b} = estimator(history up to block b-1)        # forecast of pi in block b+1
       eta_{o,b}    = min( eta0 , alpha_max*(1-gamma) / (kappa*pi_hat*gamma) )
       alpha_{o,b}  = min( alpha_max , kappa*pi_hat*eta_{o,b}*gamma/(1-gamma) )
   Properties to verify numerically: alpha <= alpha_max always; when eta0 is not binding alpha
   equals the shill-proof lower bound times kappa; when eta0 is binding alpha == alpha_max.
   Expected values (gamma=.5, eta0=1/8, alpha_max=1, kappa=1): pi_hat=1.2 -> (eta=1/8, alpha=.15);
   5 -> (1/8, .625); 16 -> (1/16, 1.0).
3. Estimators of pi_hat (pluggable), computed from per-block, per-object lists of (pi, t):
   - `last_block_weighted_mean`
   - `weighted_median_K(K)`  (median of pi weighted by t over last K blocks)
   - `ema_trimmed_mean(K, trim_frac, lam)` (drop top trim_frac by priority, then EMA over blocks)
   If no data: return prior pi0 (config, default 1.0).

## 3. Workload generator (config-driven, seeded with numpy.random.Generator)
- n_o objects; object popularity ~ Zipf(s). Each tx declares 1..k objects (config distribution).
- Compute t ~ lognormal (config), clipped to [t_min, t_max].
- Priority: pi = max(1, X), X lognormal; CALIBRATE parameters with a script so that the resulting
  median ~ 1.2, mean ~ 5, 95th percentile ~ 16 (print achieved statistics; if lognormal cannot
  match all three, use a two-component mixture and document it).
- Contingency: fraction `contingent_fraction` of txs have contingent objects (default 0.426);
  each contingent tx under-executes with probability `p_under` (default 0.196). These defaults are
  loosely mapped from statistics cited in the paper and are ASSUMPTIONS; state this in outputs.
- Demand elasticity (needed for price convergence experiments): arrivals per object per block are
  Poisson(lambda_o * (p_o / p_ref)^(-epsilon)), epsilon configurable (default 1).
- Block builder (v1, simple): sort by priority desc, include a tx if total compute <= G and every
  declared object has remaining capacity cap_factor*U*_o. Honest sets are pre-generated per block so
  attack counterfactuals use identical honest txs (common random numbers). Shills are ADDED on top
  of the block and do not displace honest txs (flag `shill_displaces`, default False).

## 4. Adversaries
A1 Scheduler shill (multi-block, main attack). In block b the scheduler inserts a shill tx on object o:
   declares o, compute t_s, priority 1, forced under-executed (pays only the unused-object fee).
   Cost = (1-gamma) * f_act(shill) (scheduler gets gamma back). Gain = gamma * sum over blocks
   b+1..b+H of [f_act(honest txs | shill) - f_act(honest txs | no shill)].
   Implement by deep-copying state at block b and running two branches with identical honest txs and
   identical sampled outcomes. Report net_profit = gain - cost, and whether net_profit > 0.
   Horizon H configurable (default 1 = paper's setting; also run H = 5, 20).
A2 Estimator griefing: attacker inserts a high-priority tx (priority pi_a, compute t_a) on object o to
   inflate pi_hat and thus alpha for honest users. Report delta_alpha, extra user cost over horizon H,
   attacker cost, and ratio.
A3 (stretch) User shill against GCM baselines (Shapley, Time-Proportional Makespan) to reproduce the
   paper's Examples 5 and 6 (see tests).

## 5. Metrics (per run, per mechanism)
Mean/total UR, SR, scheduler revenue (sum gamma*f_act), burned amount, mean alpha, mean eta,
price volatility (std of log price ratio), convergence time (blocks until |U-U*|/U* < 5% after a
demand step), attack net_profit (mean, 95% bootstrap CI), fraction of profitable attacks,
fairness of UR across priority deciles. Use >= 30 seeds per configuration.

## 6. Experiments
E0 Sanity/reproduction: closed-form table (pi in {1.2,5,16}); Alice's swap example.
E1 Empirical break-even alpha for A1 under FixedAlpha vs the closed form; report the discrepancy.
E2 FixedAlpha (0, 0.5, 1) vs DynamicAlpha: UR, SR, revenue, attack profit.
E3 Dynamic alpha at the paper's 3 priority regimes (median / mean / p95 mix).
E4 Estimator comparison under A2 griefing.
E5 Price stability and convergence (demand step 1.0x -> 1.5x -> 1.0x) for eta fixed vs eta_o dynamic.
E6 Sensitivity sweeps: gamma, eta0, alpha_max, kappa, contingent_fraction, p_under, block size.

## 7. Tech stack and layout
Python >= 3.10, numpy, pandas, scipy, matplotlib, pyyaml, pytest. Type hints everywhere.
```
sim/
  src/tfmsim/  __init__.py  types.py  fees.py  pricing.py  mechanisms.py  estimators.py
               workload.py  block.py  engine.py  attacks.py  metrics.py  gcm.py  plots.py  run.py
  tests/       test_fees.py test_pricing.py test_mechanisms.py test_estimators.py
               test_attacks.py test_gcm.py test_workload.py
  configs/     e0.yaml ... e6.yaml
results/       (csv + png, one subfolder per experiment, with config copy and git commit hash)
```
CLI: `python -m tfmsim.run --config sim/configs/e2.yaml --out results/e2`.

## 8. Required unit tests (exact numbers)
- Alice's swap: declared {px,pool,bal}, prices (1,3,1), pi=1, t=1, contingent {pool,bal}.
  Favorable: f_act = 5. Under-executed: f_act = 1 + 4*alpha (alpha=0 -> 1, 0.5 -> 3, 1 -> 5).
  UR+SR = 4 on the under-executed branch for every alpha.
- alpha_min closed form values: 0.15, 0.625, 2.0 (see 1.4).
- DynamicAlpha expected values (see section 2). alpha never exceeds alpha_max (property test).
- GCM tests (single-object txs, unlimited threads, makespan v(S) = max over objects of summed t):
  Shapley: T={tx1(1,o1), tx2(1,o1), tx3(2,o2), tx4(3,o1)} -> gas(tx4) = 8/3.
  Adding shill tx5(1,o2): gas(tx4)=137/60, gas(tx5)=11/30, total 53/20 < 8/3.
  TPM: tx1(6,o1), tx2(6,o2) -> gas 3 each; adding tx3(6,{o1,o2}) -> gas 4 each (makespan 12).
  If a test fails, DO NOT change expected values; report the discrepancy.
- Price update: with U = U* price unchanged; U > U* raises price; exact exp vs linear agree for small eta.
- Counterfactual determinism: same seed gives identical results; shill-free branch equals baseline run.

## 9. Working rules (strict)
1. Read this file fully before coding. Never invent results or numbers; every number in docs/plots
   must come from a run. If an assumption is ambiguous, implement it as a config flag and list it in
   `docs/assumptions.md`.
2. Work in small steps; after each step run `pytest -q` and fix failures before continuing.
3. Make small git commits with prefixes (sim:, test:, docs:, fix:). Do not commit data/ or large files.
4. Deterministic seeds; one numpy Generator passed explicitly; no global random state.
5. If the simulator contradicts the theory (e.g., break-even alpha differs from the closed form),
   report it clearly in `results/<exp>/NOTES.md` instead of tuning parameters until it matches.
6. Keep functions pure where possible; no hard-coded constants outside configs.
7. Every results folder must contain: config.yaml, git commit hash, assumptions used, CSV, plots.

## 10. Definition of done
All tests pass; E0-E6 run from the CLI; each experiment has CSV + plots + NOTES.md;
`results/summary.md` lists findings with references to exact result files and states which of the
mechanism's properties were supported or contradicted by the data.
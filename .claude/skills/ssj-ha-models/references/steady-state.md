# Steady state and calibration

Contents:
1. [Choosing unknowns and targets (dynamics)](#1-choosing-unknowns-and-targets)
2. [The two-DAG pattern for steady states](#2-two-dag-pattern)
3. [Solver guidance](#3-solver-guidance)
4. [Calibration strategies](#4-calibration-strategies)
5. [Mandatory post-solution checks](#5-post-solution-checks)
6. [Failure modes and fixes](#6-failure-modes)

---

## 1. Choosing unknowns and targets

For the *dynamic* system `H(U, Z) = 0` (this choice also shapes the ss DAG):

- Count: #unknowns = #targets. By Walras's law one market is redundant — drop
  it from targets but keep it as an output (free accuracy check on every
  solution). Which one you drop is a matter of convenience; the solution is
  identical (paper slides show KS solved with either asset-market or
  goods-market target).
- Prefer unknowns that (i) enter early in the DAG so everything else can be
  computed from them, and (ii) make each block self-contained. Canonical picks
  (verified in the packaged examples):
  - Krusell–Smith: `U = [K]`, `H = [asset_mkt]`, shocks `[Z]`.
  - One-asset HANK: `U = [w, Y, pi]`, `H = [asset_mkt, goods_mkt, nkpc_res]`,
    shocks `[rstar, Z]` (labor_mkt left untargeted via hetoutput `NE`... note:
    this example targets goods_mkt and leaves labor_mkt as the check).
  - Two-asset HANK: `U = [r, w, Y]`, `H = [asset_mkt, fisher, wnkpc]` — the
    NKPC, arbitrage, and investment conditions are absorbed into SolvedBlocks
    instead of adding unknowns.
- Keep n small (≤ 3 typical). Every forward-looking condition can either be a
  top-level unknown/target pair or a SolvedBlock; use SolvedBlocks when the
  condition is local to a subsystem (pricing, investment) and top-level pairs
  when it is genuinely economy-wide (market clearing, Fisher equation).

## 2. Two-DAG pattern

Steady-state calibration usually inverts different equations than the dynamics
solve. The clean pattern (used by all packaged examples): build a *separate
steady-state DAG* in which analytic "helper" blocks replace hard-to-invert
dynamic blocks, then evaluate the *dynamic* DAG at the solved calibration.

```python
@sj.simple
def firm_ss(r, Y, L, delta, alpha):
    '''Invert firm FOCs: given targets (r, Y, L), back out (K, Z, w).'''
    rk = r + delta
    K = alpha * Y / rk
    Z = Y / K ** alpha / L ** (1 - alpha)
    w = (1 - alpha) * Z * (K / L) ** alpha
    return K, Z, w

ks_ss  = create_model([household, firm_ss, mkt_clearing], name='KS SS')
ks     = create_model([household, firm,    mkt_clearing], name='KS')

calibration = {'eis': 1.0, 'delta': 0.025, 'alpha': 0.11, 'rho': 0.966,
               'sigma': 0.5, 'Y': 1.0, 'L': 1.0, 'nS': 7, 'nA': 500,
               'amax': 200, 'r': 0.01}
ss = ks_ss.solve_steady_state(calibration, {'beta': (0.98/1.01, 0.999/1.01)},
                              {'asset_mkt': 0.}, solver='brentq')
# For a model whose ss DAG differs from the dynamic DAG, re-evaluate:
ss = ks.steady_state(ss)         # populates dynamic-DAG variables; ss checks below
```

Why: normalizations like `Y = L = 1` and price targets like `r = 0.01` become
*inputs* of the ss DAG; deep parameters (`Z`, `K`, `w`, `mup`, `alpha`) become
*outputs* computed analytically. This reduces numerical root-finding to the few
genuinely implicit parameters (typically `beta`, plus `vphi`, `chi1`, ...).
The two-asset example's `partial_ss` block shows a rich version (backs out
markup, capital share, TFP, wage from wealth/output/capital targets).

After re-evaluating the dynamic DAG, **verify the helper blocks were
consistent**: the dynamic-DAG residuals (`nkpc_res`, `val`, `inv`, ...) must be
≈ 0 at `ss`. The toolkit warns if a consistency residual exceeds `ctol=1e-9`
when using helper-block workflows; treat that warning as an error.

If the ss unknowns of an embedded SolvedBlock are pinned by calibration (e.g.
`pi = 0`), pass `dissolve=['pricing_solved']` to `solve_steady_state` so the
block is evaluated, not solved.

## 3. Solver guidance

- **1 unknown**: bracket it: `unknowns={'beta': (lo, hi)}, solver='brentq'`.
  Robust; requires a sign change — check the residual at both ends if brentq
  complains (`f(a) and f(b) must have different signs`).
- **2+ unknowns**: `solver='broyden_custom'` (default) or `'hybr'`, scalar
  initial values. `broyden_custom` prints iterations with `verbose=True`.
  If unknowns must stay in a region (e.g. `chi1 > 0`), use 3-tuples
  `(lb, x0, ub)` — enforced by a linear-continuation penalty.
- Tolerances: `ttol` (solver, default 1e-12) and the HetBlock's internal
  `backward_tol=1e-8` / `forward_tol=1e-10`. For publication runs tighten the
  HetBlock tolerances (`backward_tol=1e-10, forward_tol=1e-12` via kwargs or
  `options={'hh': {...}}`) and confirm aggregates move < 1e-8.
- β for one-asset models must satisfy `beta * (1 + r) < 1` (borrowing-
  constrained models need impatience for a stationary distribution). Bracket
  accordingly, e.g. `(0.95/(1+r), 0.9999/(1+r))`.
- Expensive models: nest solvers — outer 1-D loop over the hardest parameter,
  inner analytic/cheap solves — or write the loop manually around
  `model.steady_state(...)` for full control. `solve_steady_state` is a thin
  wrapper; manual loops are legitimate and sometimes clearer.

## 4. Calibration strategies

Standard quarterly targets (state the frequency explicitly in output):
- `r`: 0.005–0.0125 quarterly (2–5% annual) — an input under the two-DAG
  pattern; `beta` solves asset-market clearing.
- Liquid/total wealth: B/Y or A/Y targets (e.g. B = 5.6 quarterly-Y in the
  one-asset HANK example ⇒ 140% of annual GDP; two-asset: tot_wealth=14,
  Bh=1.04).
- Labor: `L = 1` or `NE = 1` normalization; `vphi` solves it.
- Income process: ρ_e ≈ 0.966 quarterly (0.87 annual), cross-sectional sd of
  log earnings ≈ 0.5–0.92 depending on the target concept. Cite the source for
  whatever you use (e.g. Floden–Lindé, IKC appendix).
- Price/wage rigidity: κ's from slope estimates; μ from markup evidence.

**Distributional targets** need the internals. Example — top-10% wealth share
and constrained share as calibration diagnostics (or targets in a manual loop):

```python
D = ss.internals['hh']['D']; a = ss.internals['hh']['a']
order = np.argsort(a, axis=None)
Df, af = D.ravel()[order], a.ravel()[order]
cum_pop, cum_wealth = np.cumsum(Df), np.cumsum(af * Df)
top10 = 1 - np.interp(0.9, cum_pop, cum_wealth) / cum_wealth[-1]
constrained_share = D[..., 0].sum()          # mass exactly at the constraint
```

To *target* moments like average MPC or top shares, add the moment computation
to a manual outer loop (or a hetoutput when it is a distribution-weighted mean).
Multi-β calibrations (kron pattern, hetblocks.md §5–6) are the standard tool for
matching both aggregate wealth and high MPCs (see IKC; the Goethe
`calibration.py` targets A = 100% or 500% of annual GDP with 4 β types).

## 5. Post-solution checks

Run immediately, every time (all four are cheap):

```python
assert abs(ss['asset_mkt']) < 1e-8               # each target
assert abs(ss['goods_mkt']) < 1e-6               # untargeted market (Walras)
D = ss.internals['hh']['D']
assert abs(D.sum() - 1) < 1e-12
assert D[..., -5:].sum() < 1e-4                  # mass near top of asset grid
```

- A failing Walras check with passing targets = an accounting bug (budget
  constraint, dividend definition, adjustment-cost resource costs, transfer
  incidence). Find it now; it will not average out later.
- Also record: constrained share, wealth Gini/top shares, average MPC
  (annualized) — and compare with the paper's targets or the data. An MPC check:
  `mpc = (c[..., 1:] - c[..., :-1]) / ((1 + r) * (a_grid[1:] - a_grid[:-1]))`
  weighted by `D` (report the quarterly number and 1-(1-mpc)^4 annualization
  as approximation).

## 6. Failure modes

| Symptom | Likely cause → fix |
|---|---|
| `No convergence of policy functions after 5000 backward iterations` | β(1+r) ≥ 1; bad `backward_init`; grid too coarse near the constraint; NaNs from negative consumption (raise `amin`, check income > 0) |
| `No convergence after 100000 forward iterations` | Ergodic distribution doesn't exist (β too high ⇒ mass escapes to amax; check policy at top of grid `a[..., -1] < a_grid[-1]`); disconnected Markov chain (provide `Pi_seed` / `Dbeg` seed) |
| brentq: `f(a) and f(b) must have different signs` | Bracket doesn't span the root — evaluate the residual at both ends and widen/re-center; check the residual is monotone in the unknown |
| broyden diverges | Bad initial values; unknowns of wildly different scales (rescale, e.g. solve log β); residuals not continuously differentiable at the guess (kink from a binding constraint region appearing/disappearing) |
| Consistency-check warning (`maximum residual value of ... greater than the ctol`) | Helper ss block inconsistent with the dynamic block (e.g. forgot adjustment-cost term in dividends) — fix the block, don't raise ctol |
| ss solves but Jacobian/IRF garbage | Distribution piled at grid top (raise amax); grid too coarse (raise nA); backward tolerance too loose |

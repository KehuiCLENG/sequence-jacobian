# Writing HetBlocks: households and firms

A HetBlock is defined by a single **backward-step function**: given the
expectation of next period's backward variable(s) and current aggregate inputs,
return this period's backward variable(s), the policy(ies), and any other
individual-level outcomes. The toolkit does everything else (steady-state
iteration, distribution lotteries, fake-news Jacobians, nonlinear paths).

Before writing one from scratch, check whether a built-in block or a workshop
notebook already implements the problem (see §8 and references/resources.md) —
adapting a tested block via hetinputs/hetoutputs/remap is faster and safer.

Contents:
1. [Anatomy of the canonical one-asset block (EGM)](#1-anatomy)
2. [Designing `backward_init`](#2-backward_init)
3. [Endogenous labor supply](#3-endogenous-labor)
4. [Two assets / two policies](#4-two-assets)
5. [Multiple exogenous states; shocking transition matrices](#5-multiple-exogenous-states)
6. [Permanent types, life cycle, OLG-style structures](#6-types-and-lifecycle)
7. [Heterogeneous firms](#7-heterogeneous-firms)
8. [Discrete choice and StageBlock (experimental)](#8-discrete-choice)
9. [Grid design](#9-grid-design)
10. [Block-level unit checks](#10-block-level-checks)
11. [What is NOT expressible](#11-limits)

---

## 1. Anatomy

The canonical standard-incomplete-markets household (this is
`sj.hetblocks.hh_sim.hh`, verified source):

```python
from sequence_jacobian import het, interpolate, misc, grids
import numpy as np

def hh_init(a_grid, y, r, eis):
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)          # ~MPC of 0.1 guess
    return Va

@het(exogenous='Pi', policy='a', backward='Va', backward_init=hh_init)
def hh(Va_p, a_grid, y, r, beta, eis):
    uc_nextgrid = beta * Va_p                          # (e, a') marginal value of saving
    c_nextgrid = uc_nextgrid ** (-eis)                 # invert Euler: c on endogenous grid
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]   # cash on hand on (e, a)
    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)  # a'(e, a)
    misc.setmin(a, a_grid[0])                          # borrowing constraint
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)                     # envelope condition
    return Va, a, c
```

Line-by-line economics:
- `Va_p[e, a']` = `E[Va_{t+1}(e', a') | e]` — the toolkit already took the
  conditional expectation over `e'` using `Pi`.
- EGM: the Euler equation `u'(c) = beta * E[Va']` holds exactly on the grid of
  *tomorrow's* assets `a'`; `c_nextgrid + a_grid` is the endogenous grid of
  cash-on-hand for which each `a'` is optimal.
- `interpolate_y(x, xq, y)` interpolates the mapping coh → a' back onto the
  fixed coh values implied by today's grid. Its linear extrapolation below the
  endogenous grid is then overwritten by `setmin` (the binding-constraint
  region), which is what makes this correct.
- Envelope: `Va = (1+r) u'(c)`. Returning `Va` (not `V`) is why one backward
  variable suffices for a smooth problem.
- GOTCHA: the toolkit parses the `return` statement TEXTUALLY to get output
  names. Return plain variable names on a SINGLE line — no expressions, no
  parenthesized/multi-line returns (`return (V,\n  Vs, ...)` yields a bogus
  output named `'(V'` and a constructor error). Assign to a name first, and
  keep long return lists on one line even if it breaks line-length style.

Then wrap with hetinputs so the block takes aggregates:

```python
def make_grids(rho_e, sd_e, n_e, min_a, max_a, n_a):
    e_grid, e_dist, Pi = grids.markov_rouwenhorst(rho_e, sd_e, int(n_e))
    a_grid = grids.asset_grid(min_a, max_a, int(n_a))
    return e_grid, e_dist, Pi, a_grid

def income(w, e_grid):        # or (Z, e_grid) for post-tax income Z
    y = w * e_grid
    return y

household = hh.add_hetinputs([income, make_grids])
# household.inputs: r, beta, eis, w, rho_e, sd_e, n_e, min_a, max_a, n_a
# household.outputs: A, C
```

Return `e_dist` (stationary skill distribution) from the grid hetinput if any
downstream block needs incidence weights (`pi_e` in the HANK example).

## 2. backward_init

Requirements: returns every backward variable, correct shape, finite, and
consistent enough with the envelope condition that backward iteration
converges. The standard recipe — guess consumption as a fixed fraction of
cash-on-hand and apply the envelope condition — is robust:
`Va = (1+r) * (0.1 * coh)**(-1/eis)`. If backward iteration diverges or
oscillates, check (in order): `beta * (1+r) < 1`; income strictly positive at
every grid point (or constraint handling for zero-income states); the init
producing decreasing-in-assets marginal values.

## 3. Endogenous labor

See `sj.hetblocks.hh_labor.hh` (source: `src/sequence_jacobian/hetblocks/hh_labor.py`)
for the full pattern: MRS condition `vphi * n**(1/frisch) = we * u'(c)` solved
jointly with the Euler equation on the endogenous grid; at the borrowing
constraint, a pointwise Newton solve (`solve_cn` / vectorized `solve_uc`) for
the intratemporal condition given `a' = a_min`. Key detail: the constrained
region must be handled *separately* — interpolation alone silently violates the
intratemporal condition there. Aggregate effective labor is exposed via a
hetoutput `labor_supply(n, e_grid) -> ne` (block output `NE`), and market
clearing targets `NE - L`.

## 4. Two assets

See `sj.hetblocks.hh_twoasset.hh`. Structural facts to preserve when adapting:

```python
@het(exogenous='Pi', policy=['b', 'a'], backward=['Vb', 'Va'],
     hetinputs=[marginal_cost_grid], hetoutputs=[adjustment_costs], backward_init=hh_init)
def hh(Va_p, Vb_p, a_grid, b_grid, z_grid, e_grid, k_grid, beta, eis, rb, ra,
       chi0, chi1, chi2, Psi1):
    ...
    return Va, Vb, a, b, c, uce
```

- Order is consistent everywhere: `policy=['b','a']`, `backward=['Vb','Va']`,
  array shape `(n_z, n_b, n_a)` (exog first, then policies in listed order).
- Two backward variables = two envelope conditions (`Vb = (1+rb) u'(c)`,
  `Va = (1+ra−Psi2) u'(c)`).
- The unconstrained/constrained split (liquid constraint `b' = b_min`) uses a
  conditional-optimality search (`lhs_equals_rhs_interpolate`) plus a
  `k_grid = agrid(...)[::-1].copy()` trick — a grid over the constraint
  multiplier. Do not simplify this away; kinks are where naive interpolation
  breaks.
- `uce` (`e·u'(c)`) is returned because the wage-NKPC union block needs `UCE`.
- Precomputable individual-level objects that depend only on grids/prices
  (here `Psi1` on an `(a', a)` grid) belong in a hetinput.

## 5. Multiple exogenous states

`exogenous=['Pi1', 'Pi2']` gives state ordering `(n_1, n_2, policies...)`;
each `Pi_k` transitions its own dimension independently.

Two ways to combine processes:
1. **Kron into one chain** (when you want correlation or a joint process):
   `Pi = np.kron(Pi_b, Pi_e)`, `e = np.kron(np.ones(n_b), e_grid)`,
   `beta = np.kron(beta_grid, np.ones(n_e))[:, np.newaxis]` — note that a
   state-dependent *parameter* (like β) is just an individual-level array
   broadcast over the state space (see `goethe-workshop-2026/notebooks/calibration.py`).
2. **Separate matrices** (independent processes): cleaner, and each can be
   shocked separately.

**Shocking a transition matrix**: produce `Pi` inside a hetinput from aggregate
inputs, e.g. job separation/finding rates:

```python
def make_Pi(f, s):     # f, s are aggregate inputs -> Pi becomes time-varying
    Pi = np.array([[1 - s, s], [f, 1 - f]])
    return Pi
```

Jacobians w.r.t. `f`, `s` are then fully supported (the fake-news algorithm
includes the expectation-shock terms for perturbed transition matrices);
nonlinear paths too. Canonical example: `notebooks/labor_search.ipynb`.

## 6. Types and lifecycle

**Permanent types** (patient/impatient, sectors): copies via `remap`+`rename`
plus an aggregation simple block — see api.md §2.5 and
`examples/krusell_smith.py::remapped_dag`. Each copy carries its own
distribution; masses are calibration parameters.

**Stochastic type switching** (e.g. β redrawn every ~25 years): encode as an
extra Markov dimension (kron pattern above) — this keeps one block and one
distribution.

**Perpetual youth / death risk**: add death probability to the Markov chain
(newborn draws from some income distribution), and, with annuities, scale
returns by survival. Assets pass through death via the endogenous state, so
decide explicitly what happens to them:
- *annuity markets*: survivors receive `(1+r)/s` — implement in the budget;
- *accidental bequests*: dying agents' resources must be collected and
  redistributed. Make the bequest flow an **aggregate**: hetoutput aggregates
  resources of the dying; a lump-sum transfer hetinput hands them back;
  a GE target enforces `Beq_received = Beq_left`. Value functions may depend on
  aggregates, so this is squarely within the framework.

**Deterministic life cycle (age as a state)**: age `j = 0..J-1` as an
exogenous dimension with `Pi_age[j, j+1] = 1` (and `Pi_age[J-1, 0] = 1` for
rebirth). Implement age-dependent behavior by operating on rows of the arrays
inside the backward function. Two things must be handled by hand:
1. **Terminal condition**: a non-altruistic dying agent must not value the
   newborn's continuation. Hard-code the terminal-age policy (e.g. `a' = 0`,
   `c = coh`) so it ignores `Va_p` in those rows.
2. **Rebirth assets**: the terminal-age policy `a' = 0` makes newborns start at
   zero assets; any other bequest scheme needs the aggregate-bequest pattern
   above.
Check the age distribution in `ss.internals[...]['D']` is exactly uniform
across ages (it must be, by construction) — a failed check means the aging
matrix is wrong. Costs scale with `J`; quarterly life cycles are expensive
(J ≈ 240), annual is standard.

## 7. Heterogeneous firms

The HetBlock abstraction is agent-agnostic: "households" can be firms with
states (productivity `z`, capital `k`, possibly debt `b`) and policies
(`k'`, `b'`). Mapping a Khan–Thomas/Winberry/Ottonello–Winberry-style model:

- **States**: `exogenous='Pi_z'` (and possibly an iid death/exit indicator as a
  second exogenous dimension), `policy=['k']` or `['b','k']` (2 max).
- **Backward variable**: marginal value(s) of capital (and debt) via envelope
  conditions, or the value function itself if you need it for exit decisions.
- **Aggregate inputs**: `w`, `r`, relative prices, tax/transfer instruments.
- **Hetoutputs**: investment `i = k' − (1−δ)k + φ(·)`, labor demand `n(z,k;w)`,
  dividends, output → aggregated to `I`, `N_d`, `DIV`, `Y_firm` for market
  clearing.
- **Exit/entry with fixed mass**: encode exit as an exogenous state ("dead"
  draw realized at the start of the period); in dead rows, hard-code the
  liquidation policy (e.g. `k' = k_entrant`) so the lottery reinserts entrants
  at the entry state; route liquidation proceeds through a hetoutput into the
  dividend flow. The exogenous chain sends dead → entrant productivity draw.
  This keeps total mass at 1 (exit rate = entry rate), which is what the
  standard framework supports.
- **Convexify or smooth**: the fake-news algorithm differentiates the backward
  step numerically, so policies must respond smoothly to aggregate inputs.
  Convex adjustment costs are safe. Fixed costs/lumpy investment need taste
  shocks or another smoothing device (§8); a hard (S,s) rule without smoothing
  makes numerical derivatives unreliable — validate against the direct method
  before trusting (validation.md §6).

Frictionless-entry models with an endogenous *mass* of firms (free-entry
condition determining entry flows, time-varying total mass) do **not** fit the
standard distribution law of motion; they need the paper's Appendix A.2
extension (custom code, not automated in the toolkit).

Economics reference: `goethe-workshop-2026/markdown/lecture11_heterogeneous_firms.md`
(canonical HANK+investment, collateral constraints, MPIs, cash-flow channel).

## 8. Discrete choice

Discrete choices (participation, adjustment, default) smoothed with logit
taste shocks fit the framework. Two routes:
- Smooth the choice inside a standard backward function (logit inclusive
  values; keep the value function `V` as a backward variable alongside `Va`,
  use DC-EGM with an upper-envelope step for non-concave regions).
- The experimental **StageBlock** interface
  (`sequence_jacobian.blocks.stage_block.StageBlock` with stages
  `ExogenousMaker`, `LogitChoice`, `Continuous1D`) decomposes a period into
  stages. Working example with full DC-EGM + upper envelope:
  `tests/base/test_dchoice.py` (labor-force participation, states
  (s, x, e, a)). Study that test before using StageBlock; it is not part of the
  stable documented API.
References: NBER-2023 "Lecture 9 Discrete Choice.pdf" and "Tutorial 4 Discrete
Choice.ipynb".

## 9. Grid design

- Assets: `grids.asset_grid(amin, amax, n)` (double-exponential, dense near the
  constraint). Sizes: exploration 100–200 points; publication 200–500 with a
  convergence check. `amax` must comfortably exceed the top of the ergodic
  distribution: check `D` mass near the top (validation.md §3) — rule of thumb
  amax ≈ 2–4× the 99.99th percentile of wealth.
- Income: `markov_rouwenhorst` with `n_e` 7–11 (2–3 only for toy examples).
  It matches persistence and unconditional variance of log income exactly, and
  behaves well for ρ near 1 (unlike Tauchen).
- Non-convex problems: `utilities.discretize.nonlinspace(amax, n, phi)` for
  flexible curvature; put points where policies kink.
- Two-asset: the paper's production calibration uses 10,500 idiosyncratic
  states (nZ=3, nB=50, nA=70; Table I) — the packaged `two_asset.dag()`
  calibration is deliberately tiny (nB=10, nA=16, for fast tests) and NOT
  publication-grade. The same warning applies to `krusell_smith.dag()` (nA=10!)
  and `hank.dag()`: always re-solve with serious grids.

## 10. Block-level checks

Before embedding a new HetBlock in a model, test it standalone:

```python
calib = dict(...)                       # partial-equilibrium prices/params
ss = household.steady_state(calib)

D = ss.internals['household']['D']; a = ss.internals['household']['a']
assert abs(np.vdot(D, a) - ss['A']) < 1e-12          # aggregation identity
assert abs(D.sum() - 1) < 1e-12
# budget identity (adapt to the model): C + A = (1+r)*A_lagged + income.
# NB: hetinput outputs keep their natural shape — here y is (n_e,), so
# broadcast to the full (n_e, n_a) grid before aggregating against D:
y = np.broadcast_to(ss.internals['household']['y'][:, np.newaxis], D.shape)
assert abs(ss['C'] + ss['A'] - (1 + calib['r']) * ss['A'] - np.vdot(D, y)) < 1e-8
```

Also: policies monotone in assets (`np.all(np.diff(a, axis=-1) >= 0)` for
standard problems), consumption positive, Euler-equation residuals ≈ 0 off the
constraint (spot-check by direct computation), and a PE Jacobian sanity check —
`J['C','r']`, `J['C','y']` columns should look like the textbook iMPC/interest
rate responses (column `s`: spike/anticipation at `t ≤ s`, decay after).

## 11. Limits

Not expressible in the standard HetBlock (from the paper, Section 3.2 "Scope
and Limitations"):
- value functions depending directly on the *distribution* (not through
  aggregates): wage posting, individual bargaining, money search with
  distribution-dependent terms, OLG with endogenous bequest *distributions*
  received mid-life (aggregate lump-sum bequests are fine, see §6);
- time-varying total mass of agents (endogenous entry flows);
- more than 2 endogenous policy states;
- nonlinear functionals of the distribution as outputs (quantiles) — possible
  with the paper's Appendix A.2 extension, by hand;
- aggregate-risk-dependent behavior (portfolio choice from risk premia,
  precautionary responses to aggregate volatility): first-order in aggregates
  kills these by construction. Higher-order sequence-space methods exist
  (Goethe lecture 14) but are not in the public toolkit; say so rather than
  faking it.

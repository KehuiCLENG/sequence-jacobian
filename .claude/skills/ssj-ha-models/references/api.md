# `sequence_jacobian` API reference (verified against source, v1.0.0 / master)

Everything below was verified directly against the package source
(`src/sequence_jacobian/`) and the test suite. Import convention:

```python
import numpy as np
import sequence_jacobian as sj
from sequence_jacobian import simple, het, solved, combine, create_model, grids, interpolate, misc, hetblocks
```

Table of contents:
1. [SimpleBlock](#1-simpleblock)
2. [HetBlock](#2-hetblock)
3. [SolvedBlock](#3-solvedblock)
4. [CombinedBlock / create_model](#4-combinedblock)
5. [Block methods (all block types)](#5-block-methods)
6. [Container classes](#6-container-classes)
7. [Utilities: grids, interpolation, estimation, drawdag](#7-utilities)
8. [Deprecated names](#8-deprecated-names)

---

## 1. SimpleBlock

```python
@sj.simple
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha - 1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y
```

- Inputs = function arguments; outputs = names in the `return` statement.
  The function name becomes the block name.
- Time displacement: `x(-1)` (lag), `x(+1)` (lead), any integer. In steady
  state displacements are ignored; in impulses/Jacobians they shift sequences.
- Nonlinear functions of displaced *expressions* need `.apply`:
  `(1 + pi).apply(np.log)`, `x(+1).apply(np.log)`. Plain `+ - * / **` work
  directly on displaced variables. `np.log(pi)` on an *undisplaced* argument
  works but `.apply` is the safe uniform habit inside simple blocks.
- Closures over Python variables (e.g. a steady-state dict captured from outer
  scope, `ss['B']`) are allowed and are treated as constants.
- `.jacobian` of a SimpleBlock returns `SimpleSparse` objects — exact,
  analytic-sparsity representations `{(displacement, 0): coefficient}` that act
  like matrices (materialize with `J['Y','K'].matrix(T)`); no `T` needed at
  computation time.

## 2. HetBlock

### 2.1 Constructor

```python
def hh_init(a_grid, y, r, eis):
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    Va = (1 + r) * (0.1 * coh) ** (-1 / eis)
    return Va

@sj.het(exogenous='Pi', policy='a', backward='Va', backward_init=hh_init)
def hh(Va_p, a_grid, y, r, beta, eis):
    uc_nextgrid = beta * Va_p
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    misc.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c
```

`@het(exogenous, policy, backward, backward_init=None, hetinputs=None, hetoutputs=None)`:

- `exogenous`: name (or list of names) of Markov transition-matrix inputs.
  Each `Pi` is an `(n, n)` array input: `Pi[i, j] = Prob(next = j | today = i)`.
  Multiple exogenous states are allowed (`exogenous=['Pi1', 'Pi2']`); their
  grid dimensions come first, in order.
- `policy`: name(s) of the endogenous continuous state policy output(s),
  **lowercase**, at most 2. Each policy `x` requires an input named `x_grid`.
- `backward`: name(s) of backward variables (e.g. `'Va'`, or `['Vb','Va']`).
  For each backward variable `V`, the function must take `V_p` as an argument
  and return `V`. `V_p` is **the expectation of next period's `V` conditional
  on today's exogenous state** — the toolkit applies each Markov matrix; never
  apply `Pi` yourself.
- `backward_init`: function of a subset of the block's inputs returning initial
  values of all backward variables to start steady-state backward iteration.
  Skipped if the calibration already contains the backward variables.

### 2.2 Shapes, states, timing

- All individual-level arrays are shaped `(exog states..., policy states...)`,
  e.g. `(n_e, n_a)` or `(n_z, n_b, n_a)`. Broadcast accordingly
  (`y[:, np.newaxis]`, `a_grid[np.newaxis, :]`).
- Within period `t`: `Dbeg_t` → (apply exog Markov) → `D_t` → (apply policy
  lottery, Young 2010 two-point method) → `Dbeg_{t+1}`. Policies, outputs, and
  aggregation are all defined on `D_t` (today's exog state × assets brought in).
  Aggregates: `O_t = np.vdot(D_t, o_t)`.
- Non-backward outputs of the backward function must be lowercase; the block's
  aggregate outputs are their uppercase versions (`a → A`, `c → C`).
- Steady state stores individual-level objects in
  `ss.internals[block_name]`: `D`, `Dbeg`, the exogenous matrices, every
  backward-function output (`Va`, `a`, `c`, ...), and all hetinput/hetoutput
  outputs. Top level gets only aggregates and scalars.
- Steady-state iteration controls (kwargs to `steady_state`/`solve_steady_state`):
  `backward_tol=1e-8, backward_maxit=5000, forward_tol=1e-10, forward_maxit=100_000`.
- Optional seeding inputs: `Dbeg` (initial distribution guess) and
  `<Pi_name>_seed` (stationary-distribution seed) speed up / disambiguate the
  forward iteration (relevant when an exogenous chain has multiple ergodic sets,
  e.g. permanent types encoded in one Markov matrix).

### 2.3 hetinputs and hetoutputs

```python
def make_grids(rho_e, sd_e, n_e, min_a, max_a, n_a):
    e_grid, _, Pi = grids.markov_rouwenhorst(rho_e, sd_e, n_e)
    a_grid = grids.asset_grid(min_a, max_a, n_a)
    return e_grid, Pi, a_grid

def income(w, e_grid):
    y = w * e_grid
    return y

household = hh.add_hetinputs([income, make_grids])   # returns a NEW block
```

- `add_hetinputs([...])` / `add_hetoutputs([...])` / `remove_hetinputs([...])` /
  `remove_hetoutputs([...])` return modified copies (functional style — assign
  the result).
- hetinputs run **before** the backward step each period; they map aggregate
  inputs to individual-level arrays (grids, incomes, transfer incidence, even
  the Markov matrices themselves — that is how you shock a transition matrix).
  Their outputs are consumed by the backward function/hetoutputs and become
  internals, not aggregate outputs.
- hetoutputs run **after** the backward step; they map individual-level
  variables (and optionally `D`) to additional individual outcomes, which are
  aggregated and exposed uppercase (e.g. `labor_supply(n, e_grid) -> ne` gives
  block output `NE`).
- In Jacobians, hetinputs are always differentiated two-sided; the backward
  function and hetoutputs use `h=1e-4`, one-sided by default (`twosided=True`
  to upgrade).

### 2.4 Time-varying exogenous transitions

Because `Pi` is just an input (possibly produced by a hetinput from aggregate
variables like a job-finding rate `f`), paths and Jacobians with respect to
variables that move `Pi` are fully supported, including the expectation-shock
terms in the fake news algorithm. See the labor-search notebook
(`notebooks/labor_search.ipynb`) for the canonical example.

### 2.5 remap / rename (multiple agent types)

```python
to_map = ['beta', *household.outputs]
hh_patient   = household.remap({k: k + '_patient' for k in to_map}).rename('hh_patient')
hh_impatient = household.remap({k: k + '_impatient' for k in to_map}).rename('hh_impatient')

@sj.simple
def aggregate(A_patient, A_impatient, C_patient, C_impatient, mass_patient):
    C = mass_patient * C_patient + (1 - mass_patient) * C_impatient
    A = mass_patient * A_patient + (1 - mass_patient) * A_impatient
    return C, A
```

`remap(dict)` renames inputs/outputs; `rename(name)` renames the block. Both
return copies. Works on any block type.

## 3. SolvedBlock

Two ways to create:

```python
@sj.solved(unknowns={'pi': (-0.1, 0.1)}, targets=['nkpc'], solver='brentq')
def pricing_solved(pi, mc, r, Y, kappap, mup):
    nkpc = kappap * (mc - 1/mup) + Y(+1)/Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1)) \
           - (1 + pi).apply(np.log)
    return nkpc

production_solved = combine([labor, investment]).solved(
    unknowns={'Q': 1., 'K': 10.}, targets=['inv', 'val'], solver='broyden_custom')
```

- A SolvedBlock is a mini-model: from the outside, its unknowns become
  *outputs* and its targets disappear. `.jacobian` internally solves the mini
  GE system (its `H_U` is factored and cached via `partial_jacobians`);
  `.impulse_nonlinear` runs a mini Newton solve.
- In `solve_steady_state` of an enclosing model, pass
  `dissolve=['pricing_solved']` to treat the block's unknowns as ordinary
  calibration inputs at steady state (used when the ss values are known, e.g.
  `pi = 0`), rather than solving the mini system numerically.
- Use for: NKPC (price and wage), Q-theory investment with adjustment costs,
  equity-pricing/arbitrage conditions, government-debt feedback rules.

## 4. CombinedBlock

```python
model    = sj.create_model([household, firm, fiscal, mkt_clearing, nkpc], name='HANK')
combined = sj.combine([labor, investment])            # same class, no model alias
```

- Topologically sorts blocks into a DAG (`model.blocks` shows execution order;
  `model.inputs` / `model.outputs` the unresolved inputs and all outputs).
  A cycle raises an error — break cycles by adding unknowns/targets or a
  SolvedBlock.
- `create_model` = `combine` with a model repr; both support all block methods.
- Nested combinations are fine (a CombinedBlock inside another).

## 5. Block methods

Every block (simple, het, solved, combined) supports 4 basic + 4 solve methods.
`ss` below is a `SteadyStateDict`, impulse inputs/outputs are `ImpulseDict`s
(deviations from ss), Jacobians are `JacobianDict`s.

### 5.1 `steady_state(calibration, dissolve=[], **kwargs) -> SteadyStateDict`
Evaluates the block at a point. For HetBlocks: backward iteration to
convergence, then forward iteration for the stationary distribution, then
aggregation. `calibration` is a plain dict or SteadyStateDict containing every
input.

### 5.2 `solve_steady_state(calibration, unknowns, targets, solver="", dissolve=[], ttol=1e-12, ctol=1e-9, verbose=False, **kwargs) -> SteadyStateDict`
Wraps `steady_state` in a root finder.
- `unknowns` formats: `{'beta': (0.97, 0.999)}` 2-tuple = bracket (1 unknown,
  bracketing solver); `{'beta': 0.98, 'vphi': 0.8}` scalars = initial values
  (multivariate solver); `{'x': (lb, x0, ub)}` 3-tuple = initial value with
  bounds enforced by linear-continuation penalty.
- `targets` formats: `{'asset_mkt': 0.}` (residual = value), `{'r': 0.01}`
  (variable hits value), `{'B': 'Bh'}` (variable equals other variable), or a
  list `['asset_mkt']` (each = 0).
- `solver`: 1-D scipy: `brentq` (default for 1 bracketed unknown), `bisect`,
  `brenth`, `ridder`, `toms748`, `newton`, `secant`, `halley`; multi-D scipy:
  `hybr`, `lm`, `broyden1`, `broyden2`, `anderson`, `krylov`, `df-sane`, ...;
  custom: `broyden_custom` (default for >1 unknown), `newton_custom`;
  `"solved"` = evaluate once at given unknowns (no solving).
- `dissolve=[block_name,...]`: for SolvedBlocks whose ss unknowns are supplied
  directly in the calibration.

### 5.3 `jacobian(ss, inputs, outputs=None, T=None, Js={}, **kwargs) -> JacobianDict`
Partial-equilibrium Jacobians of `outputs` w.r.t. `inputs` around `ss`,
truncated at `T` (required for HetBlocks; SimpleBlocks return SimpleSparse and
ignore `T`). HetBlock kwargs: `h=1e-4`, `twosided=False`. `Js` supplies
precomputed Jacobians by block name (skips recomputation if the saved
JacobianDict covers the requested inputs/outputs).

### 5.4 `impulse_linear(ss, inputs, outputs=None, Js={}) -> ImpulseDict`
First-order PE response to shock paths. `inputs` is a dict/ImpulseDict of
deviation paths, e.g. `{'r': 0.001*0.9**np.arange(T)}`. Equivalent to applying
the Jacobian to the path.

### 5.5 `impulse_nonlinear(ss, inputs, outputs=None, internals={}, Js={}, ss_initial=None, **kwargs) -> ImpulseDict`
Fully nonlinear PE response (for HetBlocks: full backward pass along the path,
then forward distribution pass). `internals=['hh']` (all) or
`{'hh': ['D','c','a']}` (selected) returns paths of individual-level objects in
`result.internals['hh']` with time as the first axis, shape `(T, *grid)`.
`ss_initial`: start the distribution from a different steady state's `Dbeg`
(transition experiments). HetBlock kwarg `monotonic=False` (set True only if
policies are known monotonic — enables a faster lottery).

### 5.6 `solve_jacobian(ss, unknowns, targets, inputs, outputs=None, T=300, Js={}) -> JacobianDict`
General-equilibrium Jacobians `G`: solves `dU = -H_U^{-1} H_Z dZ` and chains
through the DAG, returning `G[output, shock]` as `T×T` matrices for every
requested output (including the unknowns themselves).

### 5.7 `solve_impulse_linear(ss, unknowns, targets, inputs, outputs=None, Js={}) -> ImpulseDict`
GE linear impulse for given shock paths (one-shot: computes what it needs of
`G` and applies it).

### 5.8 `solve_impulse_nonlinear(ss, unknowns, targets, inputs, outputs=None, internals={}, Js={}, tol=1e-8, maxit=30, verbose=True, ss_initial=None) -> ImpulseDict`
GE nonlinear impulse via quasi-Newton on the unknown paths using the factored
steady-state `H_U`. Raises `ValueError` on non-convergence. Typical
convergence: 5–15 iterations. For transitions between steady states, `ss` is
the **terminal** steady state and `ss_initial` the initial one; input paths are
deviations from the terminal ss.

### 5.9 Options plumbing
Per-block options for any method:
`model.solve_jacobian(..., options={'hh': {'twosided': True}})`; kwargs given
directly apply to the block itself. Defaults live on each class
(`solve_impulse_nonlinear_options = dict(tol=1e-8, maxit=30, verbose=True)`).

## 6. Container classes

### `SteadyStateDict`
Dict-like: `ss['r']`, `dict(**ss)` (top level), `ss.update(...)`,
`ss.internals['hh']['D']` for HetBlock internals. Deep-copy before mutating a
calibration you want to keep.

### `ImpulseDict`
- Indexing `td['Y']` (length-`T` array), `td[['Y','C']]` (sub-ImpulseDict);
  `.T` = horizon; `.internals[block][name]` if requested.
- Arithmetic with scalars, other ImpulseDicts, and SteadyStateDicts:
  `td_level = td + ss`; `pct = 100 * td['Y'] / ss['Y']`.
- `.get(keys)` fills missing entries with zeros. `.pack()` stacks to one
  vector; `ImpulseDict.unpack(v, outputs, T)` reverses.

### `JacobianDict`
- Indexing: `G['Y', 'Z']` (matrix-like), `G['Y']` (row), `G[['Y','C'], ['Z']]`
  (sub-dict). `.outputs`, `.inputs`.
- Application: `G @ {'Z': dZ}` or `G @ impulse_dict` → `ImpulseDict`.
- Composition: `J2 @ J1` chains Jacobians. `.pack(T)` → big
  `(n_out*T, n_in*T)` matrix; `JacobianDict.unpack(X, outputs, inputs, T)`.
- HetBlock entries are dense `T×T` numpy arrays; SimpleBlock entries are
  `SimpleSparse` (support `@`, slicing like matrices after `.matrix(T)`).
- Save/reload pattern: hold in memory or `np.save` per (output, input) pair;
  pass back via `Js={'hh': J_hh}`.

### `FactoredJacobianDict`
LU-factored `H_U` for repeated solving (used internally by SolvedBlocks and
`solve_impulse_nonlinear`; expose via `H_U_factored=` argument if you manage it
yourself).

## 7. Utilities

### Grids and Markov chains (`sj.grids`)
- `asset_grid(amin, amax, n)` — double-exponentially spaced grid (dense near
  `amin`); the recommended asset grid.
- `agrid(amax, n, amin=0)` — legacy log-spaced grid (still used in examples).
- `markov_rouwenhorst(rho, sigma, N=7)` → `(y, pi, Pi)`: Rouwenhorst
  discretization of a log-AR(1); `y` is the **level** grid normalized so
  `E[y]=1` under the stationary distribution `pi`; `sigma` is the
  **cross-sectional (unconditional) std of log y**; `Pi` is the transition
  matrix. Preferred for persistent processes.
- `markov_tauchen(rho, sigma, N=7, m=3, normalize=True)` — same interface.
- In `sj.utilities.discretize`: `stationary(Pi)`, `mean/variance/std/cov/corr`
  of discretized variables, `nonlinspace(amax, n, phi, amin=0)` (flexible
  curvature, useful for non-convex problems).

### Interpolation (`sj.interpolate`)
- `interpolate_y(x, xq, y)` — linear interpolation of `y(x)` at `xq` with
  linear extrapolation, batched over leading dimensions; the EGM workhorse
  (`x` must be increasing along the last axis).
- `interpolate_coord(x, xq)` → `(i, pi)` indices/weights;
  `apply_coord(i, pi, y)` applies them — use when interpolating several arrays
  on the same query points (two-asset block does this).

### `sj.misc`
- `setmin(x, xmin)` — in-place `x = max(x, xmin)`, numba-fast; use to impose
  the borrowing constraint on the policy.

### Estimation (`sj.estimation`)
- `all_covariances(M, sigmas)`: `M` is `(T, n_O, n_Z)` stacked MA coefficients
  (IRFs of `n_O` observables to unit innovations of `n_Z` shocks), `sigmas`
  length-`n_Z` innovation stds → `(T, n_O, n_O)` autocovariances via FFT.
- `log_likelihood(Y, Sigma, sigma_measurement=None)`: `Y` is `(T_obs, n_O)`
  demeaned data → Gaussian log-likelihood via Cholesky of the stacked
  covariance matrix.

### DAG drawing (`sj.drawdag`)
`sj.drawdag(model, exogenous=[...], unknowns=[...], targets=[...], leftright=True)`
— requires graphviz (Python package + system binaries); optional, wrap in
try/except.

### Built-in household blocks (`sj.hetblocks`)
- `hh_sim.hh` — canonical one-asset standard-incomplete-markets household
  (CRRA, EGM); `hh_sim.hh_extended` adds grid/income hetinputs;
  `hh_sim.example_calibration()`.
- `hh_labor.hh` — one-asset with endogenous labor (used by the one-asset HANK
  example; handles the constrained region with a vectorized Newton solve).
- `hh_twoasset.hh` — two-asset (liquid/illiquid with convex adjustment costs;
  Auclert-Bardóczy-Rognlie-Straub two-asset HANK).

### Example models (`sj.examples`)
`krusell_smith.dag()`, `hank.dag()`, `two_asset.dag()`, `rbc.dag()` each return
`(ss_model, ss, model, unknowns, targets, exogenous)` — solved, tested
reference implementations. Read their source as templates
(`src/sequence_jacobian/examples/`).

## 8. Deprecated names

Do not write new code with: `sj.agrid`, `sj.markov_rouwenhorst`,
`sj.markov_tauchen`, `sj.interpolate_y`, `sj.setmin` at top level (all emit
deprecation warnings). Use `sj.grids.*`, `sj.interpolate.*`, `sj.misc.*`.

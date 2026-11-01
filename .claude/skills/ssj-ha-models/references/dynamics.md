# Dynamics: Jacobians, GE impulse responses, nonlinear transitions, determinacy

Contents:
1. [Partial-equilibrium Jacobians](#1-pe-jacobians)
2. [Choosing T](#2-choosing-t)
3. [GE solution and using G](#3-ge-solution)
4. [Nonlinear impulse responses and transitions](#4-nonlinear)
5. [Determinacy: the winding-number test](#5-determinacy)
6. [Decompositions and diagnostics](#6-decompositions)
7. [What linearization cannot do (ZLB, risk, asymmetry)](#7-limits)

---

## 1. PE Jacobians

```python
J_hh = household.jacobian(ss, inputs=['r', 'Z'], outputs=['C', 'A'], T=300)
J_hh['C', 'Z']            # T×T matrix of intertemporal MPCs (iMPCs)
```

- Column `s` = first-order response to *news at date 0* of a unit shock at
  date `s` ("news-shock interpretation"). Row 0 = impact responses to news at
  each horizon.
- Computation is the fake news algorithm (see SKILL.md §1): cost is one
  backward pass per input + one expectation-vector recursion per output —
  `O(T)` in the expensive dimension. The `T×T` matrix build is a single big
  matrix product.
- Differentiation is numerical: `h=1e-4`, one-sided by default. The
  implementation subtracts "ghost runs" (a zero-shock backward pass) rather
  than raw steady-state values, which removes most of the error from imperfect
  ss convergence. For final numbers, verify with `twosided=True` and compare
  (expect agreement to ~1e-6 relative; see validation.md §6).
- **Reuse Jacobians aggressively.** HetBlock Jacobians are the expensive part
  of everything downstream; compute once and pass `Js={'hh': J_hh}` to every
  `solve_jacobian`, `solve_impulse_linear`, `solve_impulse_nonlinear` call.
  Reuse is exact whenever the block's steady state has not changed.
- Economic sanity anchors: `J['C','Z']` columns for the SIM household — column
  0 decays from the impact MPC; column s>0 shows anticipation (consumption
  rises before s), spike near s, then decay. Asset Jacobian columns:
  dissaving/saving mirror image. If these look wrong, the block is wrong.

## 2. Choosing T

`T` truncates the sequence space; results are exact only as `T → ∞`.
From the paper's accuracy analysis (Section 4.2):

- Default `T = 300` (quarterly) is enough for KS and one-asset HANK with
  ρ ≤ 0.9 shocks (errors < 1e-9 relative to T=1000).
- Models with strong internal persistence (two-asset HANK: illiquid-asset
  dynamics) and/or shock persistence ρ ≥ 0.95 need `T = 500+`; with ρ = 0.99
  the two-asset model has ~3 digits of accuracy at T=300 and ~5 at T=500.
- **Rule**: both the shock path and the *endogenous responses* must be
  numerically ≈ 0 well before `T`, and the reported IRF must be insensitive to
  raising `T` by 50% (validation.md §5 has the code).
- Cost scales roughly linearly in `T` for the HetBlock step and cubically in
  `n_unknowns * T` for the `H_U` solve — raising T is usually cheap; do it.

## 3. GE solution

```python
T = 300
G = model.solve_jacobian(ss, unknowns=['w','Y','pi'], targets=['asset_mkt','goods_mkt','nkpc_res'],
                         inputs=['rstar','Z'], T=T, Js={'hh': J_hh})
irf = G @ {'rstar': -0.0025 * 0.6**np.arange(T)}      # 1pp annualized cut, ρ=0.6
irf['Y'], irf['pi']                                   # deviations from ss
```

- `G[output, shock]` are the GE `T×T` maps; applying `G` to *any* shock path is
  a matrix–vector product — compute IRFs for whole families of shocks
  (persistences, news dates, sign flips) at negligible cost.
- Column `s` of `G['Y','Z']` = response to *news* about a date-`s` shock: news
  shocks, forward guidance experiments, and anticipated-policy analysis come
  for free.
- `solve_impulse_linear(ss, unknowns, targets, inputs={...})` is the one-shot
  variant when only one path is needed. Verified identity: it equals
  `G @ shock` exactly.
- Units discipline: decide levels vs. per-cent and document. Impulses are level
  deviations; report `100 * irf['Y'] / ss['Y']` for percent. For rates, report
  percentage points (and annualize quarterly rates by ×4 when labeled annual).
- Shock definition discipline: an AR(1) "monetary shock" is
  `dr*_t = σ ρ^t`; a *news* shock is a one-time entry at date s. Say which, and
  the sign convention.

## 4. Nonlinear

```python
td = model.solve_impulse_nonlinear(ss, unknowns, targets,
                                   inputs={'Z': dZ}, Js={'hh': J_hh},
                                   internals=['hh'])       # optional: distribution paths
```

- Quasi-Newton on the unknown paths with the LU-factored steady-state `H_U`;
  each iteration is a full nonlinear evaluation of the DAG (HetBlocks: full
  backward + forward pass along the path). Converges in ~5–15 iterations for
  reasonable shocks (paper: 5 for −1pp, 13 for −5pp monetary in two-asset
  HANK). `maxit=30`, `tol=1e-8` defaults.
- Failure to converge usually means: shock too large (economy leaves the
  basin — try scaling up gradually), `T` too short (path hasn't returned to
  ss), or a genuinely explosive/indeterminate model (check winding number
  first).
- **When to use**: shocks large enough for size/sign dependence, distributional
  paths (`internals`), constraint dynamics, welfare (consumption-equivalent
  calculations need level paths: `td_lvl = td + ss`).
- **Check along the path**: the untargeted market must clear nonlinearly too:
  `np.max(np.abs(td['goods_mkt'])) < 1e-8` (verified pattern from the test
  suite).

### Transitions between steady states / permanent shocks

1. Solve the terminal steady state `ss_term` (with the permanent change in
   place) and the initial one `ss_init`.
2. Everything — Jacobians, unknowns paths, deviations — is around **ss_term**
   (the Newton method needs the Jacobian where the economy ends up;
   paper Section 6).
3. ```python
   td = model.solve_impulse_nonlinear(ss_term, unknowns, targets,
            inputs={'Z': np.zeros(T)},        # deviations from terminal ss
            ss_initial=ss_init)
   ```
   With a permanent shock that jumped at date 0, the input deviation from the
   terminal ss is identically zero; the transition is driven by the initial
   condition (`ss_initial` sets the HetBlock's starting distribution `Dbeg`).
   For a phased-in permanent change, put the phase-in path (as deviations from
   terminal values) in `inputs`.
4. `T` must exceed the full convergence time to the new ss — permanent-shock
   transitions often need `T` of 500–1000. Verify `td` ≈ 0 at the end.

## 5. Determinacy

Sequence-space analogue of Blanchard–Kahn: the `nT×nT` matrix `H_U` is
invertible (generically) iff the **winding number** of the symbol of its
asymptotic time-invariant (block-Toeplitz) part is zero
(Onatski 2006 criterion; Auclert–Rognlie–Straub determinacy paper):

- wind = 0 → generic existence + uniqueness. Proceed.
- wind < 0 → indeterminacy (multiple bounded equilibria; e.g. passive Taylor
  rule, strongly countercyclical income risk). A "solution" from `G` is then
  meaningless even if the linear algebra happens to succeed.
- wind > 0 → generic nonexistence.

Practical test with this skill's script (self-contained, adapted from the
Goethe workshop `winding_number.py`):

```python
from determinacy import winding_number_H_U        # scripts/determinacy.py
H_U = model.jacobian(ss, inputs=unknowns, outputs=targets, T=T, Js=Js)
w = winding_number_H_U(H_U, unknowns, targets, T)
assert w == 0, f"winding number {w}: not determinate!"
```

The script extracts the asymptotic Toeplitz coefficients from deep columns of
each (target, unknown) Jacobian, evaluates `det j(z)` around the unit circle by
FFT, and counts the origin crossings. Run it whenever the model has feedback
rules or you change their coefficients (Taylor φ, fiscal feedback, cyclical
risk); it costs milliseconds. Sanity anchors: canonical HANK with φ_π > 1 →
0; with φ_π < 1 → −1.

## 6. Decompositions

Direct/indirect decompositions come from applying PE Jacobians to GE paths.
Monetary example (one-asset HANK): with GE paths `dr, dY, dT ⊂ irf`,

```python
dC_direct   = J_hh['C','r'] @ irf['r']                     # direct interest-rate channel
dC_indirect = J_hh['C','Z'] @ (post_tax_income_path)       # indirect income channel
# sums to irf['C'] up to first order — verify:
assert np.allclose(dC_direct + dC_indirect, irf['C'], atol=1e-10)
```

The additivity check is mandatory — a failed sum means the decomposition
omitted an input of the household block (dividends, transfers, taxes).
Same logic gives the intertemporal-Keynesian-cross reading
`dY = dG − M dT + M dY` and firm-side MPI decompositions.

For state-dependence and distributional incidence, pull `internals` from
nonlinear impulses: `td.internals['hh']['D']` (shape `(T, *grid)`),
consumption-response heterogeneity by wealth percentile, etc.

## 7. Limits

- **ZLB / occasionally binding constraints**: not native (linearization is
  regime-blind; nonlinear Newton assumes smooth equations). Standard
  workaround: guess the binding spell, impose the constrained rule over that
  window (e.g. fix the nominal rate path via a shock to the rule), solve, and
  verify the guess (shadow rate below zero inside the window, above outside);
  iterate on the window. This is a guess-and-verify layer *you* write around
  `solve_impulse_*`.
- **Aggregate risk premia, portfolio shares, volatility shocks**: absent to
  first order. Do not report "risk effects" from a first-order SSJ solution.
  (Higher-order sequence-space methods: Goethe lecture 14; portfolio-choice
  extensions: Goethe lecture 9 — both beyond the public toolkit.)
- **Asymmetry/size-dependence**: exists only in nonlinear solutions; the linear
  solution is symmetric by construction. If the question is about asymmetry,
  everything must be `solve_impulse_nonlinear`.
- **Long-run/unit-root shocks**: a permanent shock has no finite-T
  representation as a deviation path around a fixed ss — use the
  terminal-ss + `ss_initial` machinery, never a "very persistent" AR(1) hack
  with ρ→1 at fixed T.

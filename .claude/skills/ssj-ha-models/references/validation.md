# Validation battery — run before reporting any result

Publication-quality means every number is backed by a passed check. Work
through this list top to bottom; `scripts/validation_tools.py` implements the
generic checks as functions (import and call them; they are verified against
the packaged KS and HANK models). Where a check fails, fix the cause — never
loosen a tolerance to make a report pass, and never present unchecked numbers.

## 1. Steady-state residuals

```python
for tgt, val in targets_ss.items():
    resid = ss[tgt] - (ss[val] if isinstance(val, str) else val)
    assert abs(resid) < 1e-8, (tgt, resid)
```

Report the residuals (they belong in a computational appendix). Then tighten
the HetBlock's internal tolerances one notch (`backward_tol`, `forward_tol` ÷
100) and confirm headline aggregates move by < 1e-8 — if they move more, the
baseline tolerances were binding.

## 2. Walras's law (untargeted market)

The market you dropped from targets must clear at machine-level accuracy in
the steady state (`< 1e-6`, usually ~1e-9) **and along every impulse path**:

```python
assert abs(ss['goods_mkt']) < 1e-6
td = model.solve_impulse_nonlinear(...)
assert np.max(np.abs(td['goods_mkt'])) < 1e-8
```

A violation is an accounting bug (a resource cost not in the goods market, a
dividend mis-defined, transfer incidence inconsistent). It will contaminate
IRFs quantitatively even when it looks small — find it.

## 3. Distribution health

```python
D, Dbeg = ss.internals['hh']['D'], ss.internals['hh']['Dbeg']
assert abs(D.sum() - 1) < 1e-12 and abs(Dbeg.sum() - 1) < 1e-12
assert np.all(D >= 0)
top_mass = D[..., -5:].sum()
assert top_mass < 1e-4, f"{top_mass:.2e} mass in top 5 asset points -> raise amax"
```

Also report the constrained share and check it against the calibration's
intent (a HANK model calibrated to high MPCs should show a substantial
constrained/near-constrained mass; ~0 means the calibration missed).

## 4. Grid convergence

Re-solve the steady state *and* the headline IRF with (i) double the asset
points, (ii) +2–4 income states, (iii) double `amax`. Publication bar:
steady-state aggregates move < 0.1%, IRF peak and half-life move < ~1%.
Report the comparison table. If moves are larger, increase resolution until
they are not — do not average or hand-wave.

```python
for override in [{'nA': 2*nA}, {'nS': nS+3}, {'amax': 2*amax}]:
    cali2 = {**calibration, **override}
    ss2 = ss_model.solve_steady_state(cali2, unknowns_ss, targets_ss, solver=...)
    G2 = model.solve_jacobian(ss2, unknowns, targets, inputs, T=T)
    compare_irfs(G @ shock, G2 @ shock)     # validation_tools.compare_irfs
```

## 5. Truncation horizon

Two requirements (paper, Section 4.2):
1. The shock path **and** every reported endogenous response are numerically
   ≈ 0 well before `T` (inspect the tail: `np.abs(irf['Y'][-20:]).max()`
   should be ≲ 1e-9 × the peak).
2. Raising `T` by 50% leaves the reported IRF unchanged:

```python
validation_tools.truncation_check(model, ss, unknowns, targets, inputs,
                                  shock_path_fn, T=300, T_hi=450)
# passes when max abs difference over the first T periods < 1e-8 x peak
```

Persistent models (two-asset; ρ ≥ 0.95 shocks; permanent-shock transitions)
routinely need T = 500–1000.

## 6. Jacobian accuracy (fake news vs. direct)

The fake-news Jacobian must match a direct finite-difference impulse: shock
one input by a small `h` at a single date `s`, compute the block's nonlinear
impulse, divide by `h`, compare to column `s`:

```python
validation_tools.jacobian_vs_direct(household, ss, input_='r', output='C',
                                    T=300, s=5, h=1e-5)
# relative sup-norm error should be ~1e-3 or better (both sides carry
# O(h) error; agreement degrading past ~1e-2 signals a kink or a bug)
```

Also recompute the block Jacobian with `twosided=True` and compare to the
one-sided default — for final numbers, prefer two-sided if they differ
visibly. If the direct check fails badly at some dates: suspect
non-differentiable policies (discrete jumps, hard (S,s) rules) — the
fake-news derivative is then unreliable and the model needs smoothing
(hetblocks.md §7–8).

## 7. Linear vs. nonlinear

```python
validation_tools.linear_vs_nonlinear(model, ss, unknowns, targets,
                                     {'Z': dZ}, Js=Js, scales=(1.0, 0.1, 0.01))
```

- As the shock scales to zero, nonlinear/scale must converge to the linear IRF
  — down to a *numerical floor* (~1e-4–1e-3 relative) set by the Jacobian's
  one-sided differentiation error and the Newton tolerance; do not expect the
  gap to fall forever. The tool checks "halves or hits the floor". Genuine
  failures (wrong linearization, non-convergence) produce gaps orders of
  magnitude above the floor — if you see one, stop and debug.
- At the *reported* shock size, show both linear and nonlinear responses; if
  they differ meaningfully, the paper's headline numbers should be nonlinear
  and the difference is itself a result (size dependence).

## 8. Determinacy

`scripts/determinacy.py`: winding number of `H_U` must be 0 (dynamics.md §5).
Report it. Re-run whenever a feedback-rule coefficient changes (Taylor rule
sweeps, fiscal rules, risk cyclicality). If your solution exists but wind < 0,
you have selected one of many equilibria arbitrarily — the IRFs are not
well-defined objects to report.

## 9. Economic sanity

Cheap, catches sign/timing bugs that numerics cannot:
- rate cut (Taylor-rule shock < 0) → output and inflation up in a HANK with
  φ_π > 1;
- TFP up in KS → consumption, investment, wage up; `r` down on impact... (K
  predetermined: `r` *rises* on impact with higher Z, then falls as K
  accumulates — reason it through for the actual model rather than pattern-
  matching);
- deficit-financed transfer → consumption up, more so with higher MPCs;
- iMPC matrix `M = J['C','Z']` (PE, constant `r`): column 0 declines from a
  plausible impact MPC (~0.15–0.55 quarterly depending on calibration), and
  each column satisfies the intertemporal budget constraint
  `sum_t (1+r)**(s-t) * M[t, s] ≈ 1` up to truncation error (the PDV of the
  consumption response to a date-`s` transfer equals the transfer). Check it
  explicitly for a few columns — a broken household budget constraint shows up
  here immediately.

## 10. Reproducibility & reporting

- Freeze: package versions, git commit, calibration dict (verbatim), grid
  sizes, `T`, solver names/tolerances, seeds.
- Cache expensive objects to disk (`np.savez` per Jacobian entry) with a hash
  of (ss aggregates, grid sizes, T) in the filename; stale-cache bugs are
  vicious.
- The results section of any report must state: shock definition (variable,
  size, persistence, units), horizon, linear or nonlinear, and which checks
  above were run with what outcomes. A results table without its validation
  table is not done.

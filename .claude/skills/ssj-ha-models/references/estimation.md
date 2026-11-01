# Simulation, second moments, and estimation

Everything here rests on **certainty equivalence**: to first order, the
stochastic economy's endogenous variables follow the MA(T−1) process whose
coefficients are the perfect-foresight IRFs. One `G` matrix therefore delivers
simulation, all second moments, and the exact Gaussian likelihood.

Contents:
1. [MA representation](#1-ma-representation)
2. [Simulation](#2-simulation)
3. [Analytical second moments](#3-second-moments)
4. [Likelihood](#4-likelihood)
5. [Estimation loops and Jacobian reuse](#5-estimation-loops)
6. [Bayesian estimation](#6-bayesian)
7. [IRF matching and method of moments](#7-irf-matching)
8. [Data handling](#8-data)

---

## 1. MA representation

Let shock `z` have innovation-IRF `dZ` (the response of the *shock variable*
to a unit innovation, e.g. `dZ_s = ρ^s` for an AR(1)) and let `G` be the GE
Jacobian dict. Then the MA coefficients of observable `X` w.r.t. that
innovation are:

```python
dX = G['X', 'z'] @ dZ          # length-T vector of MA coefficients
```

Stack over observables `O` and shocks `Z` into `M` with shape `(T, n_O, n_Z)`:

```python
M = np.stack([np.stack([G[o, z] @ dZ_of[z] for z in shocks], axis=1)
              for o in observables], axis=1)      # (T, n_O, n_Z)
```

Shock-process parameters (ρ, σ, ARMA terms) live entirely in `dZ_of` — `G`
never changes when they move.

## 2. Simulation

```python
sigmas = np.array([...])                     # innovation std per shock
rng = np.random.default_rng(seed)
eps = rng.standard_normal((Tsim + T, n_Z)) * sigmas
X_sim = np.empty(Tsim)
for t in range(Tsim):
    X_sim[t] = np.einsum('so,s->', M[:, i_obs, :], eps[t:t+T][::-1, :])  # or explicit loop
```

(For one shock this is `np.dot(dX, eps[t:t+T][::-1])`; see the Goethe lecture 8
notebook.) Simulation is for illustration/verification — for moments and
likelihood use the analytic formulas below (no Monte Carlo error).

## 3. Second moments

`sj.estimation.all_covariances(M, sigmas)` returns
`Sigma[l, i, j] = Cov(X_{i,t}, X_{j,t−l})` for lags `l = 0..T−1`, computed by
FFT (exact given the MA(T−1) truncation):

```python
from sequence_jacobian import estimation
Sigma = estimation.all_covariances(M, sigmas)       # (T, n_O, n_O)
sd = np.sqrt(np.diag(Sigma[0]))
corr_l = Sigma[l] / np.outer(sd, sd)                # cross-correlations at lag l
autocorr_i = Sigma[:, i, i] / Sigma[0, i, i]
```

Verify once against a long simulation (they must agree up to Monte Carlo
error) — this is a powerful end-to-end bug catcher.

## 4. Likelihood

```python
L = estimation.log_likelihood(Y, Sigma, sigma_measurement)   # Y: (T_obs, n_O), demeaned
```

Internally: stacks `Sigma` into the `(n_O·T_obs)²` covariance matrix `V` (with
iid measurement error variance on the diagonal), Cholesky-factorizes, and
returns `−½ log det V − ½ Y'V⁻¹Y` (constants dropped). Cost scales with
`(n_O · T_obs)³` — fine for typical macro samples (7 observables × 150–250
quarters); for very long samples consider the Whittle approximation (not in
the toolkit).

Requirements: `T_obs ≤` effective MA length; `n_shocks ≥ n_observables` for a
nonsingular `V` unless measurement error is included (stochastic singularity,
exactly as in DSGE estimation).

## 5. Estimation loops

The reuse hierarchy (paper Section 5.4) — exploit it explicitly in code
structure:

| Parameters being varied | What to recompute per draw | Typical cost |
|---|---|---|
| Shock processes (ρ, σ, ARMA) | `dZ`, `M`, `Sigma`, `L` only — `G` fixed | ~ms |
| Non-ss model params (κ, φ, adjustment costs, rule coefficients) | Jacobians of the *changed simple/solved blocks* + re-solve `G`; HetBlock `Js` fixed | ~10–200 ms |
| ss-relevant params (β, income risk, borrowing limits) | Steady state + HetBlock Jacobians + `G` | seconds+ |

Skeleton:

```python
J_hh = household.jacobian(ss, inputs=[...], outputs=[...], T=T)   # once

def loglik(theta):
    shock_params, model_params = split(theta)
    if model_params_changed:                       # tier 2
        G = model.solve_jacobian(ss, unknowns, targets, inputs=shocks, T=T,
                                 Js={'hh': J_hh})  # cheap: hh J reused
    M = build_M(G, shock_params)                   # tier 1
    Sigma = estimation.all_covariances(M, sigmas(shock_params))
    return estimation.log_likelihood(Y, Sigma, sigma_meas)
```

For tier-2 parameters that enter one simple block (e.g. Taylor φ), update the
calibration copy `ss2 = deepcopy(ss); ss2['phi'] = phi` and call
`solve_jacobian` on `ss2` — simple-block Jacobians are recomputed analytically
at negligible cost; the reused `Js` skips the HetBlock. This is valid because
φ does not change the steady state (rule is at its ss point); think before
applying the same trick to any parameter — *it must not move the ss*.

## 6. Bayesian

Standard recipe (paper Section 5.4, all reproducible with scipy):
- Priors: Inverse-Gamma(mean 0.4, sd 4) for shock σ's; Beta(0.5, 0.2) for
  persistences; model-specific priors for structural parameters. Transform
  bounded parameters (log for σ, logit for ρ) or use L-BFGS-B bounds.
- Posterior mode: `scipy.optimize.minimize(neg_log_post, x0, method='L-BFGS-B',
  bounds=...)` starting from the prior mode.
- Posterior: Random-Walk Metropolis–Hastings; proposal = multivariate normal
  with covariance = inverse Hessian at the mode (finite-difference it), scaled
  to hit ~25% acceptance; 100k–200k draws, discard half. Report recursive
  means and acceptance rate as convergence diagnostics.
- Full worked implementation: `nber-workshop-2023/Tutorials/Tutorial 3
  Estimating HANK.ipynb` with `estimation/routines.py` (mode-finding, RWMH,
  plots) and `us_data.csv`; updated variants in the Goethe lecture 8 notebook
  (`goethe-workshop-2026/notebooks/lecture8_stochastic.ipynb` + estimation/).

## 7. IRF matching

With an empirical IRF `irf_hat` (e.g. identified monetary shock) and
model IRF `IRF(θ) = G(θ) @ dZ(θ)`:

```python
def objective(theta):
    err = model_irf(theta)[:H] - irf_hat[:H]
    return err @ np.linalg.solve(V_hat, err)      # V_hat: empirical IRF covariance
```

Same reuse logic applies. This is the CEE/Christiano-style strategy; it is also
the standard way to calibrate σ (EIS) and adjustment costs to the peak
responses (Goethe lecture 11 practice). Method of (second) moments: minimize
the distance between `Sigma`-implied moments and empirical ones with a weight
matrix.

## 8. Data

- Demean/detrend exactly as the model concept requires: linearly detrend log
  real quantities; demean inflation and nominal rates (paper follows
  Smets–Wouters 1966:1–2004:4 conventions). The model is in deviations; the
  data must be too.
- Match units: if `Y` is percent deviation ×100, `M` rows must be too.
- Frequency: quarterly model ⇒ quarterly data; annualized rates ÷4 (or model
  rates ×4 — one direction, consistently).
- Missing data/mixed frequency: the stacked-covariance approach extends by
  dropping rows (time-specific selection matrix); implement by subsetting `V`
  and `Y` jointly.
- Always plot data vs. a simulated path at the estimated parameters — a
  10-second visual check that catches unit errors instantly.

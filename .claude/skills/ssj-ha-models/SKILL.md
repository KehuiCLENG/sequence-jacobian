---
name: ssj-ha-models
description: Build, solve, estimate, and rigorously validate heterogeneous-agent macroeconomic models (Krusell-Smith, one- and two-asset HANK, heterogeneous firms, open economy, and variants) using the Sequence-Space Jacobian (SSJ) method and the sequence_jacobian Python toolkit. Use this skill whenever the task involves heterogeneous agents, incomplete markets, HANK, sequence-space Jacobians, the fake news algorithm, MIT shocks or perfect-foresight transitions, intertemporal MPCs, HetBlocks or macro-model DAGs, general-equilibrium impulse responses, winding-number determinacy tests, or estimation/simulation of HA models — even if the user does not name the toolkit. Also use it when writing, reviewing, or debugging any code that imports sequence_jacobian, and when setting up a Python environment for HA-macro work.
---

# Solving Heterogeneous-Agent Models with Sequence-Space Jacobians

This skill guides you through building and solving general-equilibrium
heterogeneous-agent (HA) models — heterogeneous households and/or firms — with
the SSJ method of Auclert, Bardóczy, Rognlie & Straub (2021, *Econometrica* 89(5),
2375–2408) and its official Python toolkit, `sequence_jacobian` (the repo
`shade-econ/sequence-jacobian`). The target standard is unsupervised,
publication-quality work: every number you report must be validated, every
approximation quantified, and every convention handled exactly as the toolkit
defines it.

**Read order.** Read this file fully. Then, before each stage of work, read the
matching reference file listed in [Reference file index](#reference-file-index).
Do not code a HetBlock, a steady state, or an estimation from memory of other
codebases — the conventions here are specific and violating them produces
silently wrong results.

## 0. Environment first

Nothing works without a verified environment. Follow
[references/environment.md](references/environment.md). Summary:

- If working inside the `sequence-jacobian` repo, a venv may already exist at
  `.venv/` (Python 3.12, numpy/scipy/numba/matplotlib/pandas, package installed
  editable). Verify with `python scripts/check_env.py` (in this skill's
  `scripts/` folder). Expect "ALL CHECKS PASSED" in well under a minute.
- Otherwise create a venv (Python ≥ 3.8; 3.11/3.12 recommended) and
  `pip install sequence-jacobian` (plus matplotlib, pandas, pytest).
- First `import sequence_jacobian` triggers numba JIT compilation (a few
  seconds); this is normal and happens once per process.

## 1. The method in one page (get this right before coding)

**Sequence space.** Solve for perfect-foresight paths of aggregates over a
truncated horizon `T` (typically 300–500 quarters), not for state-space policy
rules. By certainty equivalence, the first-order perfect-foresight impulse
responses to date-`s` shocks are exactly the MA(∞) coefficients of the model
with aggregate risk — so linear IRFs, simulation, second moments, and
likelihood all come from the same objects.

**Blocks and the DAG.** A model is a set of *blocks*, each mapping sequences of
aggregate inputs to sequences of aggregate outputs, arranged in a directed
acyclic graph. Agents may interact only through aggregates — that is the key
restriction of the method. Block types in the toolkit:

| Block | Decorator / constructor | Use for |
|---|---|---|
| `SimpleBlock` | `@sj.simple` | Aggregate equations with leads/lags: production, fiscal rules, market clearing, Taylor rules |
| `HetBlock` | `@sj.het(...)` | Heterogeneous households or firms: one backward-iteration function |
| `SolvedBlock` | `@sj.solved(...)` or `block.solved(...)` | Implicit forward-looking conditions solved internally (NKPC, Q-theory investment, debt rules) — a mini-model embedded as a block |
| `CombinedBlock` | `sj.combine([...])` / `sj.create_model([...])` | The assembled DAG; supports every method a block does |

**Equilibrium.** Choose aggregate *unknowns* `U` (e.g. `K`; or `w, Y, pi`) and
an equal number of *targets* `H` (market-clearing/optimality residuals that
must be zero, e.g. `asset_mkt`, `nkpc_res`). The DAG collapses to
`H(U, Z) = 0` where `Z` are exogenous shock paths. To first order,
`dU = -H_U^{-1} H_Z dZ`, and every other variable follows by the chain rule
along the DAG. The toolkit automates all of this: `solve_jacobian` returns the
GE Jacobians `G` mapping any `dZ` to any `dX`.

**HetBlock Jacobians — the fake news algorithm.** For an HA block, the `T×T`
Jacobian of each output w.r.t. each input is computed in `O(T)` backward/forward
passes instead of `O(T²)`: (1) one backward sweep from a shock at date `T−1`
gives all perturbed policies and the date-0 aggregate responses `Y_s` and
date-1 distribution responses `D_s`; (2) expectation vectors
`E_t = (Λ_ss)^t y_ss` come from repeated application of the steady-state
transition matrix; (3) the fake-news matrix is `F[0,s]=Y_s`,
`F[t,s]=E_{t−1}'D_s`; (4) `J[t,s] = J[t−1,s−1] + F[t,s]`. You never implement
this yourself — `HetBlock.jacobian` does — but you must understand it to debug
and to judge accuracy (differentiation is numerical, default one-sided with
step `h=1e-4`; two-sided is ~1–2 orders of magnitude more accurate).

**Nonlinear transitions.** `solve_impulse_nonlinear` runs a quasi-Newton
iteration on the full nonlinear path using the *steady-state* `H_U` as the
Jacobian; it typically converges in 5–15 iterations. For permanent shocks,
compute the *terminal* steady state, linearize there, and pass the initial
steady state via `ss_initial`.

**Scope limits (be honest about these).** The framework requires that
individual value functions depend on the future only through aggregate inputs
`X_t` and their own continuation value — the value function may *not* depend
directly on the distribution `D_t`. Excluded without extensions: OLG with
endogenous mid-life bequest distributions, wage-posting/individual-bargaining
search, money search with distribution-dependent pricing. Entry/exit and some
OLG structures need the extended framework of the paper's Appendix A (not fully
automated in the public toolkit — see
[references/hetblocks.md](references/hetblocks.md) for what is and is not
expressible). Being first-order in aggregates, the method delivers no risk
premia, no time-varying precautionary effects of *aggregate* risk, and no
occasionally-binding aggregate constraints (a ZLB spell needs special handling;
see [references/dynamics.md](references/dynamics.md)).

## 2. Standard workflow

Work through these stages in order. Never skip stage 7.

### Stage 1 — Design the DAG on paper
Write down: blocks, their inputs/outputs, the exogenous shocks, the unknowns,
and the targets. Count them: #unknowns = #targets, and by Walras's law drop one
market (keep it as an *untargeted* output — it becomes a free accuracy check).
Prefer few unknowns (n ≤ 3 is typical): solving cost is dominated by inverting
the `nT×nT` matrix `H_U`, and a good DAG "solves out" intermediate variables.
Read [references/steady-state.md](references/steady-state.md) §1 for
unknown/target selection heuristics and worked examples (KS, one-asset HANK,
two-asset HANK).

### Stage 2 — Write the blocks
- Simple blocks: plain Python functions of aggregates, decorated `@sj.simple`;
  Dynare-style time shifts `K(-1)`, `pi(+1)`; wrap nonlinear functions of
  shifted expressions with `.apply(...)`, e.g. `(1+pi).apply(np.log)`.
- Het blocks: one backward-step function under `@sj.het(exogenous=...,
  policy=..., backward=..., backward_init=...)`, plus `hetinputs` (map
  aggregates to individual-level inputs, e.g. grids and incomes) and
  `hetoutputs` (individual-level outcomes to aggregate). Start from a canonical
  template in [references/hetblocks.md](references/hetblocks.md) — it contains
  verified one-asset EGM, endogenous-labor, two-asset, multi-exogenous-state,
  permanent-type, life-cycle, and heterogeneous-firm patterns — and the full
  naming/ordering conventions (see §4 pitfalls below).
- Solved blocks: for forward-looking implicit equations you'd otherwise have to
  add as unknowns/targets at the top level.
Exact signatures and semantics of everything: [references/api.md](references/api.md).

### Stage 3 — Steady state and calibration
Assemble `ss_model = sj.create_model([...])`, possibly with steady-state-only
helper blocks that analytically invert calibration targets (e.g. back out `Z, K`
from targets for `Y, r`). Then:
```python
ss = ss_model.solve_steady_state(calibration, unknowns_ss, targets_ss, solver=...)
```
Solver choice, unknown formats (bounds vs. initial values), `dissolve=` for
SolvedBlocks, distributional calibration (Lorenz/MPC targets), and the two-DAG
pattern (separate ss and dynamics DAGs, then `ss = model.steady_state(cali)`)
are in [references/steady-state.md](references/steady-state.md).
**Immediately check**: all targets < 1e-8; the untargeted market clears
(typically < 1e-6, pure accounting error otherwise); distribution mass near the
top of the asset grid is negligible.

### Stage 4 — Jacobians and the GE solution
```python
T = 300                                    # justify; see truncation checks
G = model.solve_jacobian(ss, unknowns, targets, inputs=exogenous, T=T)
irf = G @ {'Z': 0.01 * 0.9**np.arange(T)}  # any shock path, instantly
```
Compute expensive HetBlock Jacobians once and reuse them everywhere via
`Js={'hh': J_hh}`. Details, timing conventions, news shocks, decompositions:
[references/dynamics.md](references/dynamics.md).

### Stage 5 — Determinacy
Before trusting any linear solution in a model with feedback rules (Taylor
rules, fiscal rules, cyclical risk), run the winding-number test on `H_U`:
winding number 0 ⇒ (generic) existence & uniqueness; < 0 ⇒ indeterminacy;
> 0 ⇒ nonexistence. Use `scripts/determinacy.py` (self-contained, verified).
Theory and interpretation: [references/dynamics.md](references/dynamics.md) §5.

### Stage 6 — Nonlinear IRFs, transitions, estimation (as needed)
- Nonlinear: `model.solve_impulse_nonlinear(ss, unknowns, targets, inputs=...)`;
  compare against linear at the same shock size and report both.
- Permanent shocks / regime changes: terminal-ss linearization + `ss_initial`.
- Simulation, second moments, likelihood, Bayesian estimation:
  [references/estimation.md](references/estimation.md). The key economy: shock-
  process parameters reuse `G` unchanged; non-steady-state parameters reuse the
  HetBlock `Js`; only steady-state-relevant parameters force full recomputation.

### Stage 7 — Validation battery (non-negotiable)
Run the full checklist in [references/validation.md](references/validation.md)
before reporting any result. Minimum bar for publication-quality output:
1. steady-state target residuals and untargeted-market residual reported;
2. grid-convergence check (finer asset grid, more income states, higher amax);
3. truncation check (`T` vs `1.5T` IRFs agree; shock *and* endogenous response
   ≈ 0 well before `T`);
4. fake-news Jacobian vs. direct finite-difference impulse (columns agree);
5. linear vs. nonlinear IRF comparison at the reported shock size;
6. determinacy (winding number = 0);
7. untargeted market clears along the nonlinear path (< 1e-8);
8. sign/economics sanity checks.
`scripts/validation_tools.py` implements the reusable checks; the reference
file shows how to interpret and report them.

## 3. Decision guide

| Situation | Do this |
|---|---|
| Forward-looking aggregate equation (NKPC, Tobin's Q, arbitrage) | `@sj.solved` block, or add unknown+target at top level (equivalent; solved blocks keep the top-level system small) |
| Backward-looking aggregate identity | `@sj.simple` block |
| Need vector/grid objects from aggregates (grids, tax schedules, incidence rules) | `hetinput` functions on the HetBlock |
| Need individual-level derived outcomes aggregated (labor supply `e·n`, `uce`, adjustment costs) | `hetoutput` functions |
| 1 ss unknown with a bracket | `solver='brentq'`, unknowns `{'beta': (lo, hi)}` |
| ≥2 ss unknowns | `solver='broyden_custom'` (or `'hybr'`), scalar initial values; 3-tuples `(lb, x0, ub)` if bounds needed |
| Discount-factor heterogeneity, multiple agent types | `remap` + `rename` copies of the HetBlock + an aggregation `@simple` block |
| Shock to an exogenous transition matrix (job-finding rate, risk shocks) | Make `Pi` (or its inputs) outputs of a hetinput; fully supported in Jacobians |
| Persistent model (two-asset, ρ ≥ 0.95 shocks) | Raise `T` to 500+; verify with truncation check |
| Large shock, binding constraints matter | `solve_impulse_nonlinear`, report vs. linear |
| Permanent shock | Solve terminal ss; linearize/Newton there; `ss_initial=` initial ss |
| Estimating shock processes only | Compute `G` once; likelihood loop never re-solves the model |
| Occasionally-binding constraint (ZLB) | Not native; see dynamics.md §7 for the news-shock/guess-and-verify approach |

## 4. Critical conventions — violating any of these gives wrong answers

1. **HetBlock state ordering**: arrays are shaped exogenous states first (in
   the order listed in `exogenous=`), then endogenous policy states (in the
   order listed in `policy=`). E.g. `(n_e, n_a)` or `(n_z, n_b, n_a)`.
2. **Grid naming**: each policy `x` requires an input named exactly `x_grid`.
   Policy `'a'` → `a_grid`. The toolkit looks it up by this name.
3. **Case convention**: individual-level outputs of the backward function are
   lowercase (`a`, `c`, `n`); the toolkit aggregates them against the
   distribution `D` and exposes them capitalized (`A`, `C`, `N`) as block
   outputs. Never return uppercase individual variables.
4. **`_p` suffix**: the backward function receives `Va_p` = the *expectation of
   next period's* backward variable conditional on today's exogenous state —
   the toolkit applies the Markov matrix for you. Do not apply `Pi` inside the
   backward function.
5. **Backward variable**: must be both an argument (`Va_p`) and an output
   (`Va`) of the backward function; `Va` is defined on today's
   (exog, endog-state) grid, e.g. `Va = (1+r) * c**(-1/eis)` by the envelope
   condition. Names `D`/`Dbeg` are reserved.
6. **Distribution timing**: within period t: `Dbeg_t` (over last period's
   income state and assets chosen last period) —[exog Markov]→ `D_t` (today's
   exog state × beginning-of-period assets; policies and aggregation use `D`)
   —[policy lottery]→ `Dbeg_{t+1}`. Aggregates are `np.vdot(D_t, y_t)`.
7. **SimpleBlock timing**: `K(-1)` is capital chosen at t−1 and used at t —
   predetermined variables always appear lagged where used (production
   `Y = Z * K(-1)**alpha * L**(1-alpha)`). Returns realized between t−1 and t
   follow the model's own convention — be explicit about ex-ante vs. ex-post
   rates (e.g. a `returns` block with `r = r_ante(-1)`).
8. **`.apply`**: inside `@simple`/`@solved` functions, `np.log(x(+1))` fails;
   write `x(+1).apply(np.log)`. Plain arithmetic (`+ - * / **`) works directly.
9. **Two policies max** per HetBlock, ordered consistently in `policy=`,
   `backward=`, and grid dimensions (see the two-asset template).
10. **Jacobian differentiation**: default one-sided, `h=1e-4`. For final
    results or when accuracy is questioned, recompute with `twosided=True`
    and compare.
11. **Vector-valued ss objects** (grids, `e_grid`, Markov matrices) are
    automatically excluded from aggregate outputs/Jacobians; do not use them
    as unknowns or targets.
12. **Impulses are deviations**: everything returned by `impulse_*` and
    `solve_impulse_*` is in deviations from the (terminal) steady state; add
    `ss` back for levels (`td_lvl = td + ss`). Inputs you pass are also
    deviations, added to ss values internally.
13. **Integer parameters** passed through calibration dicts (grid sizes `nA`,
    `nS`) may arrive as floats inside hetinputs; `markov_rouwenhorst` casts
    internally, but cast explicitly (`int(n)`) in your own grid constructors.
14. **Don't shadow toolkit names**: block function names become block names in
    the DAG; keep them unique across the model.

## 5. Quality bar for unsupervised work

- Never present a number you have not validated per Stage 7. If a check fails,
  fix the model or say plainly that it fails and why — do not bury it.
- Report, at minimum: calibration table with targets vs. achieved values;
  steady-state residuals; `T`, grid sizes, and their convergence checks; IRFs
  with the shock clearly defined (size, persistence, units); linear-vs-nonlinear
  comparison when shocks are large; determinacy statement.
- Quarterly calibration is the default in this literature; state the frequency
  and keep rates/persistence consistent with it (r ≈ 0.005–0.0125 quarterly).
- Seed all simulations; record package versions (`pip freeze`) with results.
- Prefer reusing the toolkit's tested components (`sj.hetblocks.hh_sim`,
  `sj.grids`, `sj.interpolate`) over hand-rolled ones; if you must hand-roll,
  test against the toolkit equivalent.

## Reference file index

| File | Read when |
|---|---|
| [references/environment.md](references/environment.md) | Setting up or verifying the Python environment |
| [references/api.md](references/api.md) | Any coding against `sequence_jacobian` — full verified API |
| [references/hetblocks.md](references/hetblocks.md) | Writing/adapting a household or firm HetBlock |
| [references/steady-state.md](references/steady-state.md) | Calibration and steady-state solving |
| [references/dynamics.md](references/dynamics.md) | Jacobians, GE IRFs, nonlinear transitions, determinacy |
| [references/estimation.md](references/estimation.md) | Simulation, moments, likelihood, Bayesian estimation |
| [references/validation.md](references/validation.md) | Before reporting any result |
| [references/resources.md](references/resources.md) | Locating notebooks, lectures, papers (local repo + online) |

Scripts (run with the project venv's python):
- `scripts/check_env.py` — end-to-end environment verification (solves KS + HANK).
- `scripts/determinacy.py` — winding-number determinacy test for `H_U`.
- `scripts/validation_tools.py` — reusable validation checks (Jacobian vs.
  direct, truncation, linear-vs-nonlinear, market clearing).

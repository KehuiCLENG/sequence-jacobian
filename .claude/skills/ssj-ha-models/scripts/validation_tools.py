"""Reusable validation checks for SSJ heterogeneous-agent models.

Import these into a model project and run the full battery described in
references/validation.md before reporting results. All functions return a
dict of diagnostics and (optionally) assert against a tolerance; they raise
AssertionError with an informative message on failure when `strict=True`.

Verified against the packaged Krusell-Smith and one-asset HANK examples.
"""
import numpy as np


def market_clearing_check(ss, targets, untargeted=(), tol_target=1e-8,
                          tol_untargeted=1e-6, strict=True):
    """Steady-state residuals of targets and untargeted (Walras) markets.

    targets: dict like {'asset_mkt': 0.} or {'B': 'Bh'}; untargeted: names
    whose ss value should be ~0 by Walras's law."""
    out = {}
    for k, v in targets.items():
        ref = ss[v] if isinstance(v, str) else v
        out[k] = float(ss[k] - ref)
        if strict:
            assert abs(out[k]) < tol_target, f"target {k} residual {out[k]:.2e}"
    for k in untargeted:
        out[k] = float(ss[k])
        if strict:
            assert abs(out[k]) < tol_untargeted, \
                f"untargeted market {k} residual {out[k]:.2e} (accounting bug?)"
    return out


def distribution_check(ss, block_name, top_points=5, tol_top=1e-4, strict=True):
    """Distribution sums to 1, is nonnegative, and has negligible mass at the
    top of the (last) endogenous grid."""
    D = ss.internals[block_name]['D']
    out = {'mass': float(D.sum()), 'min': float(D.min()),
           'top_mass': float(D[..., -top_points:].sum()),
           'constrained_mass': float(D[..., 0].sum())}
    if strict:
        assert abs(out['mass'] - 1) < 1e-10, out
        assert out['min'] > -1e-14, out
        assert out['top_mass'] < tol_top, \
            f"mass {out['top_mass']:.2e} in top {top_points} grid points -> raise amax"
    return out


def jacobian_vs_direct(block, ss, input_, output, T, s=5, h=1e-5, J=None,
                       tol_rel=1e-2, strict=True, **jac_kwargs):
    """Fake-news Jacobian column vs direct finite-difference nonlinear impulse.

    Shocks `input_` by h at date s only, computes the block's nonlinear
    impulse, divides by h, and compares to column s of J[output, input_].
    Relative sup-norm error should be small (~1e-3; both objects carry O(h)
    differentiation error)."""
    if J is None:
        J = block.jacobian(ss, inputs=[input_], outputs=[output], T=T, **jac_kwargs)
    col = np.asarray(J[output, input_])[:, s] if not hasattr(J[output, input_], 'matrix') \
        else J[output, input_].matrix(T)[:, s]

    shock = np.zeros(T)
    shock[s] = h
    td = block.impulse_nonlinear(ss, {input_: shock}, outputs=[output])
    direct = td[output] / h

    scale = np.max(np.abs(col))
    rel_err = float(np.max(np.abs(direct - col)) / scale)
    if strict:
        assert rel_err < tol_rel, \
            (f"J[{output},{input_}] column {s} disagrees with direct method: "
             f"rel err {rel_err:.2e} (kinked policies? bug?)")
    return {'rel_err': rel_err, 'scale': float(scale)}


def linear_vs_nonlinear(model, ss, unknowns, targets, inputs, Js={},
                        scales=(1.0, 0.1, 0.01), key=None, strict=True,
                        floor_tol=1e-3):
    """Nonlinear IRF divided by shock scale must converge to the linear IRF as
    the scale goes to 0. Returns {scale: sup-norm relative gap} for `key`
    (default: first unknown).

    Convergence is limited by a numerical floor: the linear Jacobian carries
    O(h) one-sided differentiation error (h=1e-4) and the Newton solve stops
    at a finite tolerance (scaled with the shock here). The strict check
    therefore requires the smallest-scale gap to either halve relative to the
    largest-scale gap or lie below `floor_tol`; genuine failures (wrong
    linearization, non-convergence) produce gaps orders of magnitude larger."""
    key = key or unknowns[0]
    lin = model.solve_impulse_linear(ss, unknowns, targets, inputs, Js=Js)
    denom = np.max(np.abs(lin[key]))
    gaps = {}
    for sc in sorted(scales, reverse=True):
        scaled = {k: sc * np.asarray(v) for k, v in dict(inputs).items()}
        nonlin = model.solve_impulse_nonlinear(ss, unknowns, targets, scaled,
                                               Js=Js, verbose=False,
                                               tol=max(1e-12, 1e-8 * sc))
        gaps[sc] = float(np.max(np.abs(nonlin[key] / sc - lin[key])) / denom)
    if strict:
        ordered_scales = sorted(gaps, reverse=True)
        g_large, g_small = gaps[ordered_scales[0]], gaps[ordered_scales[-1]]
        assert (g_small < 0.5 * g_large) or (g_small < floor_tol), \
            f"nonlinear does not converge to linear as shock -> 0: {gaps}"
    return gaps


def truncation_check(model, ss, unknowns, targets, inputs, shock_fn,
                     T=300, T_hi=450, outputs=None, Js_fn=None,
                     tol_rel=1e-6, strict=True):
    """Compare GE IRFs computed at horizons T and T_hi over the first T
    periods. shock_fn(T) -> dict of shock paths of length T. Js must be
    horizon-specific: optionally pass Js_fn(T) -> Js dict; otherwise Jacobians
    are recomputed at each horizon (do NOT reuse a T-specific Js at T_hi)."""
    out = {}
    irfs = {}
    for TT in (T, T_hi):
        Js = Js_fn(TT) if Js_fn is not None else {}
        G = model.solve_jacobian(ss, unknowns, targets, inputs, T=TT, Js=Js)
        irfs[TT] = G @ shock_fn(TT)
    keys = outputs or [k for k in irfs[T] if k in irfs[T_hi]]
    for k in keys:
        peak = np.max(np.abs(irfs[T][k]))
        if peak == 0:
            continue
        out[k] = float(np.max(np.abs(irfs[T][k] - irfs[T_hi][k][:T])) / peak)
        if strict:
            assert out[k] < tol_rel, \
                f"IRF of {k} not converged in T: rel gap {out[k]:.2e} -> raise T"
    return out


def impc_budget_check(M, r, cols=(0, 5, 10), tol=1e-2, strict=True):
    """Intertemporal budget constraint on an iMPC matrix M = J['C','Z'] from a
    PE household block with constant interest rate r: for each column s,
    sum_t (1+r)^(s-t) M[t,s] ~= 1 (up to truncation error, which is larger for
    high s)."""
    M = np.asarray(M)
    T = M.shape[0]
    out = {}
    for s in cols:
        disc = (1 + r) ** (s - np.arange(T))
        out[s] = float(disc @ M[:, s])
        if strict:
            assert abs(out[s] - 1) < tol, \
                f"iMPC column {s} discounted sum {out[s]:.4f} != 1 (budget constraint bug?)"
    return out


def compare_irfs(irf1, irf2, keys=None, normalize=True):
    """Sup-norm differences between two ImpulseDicts (optionally relative to
    the peak of irf1). Returns {key: gap}."""
    keys = keys or [k for k in irf1 if k in irf2]
    out = {}
    for k in keys:
        a, b = np.asarray(irf1[k]), np.asarray(irf2[k])
        n = min(len(a), len(b))
        gap = np.max(np.abs(a[:n] - b[:n]))
        peak = np.max(np.abs(a[:n]))
        out[k] = float(gap / peak) if (normalize and peak > 0) else float(gap)
    return out

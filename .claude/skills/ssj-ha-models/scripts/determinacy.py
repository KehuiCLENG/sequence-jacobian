"""Winding-number determinacy test for sequence-space models.

Sequence-space analogue of Blanchard-Kahn (Onatski 2006; Auclert-Rognlie-
Straub, "Determinacy and Existence in the Sequence Space"): the Jacobians of a
stationary heterogeneous-agent model are quasi-Toeplitz, J = T(j) + E with E
compact. H_U is (generically) invertible on l^2 iff the winding number of
det j(z) around the origin, as z traverses the unit circle, is zero:

    wind = 0   -> generic existence & uniqueness (proceed)
    wind < 0   -> indeterminacy (multiple bounded equilibria)
    wind > 0   -> generic nonexistence

Usage:
    H_U = model.jacobian(ss, inputs=unknowns, outputs=targets, T=T, Js=Js)
    w = winding_number_H_U(H_U, unknowns, targets, T)

The FFT sampling and path-counting routines follow the Goethe 2026 workshop's
`winding_number.py` (shade-econ), which implements the method of the
determinacy paper; the H_U extraction wrapper and robustness check are added
here.
"""
import numpy as np

try:
    from numba import njit
except ImportError:                                   # pragma: no cover
    def njit(f):
        return f


def winding_number_H_U(H_U, unknowns, targets, T, t0=None, m=None, K=8192,
                       check_robustness=True):
    """Winding number of det j(z) for the block-Toeplitz part of H_U.

    Parameters
    ----------
    H_U : JacobianDict with H_U[target, unknown] entries (T x T)
    unknowns, targets : lists of names, equal length n
    T : truncation horizon used to compute H_U
    t0 : column at which to read Toeplitz coefficients (default T//2)
    m : one-sided number of Laurent coefficients (default min(t0, T-t0-1),
        capped at T//3)
    K : number of unit-circle sample points (power of 2, > 2m+1)
    check_robustness : recompute at half m and shifted t0; raise if unstable

    Returns
    -------
    int winding number
    """
    n = len(unknowns)
    if len(targets) != n:
        raise ValueError("unknowns and targets must have equal length")
    if t0 is None:
        t0 = T // 2
    if m is None:
        m = min(t0, T - t0 - 1, T // 3)

    j = _extract_symbol(H_U, unknowns, targets, T, t0, m)
    w = _winding_number_symbol(j, K)

    if check_robustness:
        j2 = _extract_symbol(H_U, unknowns, targets, T, t0, m // 2)
        j3 = _extract_symbol(H_U, unknowns, targets, T, t0 + T // 8, m // 2)
        w2, w3 = _winding_number_symbol(j2, K), _winding_number_symbol(j3, K)
        if not (w == w2 == w3):
            raise RuntimeError(
                f"Winding number unstable across truncations ({w}, {w2}, {w3}): "
                f"raise T (Jacobian's Toeplitz part has not converged).")
    return w


def _extract_symbol(H_U, unknowns, targets, T, t0, m):
    """(2m+1, n, n) Laurent coefficients j_k = H_U[t0+k, t0], k = -m..m."""
    n = len(unknowns)
    j = np.zeros((2 * m + 1, n, n))
    for a, tgt in enumerate(targets):
        for b, unk in enumerate(unknowns):
            try:
                Jab = H_U[tgt, unk]
            except (KeyError, ValueError):
                continue                                # structurally zero
            Jab = Jab.matrix(T) if hasattr(Jab, 'matrix') else np.asarray(Jab)
            j[:, a, b] = Jab[t0 - m: t0 + m + 1, t0]
    return j


def _winding_number_symbol(j, K=8192):
    e = sample_values(j, K)
    if np.min(np.abs(e)) < 1e-12 * np.max(np.abs(e)):
        raise RuntimeError("det j(z) ~ 0 on the unit circle: borderline "
                           "determinacy, winding number ill-defined.")
    return winding_number_of_path(e.real.copy(), e.imag.copy())


def sample_values(j, K=8192):
    """Evaluate det j(z) counterclockwise at K evenly spaced roots of unity,
    wrapped back to z=1. j is (2T-1,) or (2T-1, n, n) Laurent coefficients
    (powers -(T-1)..(T-1)). [Goethe workshop winding_number.py]"""
    assert (K % 2 == 0) and (len(j) % 2 == 1) and (K > len(j))
    T = len(j) // 2 + 1
    if j.ndim == 3:
        assert j.shape[1] == j.shape[2]

    jhat = np.zeros((K, *j.shape[1:])) if j.ndim == 3 else np.zeros(K)
    jhat[(K // 2 - (T - 1)):(K // 2 + T)] = j

    x = np.fft.rfft(jhat, axis=0)                 # z^(K/2)*j(z) at clockwise roots

    alt = np.tile([1, -1], K // 2 + 1)[:K // 2 + 1]   # divide by z^(K/2) = +/-1
    if j.ndim == 3:
        x = np.linalg.det(x * alt[:, np.newaxis, np.newaxis])
    else:
        x = x * alt

    return np.concatenate((x.conj(), x[:-1][::-1]))   # conj symmetry, counterclockwise


@njit
def winding_number_of_path(x, y):
    """Winding number around origin of closed path (x, y), counting
    counterclockwise crossings of the positive x-axis ray.
    [Goethe workshop winding_number.py]"""
    assert x[-1] == x[0] and y[-1] == y[0]
    w = 0
    s = (y[0] >= 0)
    for k in range(1, len(x)):
        if (y[k] >= 0) != s:
            s = (y[k] >= 0)
            if x[k] > 0 and x[k - 1] > 0:
                w += 2 * s - 1
            elif not (x[k] <= 0 and x[k - 1] <= 0):
                crossed_on_right = (x[k - 1] * y[k] - x[k] * y[k - 1]) / (y[k] - y[k - 1])
                if crossed_on_right > 0:
                    w += 2 * s - 1
    return w


if __name__ == '__main__':
    # Self-test on packaged models: KS must be determinate (wind = 0);
    # HANK is determinate at phi=1.5 and indeterminate at phi=0.7.
    from sequence_jacobian.examples import krusell_smith, hank

    T = 300
    _, ss, ks, unknowns, targets, _ = krusell_smith.dag()
    H_U = ks.jacobian(ss, inputs=unknowns, outputs=targets, T=T)
    w = winding_number_H_U(H_U, unknowns, targets, T)
    print(f"Krusell-Smith winding number: {w}")
    assert w == 0

    _, ss_h, hk, unknowns_h, targets_h, _ = hank.dag()
    H_U = hk.jacobian(ss_h, inputs=unknowns_h, outputs=targets_h, T=T)
    w_active = winding_number_H_U(H_U, unknowns_h, targets_h, T)
    print(f"one-asset HANK (phi=1.5) winding number: {w_active}")
    assert w_active == 0

    import copy
    ss_p = copy.deepcopy(ss_h)
    ss_p['phi'] = 0.7                       # passive Taylor rule (phi does not change the ss)
    H_U_p = hk.jacobian(ss_p, inputs=unknowns_h, outputs=targets_h, T=T)
    w_passive = winding_number_H_U(H_U_p, unknowns_h, targets_h, T)
    print(f"one-asset HANK (phi=0.7) winding number: {w_passive}")
    assert w_passive < 0

    print("determinacy.py self-test PASSED")

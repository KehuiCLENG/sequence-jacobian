"""Environment verification for SSJ heterogeneous-agent work.

Run with the project venv's python. Solves the packaged Krusell-Smith and
one-asset HANK models end to end with assertions. Must end with
'ALL CHECKS PASSED'. Expected runtime ~10-30s (numba compilation dominates).
"""
import sys
import time

t0 = time.time()

import numpy as np
import scipy
import numba
import sequence_jacobian as sj
from sequence_jacobian.examples import krusell_smith, hank

print(f"python  {sys.version.split()[0]}")
print(f"numpy   {np.__version__}")
print(f"scipy   {scipy.__version__}")
print(f"numba   {numba.__version__}")
print(f"ssj     from {sj.__file__}")

# --- Krusell-Smith: steady state -> GE Jacobian -> linear & nonlinear IRF ---
ks_ss_model, ss, ks_model, unknowns, targets, inputs = krusell_smith.dag()
assert abs(ss['asset_mkt']) < 1e-8, ss['asset_mkt']
assert abs(ss['goods_mkt']) < 1e-7, ss['goods_mkt']
print(f"[{time.time()-t0:5.1f}s] KS steady state OK (r={ss['r']:.4f}, K={ss['K']:.4f})")

T = 300
G = ks_model.solve_jacobian(ss, unknowns, targets, inputs, T=T)
dZ = 0.01 * 0.8 ** np.arange(T)
irf_lin = G @ {'Z': dZ}
irf_nonlin = ks_model.solve_impulse_nonlinear(ss, unknowns, targets, {'Z': dZ}, verbose=False)
gap = np.max(np.abs(irf_nonlin['Y'] - irf_lin['Y']))
assert gap < 5e-4, gap        # small shock => nonlinear ~ linear
print(f"[{time.time()-t0:5.1f}s] KS GE Jacobian + IRFs OK (|nonlin-lin| = {gap:.1e})")

# --- One-asset HANK: monetary shock has the right sign ---
hank_ss_model, ss_h, hank_model, unknowns_h, targets_h, exog_h = hank.dag()
assert abs(ss_h['asset_mkt']) < 1e-8
G_h = hank_model.solve_jacobian(ss_h, unknowns_h, targets_h, exog_h, T=T)
irf_h = G_h @ {'rstar': -0.0025 * 0.61 ** np.arange(T)}
assert irf_h['Y'][0] > 0 and irf_h['pi'][0] > 0    # expansionary rate cut
print(f"[{time.time()-t0:5.1f}s] HANK GE Jacobian + monetary IRF OK "
      f"(dY0={irf_h['Y'][0]:.4f}, dpi0={irf_h['pi'][0]:.4f})")

print(f"[{time.time()-t0:5.1f}s] ALL CHECKS PASSED")

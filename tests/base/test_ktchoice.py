"""Finite-difference check of the Khan-Thomas random-fixed-cost stage
(KTChoice): backward_step_shock must match central differences of
backward_step for shocks to the post-choice value V, to the other backward
variables (Vs, Vk) and to the wage w, including the choice-probability
perturbation dP0 = 1{interior}[ddV/(w kbar) - P0 dw/w] and the envelope
dEV = P0 dV0 + (1-P0) dV1 - (kbar/2) P0^2 dw.

Convention under test: `precompute` and `backward_step_shock` receive the
stage's backward INPUTS (the toolkit stores a stage's backward inputs under
its name), so ss['V'] is the post-choice value on (nz, 2, ns, nk)."""
import numpy as np

from sequence_jacobian.blocks.support.stages import KTChoice


def _flow(V):
    flowu = np.zeros((2, V.shape[0], 1) + V.shape[2:])
    return flowu


def test_ktchoice_matches_finite_differences():
    rng = np.random.default_rng(0)
    nz, ns, nk = 3, 5, 4
    w, kbar = 0.6, 0.02
    V = rng.normal(size=(nz, 2, ns, nk)) * 0.05 + 1.0
    # spread dV = V0 - V1 over the interior (0, w kbar) and both corners
    dV = rng.uniform(-0.5, 1.5, size=(nz, ns, nk)) * w * kbar
    V[:, 1] = V[:, 0] - dV
    Vs = rng.normal(size=(nz, 2, ns, nk))
    Vk = rng.normal(size=(nz, 2, ns, nk))

    st = KTChoice(value='V', backward=['Vs', 'Vk'], index=1,
                  taste_shock_scale='varsig_eps', f=_flow, name='adjust',
                  wage_input='w', kbar_input='kt_kbar')
    ss_in = {'V': V, 'Vs': Vs, 'Vk': Vk, 'w': w, 'kt_kbar': kbar,
             'varsig_eps': 0.02}
    out, lom = st.backward_step(ss_in, lawofmotion=True)
    assert np.allclose(lom.P.sum(axis=0), 1.0)
    P0 = lom.P[0]
    assert 0.2 < np.mean((P0 > 0) & (P0 < 1)) < 0.8, 'test needs interior cells'
    pre = st.precompute(dict(ss_in), lom)

    def fd(name, dx, h=1e-6):
        up, dn = dict(ss_in), dict(ss_in)
        up[name] = ss_in[name] + h * dx
        dn[name] = ss_in[name] - h * dx
        o_up, l_up = st.backward_step(up, lawofmotion=True)
        o_dn, l_dn = st.backward_step(dn, lawofmotion=True)
        d = {k: (o_up[k] - o_dn[k]) / (2 * h) for k in o_up}
        return d, (l_up.P - l_dn.P) / (2 * h)

    for name in ('V', 'Vs', 'Vk', 'w'):
        dx = rng.normal(size=np.shape(ss_in[name])) if name != 'w' else 1.0
        an, dlom = st.backward_step_shock(dict(ss_in), {name: dx}, pre)
        num, dP = fd(name, dx)
        for k in an:
            scale = max(np.max(np.abs(num[k])), 1e-8)
            assert np.max(np.abs(an[k] - num[k])) < 1e-6 * scale, (name, k)
        scale = max(np.max(np.abs(dP)), 1e-8)
        assert np.max(np.abs(dlom.P - dP)) < 1e-6 * scale, (name, 'P')

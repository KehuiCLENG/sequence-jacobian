"""Regression test: StageBlock with the EXOGENOUS stage LAST in the
period cycle (delayed-revelation timing, e.g. a draw revealed after
decisions commit).

Bug (pre-fix): in backward_step_fakenews, the last stage is processed
FIRST in the backward pass, with no backward feedback yet (empty
dback). For a shocked input that does not touch the exogenous stage
(here r, atw: they enter only the continuous choice stage), the
Exogenous stage's backward_step_shock returned an empty dict, and
stage_block.py's unconditional
    dback = {k: dout[k] for k in stage.backward_outputs}
raised KeyError ('Va' here). Exog-FIRST blocks never hit this because
their exogenous stage is processed last, when dback is populated.

Fix under test: sparse dback (absent key = zero perturbation) plus a
zeros default for a missing value shock in LogitChoice.

Correctness assertions beyond crash-freedom:
1. CYCLE-ROTATION INVARIANCE: [exog, choice] and [choice, exog] are
   rotations of the same stationary cycle, so steady states coincide
   and the fake-news Jacobians w.r.t. shocks entering the CHOICE
   stage (r, atw) must be identical.
2. FD CHECK: the 'shift' (transition-matrix) Jacobian of the
   exog-last block agrees with impulse_nonlinear finite differences
   (impulse_nonlinear does not use backward_step_shock, so it is an
   independent reference).
"""
import numpy as np

from sequence_jacobian.blocks.stage_block import StageBlock
from sequence_jacobian.hetblocks.hh_sim import hh_init
from sequence_jacobian.blocks.support.stages import Continuous1D, ExogenousMaker
from sequence_jacobian import interpolate, grids, misc
from sequence_jacobian.classes import ImpulseDict


def make_grids(rho_e, sd_e, nE, amin, amax, nA):
    e_grid, e_dist, Pi_ss = grids.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nE)
    a_grid = grids.agrid(amin=amin, amax=amax, n=nA)
    return e_grid, e_dist, Pi_ss, a_grid


def alter_Pi(Pi_ss, shift):
    Pi = Pi_ss.copy()
    Pi[:, 0] -= shift
    Pi[:, -1] += shift
    return Pi


def income(atw, N, e_grid, transfer):
    y = atw * N * e_grid + transfer
    return y


def household_new(Va, a_grid, y, r, beta, eis):
    uc_nextgrid = beta * Va
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    a = interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    misc.setmin(a, a_grid[0])
    c = coh - a
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c


CALIB = {'r': 0.004, 'eis': 0.5, 'rho_e': 0.91, 'sd_e': 0.92, 'nE': 3,
         'amin': 0.0, 'amax': 200, 'nA': 100, 'transfer': 0.143, 'N': 1,
         'atw': 1, 'beta': 0.97, 'shift': 0}


def _blocks():
    het_a = Continuous1D(backward='Va', policy='a', f=household_new,
                         name='choice')
    exog_first = StageBlock([ExogenousMaker('Pi', 0, 'exog_e'), het_a],
                            name='hh', backward_init=hh_init,
                            hetinputs=(make_grids, income, alter_Pi))
    het_b = Continuous1D(backward='Va', policy='a', f=household_new,
                         name='choice')
    exog_last = StageBlock([het_b, ExogenousMaker('Pi', 0, 'exog_e')],
                           name='hh', backward_init=hh_init,
                           hetinputs=(make_grids, income, alter_Pi))
    return exog_first, exog_last


def test_exog_last_jacobian():
    b_first, b_last = _blocks()
    ss1 = b_first.steady_state(CALIB)
    ss2 = b_last.steady_state(CALIB)

    # rotation invariance of the stationary cycle
    assert np.isclose(ss1['A'], ss2['A'])
    assert np.isclose(ss1['C'], ss2['C'])
    assert np.allclose(ss1.internals['hh']['choice']['D'],
                       ss2.internals['hh']['choice']['D'])

    T = 100
    # pre-fix this line raises KeyError 'Va' for inputs r/atw (the
    # exogenous stage is processed first with empty dback)
    J2 = b_last.jacobian(ss2, ['r', 'atw', 'shift'], ['A', 'C'], T)
    J1 = b_first.jacobian(ss1, ['r', 'atw'], ['A', 'C'], T)

    # 1. rotation invariance for shocks entering the choice stage
    # (equality is exact in exact arithmetic; the two blocks reach
    # their fixed points by independent iterations, so compare
    # relative to the Jacobian scale — observed agreement ~7e-9)
    for i in ('r', 'atw'):
        for o in ('A', 'C'):
            scale = np.max(np.abs(J1[o, i]))
            assert np.max(np.abs(J2[o, i] - J1[o, i])) < 1e-6 * scale, (o, i)

    # 2. FD check of the exog-last Jacobians (impulse_nonlinear is an
    # independent reference: it does not use backward_step_shock).
    # StageBlocks carry an ambient ~4e-4 relative lin-vs-nonlinear
    # gap even in the long-standing exog-first configuration (same
    # household, measured here as the reference), so the bar is
    # comparative: exog-last must not be worse than 2x the exog-first
    # gap, plus an absolute 1e-3 relative cap.
    h = 1e-4
    J1s = b_first.jacobian(ss1, ['r', 'shift'], ['A', 'C'], T)
    dx = 0.9 ** np.arange(T)
    for i in ('r', 'shift'):
        td1 = b_first.impulse_nonlinear(
            ss1, ImpulseDict({i: h * dx}), outputs=['A', 'C'])
        td2 = b_last.impulse_nonlinear(
            ss2, ImpulseDict({i: h * dx}), outputs=['A', 'C'])
        for o in ('A', 'C'):
            ref = np.max(np.abs(J1s[o, i] @ dx - td1[o] / h))
            gap = np.max(np.abs(J2[o, i] @ dx - td2[o] / h))
            scale = max(np.max(np.abs(J2[o, i] @ dx)), 1.0)
            assert gap < max(2 * ref, 1e-6), (o, i, gap, ref)
            assert gap < 1e-3 * scale, (o, i, gap, scale)

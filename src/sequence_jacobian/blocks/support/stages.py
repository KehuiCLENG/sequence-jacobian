from typing import List, Optional
import numpy as np
import copy

# from sequence_jacobian.blocks.support.het_support import DiscreteChoice
from sequence_jacobian.blocks.support.law_of_motion import DiscreteChoice
from ...utilities.function import ExtendedFunction, CombinedExtendedFunction
from ...utilities.ordered_set import OrderedSet
from ...utilities.misc import make_tuple, logit_choice
from .law_of_motion import (lottery_1d, ShockedPolicyLottery1D,
                            lottery_2d, ShockedPolicyLottery2D,
                            MixtureLottery,
                            Markov)

class Stage:
    def backward_step(self, inputs, lawofmotion=False):
        pass

    def backward_step_shock(self, ss, shocks, precomputed=None):
        pass

    def precompute(self, ss, ss_lawofmotion=None):
        pass

    def backward_step_separate(self, inputs, lawofmotion=False, hetoutputs=False):
        """Wrapper around backward_step that also obtains hetoutputs,
        and returns backward, report, and sometimes lom separately"""
        outputs = self.backward_step(inputs, lawofmotion)
        if lawofmotion:
            outputs, lom = outputs
        
        backward_outputs = {k: outputs[k] for k in self.backward_outputs}
        report = {k: outputs[k] for k in self.original_report}

        if hetoutputs and self.hetoutputs is not None:
            inputs = {**inputs, **outputs}
            #report.update(self.hetoutputs(all_inputs))
            # for some reason self.hetoutputs returns its inputs too, need to fix that
            report.update({k: v for k, v in self.hetoutputs(inputs).items() if k in self.hetoutputs.outputs})

        if lawofmotion:
            return (backward_outputs, report), lom
        else:
            return backward_outputs, report

    def __init__(self, hetoutputs=None):
        # instance variables of a stage:
        # self.name = ""
        # self.backward_outputs = OrderedSet([])
        # self.report = OrderedSet([])
        # self.inputs = OrderedSet([])

        self.original_inputs = self.inputs.copy()
        self.original_report = self.report.copy()

        if hetoutputs is not None:
            hetoutputs = CombinedExtendedFunction(hetoutputs)
        self.process_hetoutputs(hetoutputs, tocopy=False)

    def process_hetoutputs(self, hetoutputs: Optional[CombinedExtendedFunction], tocopy=True):
        if tocopy:
            self = copy.copy(self)
        self.inputs = self.original_inputs.copy()
        self.report = self.original_report.copy()

        if hetoutputs is not None:
            self.inputs |= (hetoutputs.inputs - self.report - self.backward_outputs)
            self.report |= hetoutputs.outputs

        self.hetoutputs = hetoutputs

        return self

    def add_hetoutputs(self, functions):
        if self.hetoutputs is None:
            return self.process_hetoutputs(CombinedExtendedFunction(functions))
        else:
            return self.process_hetoutputs(self.hetoutputs.add(functions))

    def remove_hetoutputs(self, names):
        return self.process_hetoutputs(self.hetoutputs.remove(names))

    # def return_hetinputs(self, d):
    #     if self.hetinputs is not None:
    #         return self.hetinputs(d)
    #     else:
    #         return {}

class Continuous1D(Stage):
    """Stage that does one-dimensional endogenous continuous choice"""
    def __init__(self, backward, policy, f, name=None, hetoutputs=None):
        # subclass-specific attributes
        self.f = ExtendedFunction(f)
        self.policy = policy

        # attributes needed for any stage
        if name is None:
            name = self.f.name
        self.name = name
        self.backward_outputs = OrderedSet(make_tuple(backward))
        self.report = self.f.outputs - self.backward_outputs
        self.inputs = self.f.inputs

        super().__init__(hetoutputs)

    def __repr__(self):
        return f"<Stage-Continuous1D '{self.name}' with policy '{self.policy}'>"

    def backward_step(self, inputs, lawofmotion=False):
        outputs = self.f(inputs)

        if not lawofmotion:
            return outputs
        else:
            # TODO: option for monotonic?!
            return outputs, lottery_1d(outputs[self.policy], inputs[self.policy + '_grid'], monotonic=False)
    
    def backward_step_shock(self, ss, shocks, precomputed):
        space, i, grid, f = precomputed
        outputs = f.diff(shocks)
        dpi = -outputs[self.policy] / space
        return outputs, ShockedPolicyLottery1D(i, dpi, grid)

    def precompute(self, ss, ss_lawofmotion):
        i = ss_lawofmotion.i.reshape(ss_lawofmotion.shape)
        grid = ss_lawofmotion.grid
        return grid[i + 1] - grid[i], i, grid, self.f.differentiable(ss)


class Continuous2D(Stage):
    """Stage that does two-dimensional endogenous continuous choice"""
    def __init__(self, backward, policy, f, name=None, hetoutputs=None):
        # subclass-specific attributes
        self.f = ExtendedFunction(f)
        self.policy = OrderedSet(policy)

        # attributes needed for any stage
        if name is None:
            name = self.f.name
        self.name = name
        self.backward_outputs = OrderedSet(make_tuple(backward))
        self.report = self.f.outputs - self.backward_outputs
        self.inputs = self.f.inputs

        super().__init__(hetoutputs)

    def __repr__(self):
        return f"<Stage-Continuous2D '{self.name}' with policies {self.policy}>"

    def backward_step(self, inputs, lawofmotion=False):
        outputs = self.f(inputs)

        if not lawofmotion:
            return outputs
        else:
            # TODO: option for monotonic?!
            return outputs, lottery_2d(outputs[self.policy[0]], outputs[self.policy[1]],
                                       inputs[self.policy[0] + '_grid'], inputs[self.policy[1] + '_grid'])
    
    def backward_step_shock(self, ss, shocks, precomputed):
        space1, space2, i1, i2, pi1, pi2, grid1, grid2, f = precomputed
        outputs = f.diff(shocks)
        dpi1 = -outputs[self.policy[0]] / space1
        dpi2 = -outputs[self.policy[1]] / space2
        # the 2D lottery derivative needs steady-state weights of BOTH
        # coordinates (d(L1 x L2) = dL1 x L2 + L1 x dL2)
        return outputs, ShockedPolicyLottery2D(i1, pi1, i2, pi2,
                                               grid1, grid2, dpi1, dpi2)

    def precompute(self, ss, ss_lawofmotion):
        i1 = ss_lawofmotion.i1.reshape(ss_lawofmotion.shape)
        i2 = ss_lawofmotion.i2.reshape(ss_lawofmotion.shape)
        pi1 = ss_lawofmotion.pi1.reshape(ss_lawofmotion.shape)
        pi2 = ss_lawofmotion.pi2.reshape(ss_lawofmotion.shape)
        grid1 = ss_lawofmotion.grid1
        grid2 = ss_lawofmotion.grid2

        return (grid1[i1 + 1] - grid1[i1], grid2[i2 + 1] - grid2[i2],
                i1, i2, pi1, pi2, grid1, grid2, self.f.differentiable(ss))


class Continuous2DKernel(Continuous2D):
    """Continuous2D whose FIRST policy dimension is smeared by a
    finite stochastic kernel: the landing of policy[0] decomposes as

        landing_m = retained * mult_m + (pol1 - retained),  m = 1..M

    where `retained` and the kernel (mults, weights, dmults) are
    supplied by a user callable `kernel_fn(inputs)` returning
        (retained, mults, weights, dmults, dsig_name)  or  None.
    None (or a single unit node) reproduces the base class exactly —
    the deterministic-nesting gate relies on this. `dmults` is the
    derivative of the node multipliers w.r.t. the aggregate input
    named `dsig_name` (e.g. 'siglag'), used by backward_step_shock:
    the kernel contribution to the landing derivative is
    retained * dmults_m * d(input), added to the policy derivative
    per node (the dPi-class shocked-transition pattern)."""

    def __init__(self, backward, policy, f, kernel_fn, name=None,
                 hetoutputs=None):
        super().__init__(backward, policy, f, name, hetoutputs)
        self.kernel_fn = kernel_fn

    def __repr__(self):
        return (f"<Stage-Continuous2DKernel '{self.name}' with "
                f"policies {self.policy}>")

    def backward_step(self, inputs, lawofmotion=False):
        outputs = self.f(inputs)
        if not lawofmotion:
            return outputs
        ker = self.kernel_fn(inputs)
        grid1 = inputs[self.policy[0] + '_grid']
        grid2 = inputs[self.policy[1] + '_grid']
        if ker is None:
            return outputs, lottery_2d(outputs[self.policy[0]],
                                       outputs[self.policy[1]],
                                       grid1, grid2)
        retained, mults, weights, _, _ = ker
        pol1 = outputs[self.policy[0]]
        pol2 = outputs[self.policy[1]]
        inflow = pol1 - retained
        loms = [lottery_2d(np.clip(retained * um + inflow,
                                   grid1[0], grid1[-1]),
                           pol2, grid1, grid2) for um in mults]
        return outputs, MixtureLottery(weights, loms)

    def precompute(self, ss, ss_lawofmotion):
        ker = self.kernel_fn(ss)
        if ker is None:
            return ('base', super().precompute(ss, ss_lawofmotion))
        retained, mults, weights, dmults, dsig = ker
        if not isinstance(ss_lawofmotion, MixtureLottery):
            raise ValueError('kernel active but ss law of motion is '
                             'not a mixture — inconsistent state')
        pieces = []
        for lom_m in ss_lawofmotion.loms:
            i1 = lom_m.i1.reshape(lom_m.shape)
            i2 = lom_m.i2.reshape(lom_m.shape)
            pi1 = lom_m.pi1.reshape(lom_m.shape)
            pi2 = lom_m.pi2.reshape(lom_m.shape)
            pieces.append((lom_m.grid1[i1 + 1] - lom_m.grid1[i1],
                           lom_m.grid2[i2 + 1] - lom_m.grid2[i2],
                           i1, i2, pi1, pi2,
                           lom_m.grid1, lom_m.grid2))
        return ('kernel', (pieces, retained, mults, weights, dmults,
                           dsig, self.f.differentiable(ss)))

    def backward_step_shock(self, ss, shocks, precomputed):
        mode, data = precomputed
        if mode == 'base':
            return super().backward_step_shock(ss, shocks, data)
        (pieces, retained, mults, weights, dmults, dsig,
         fdiff) = data
        outputs = fdiff.diff(shocks)
        dpol1 = outputs[self.policy[0]]
        dpol2 = outputs[self.policy[1]]
        dsig_amt = shocks.get(dsig, 0.0)
        shocked = []
        for m, (space1, space2, i1, i2, pi1, pi2, g1, g2) in \
                enumerate(pieces):
            # landing_m = retained*mult_m + (pol1 - retained):
            # d(landing_m) = dpol1 + retained*dmults_m*d(sig input)
            dland = dpol1 + retained * dmults[m] * dsig_amt
            dpi1 = -dland / space1
            dpi2 = -dpol2 / space2
            shocked.append(ShockedPolicyLottery2D(
                i1, pi1, i2, pi2, g1, g2, dpi1, dpi2))
        return outputs, MixtureLottery(weights, shocked)


class ExogenousMaker:
    """Call make_stage with backward returned by next stage to get Exogenous stage"""
    def __init__(self, markov_name, index, name=None, hetoutputs=None):
        self.markov_name = markov_name
        self.index = index
        if name is None:
            name = f"exog_{markov_name}"
        self.name = name
        self.hetoutputs = hetoutputs

    def make_stage(self, backward):
        return Exogenous(self.markov_name, self.index, self.name, backward, self.hetoutputs)


class Exogenous(Stage):
    """Stage that applies exogenous Markov process along one dimension"""
    def __init__(self, markov_name, index, name, backward, hetoutputs=None):
        # subclass-specific attributes
        self.markov_name = markov_name
        self.index = index

        # attributes needed for any stage
        self.name = name
        self.backward_outputs = backward
        self.report = OrderedSet([])
        self.inputs = backward | [markov_name]

        super().__init__(hetoutputs)

    def __repr__(self):
        return f"<Stage-Exogenous '{self.name}' with Markov matrix '{self.markov_name}'>"
    
    def backward_step(self, inputs, lawofmotion=False):
        Pi = Markov(inputs[self.markov_name], self.index)
        outputs = {k: Pi @ inputs[k] for k in self.backward_outputs}

        if not lawofmotion:
            return outputs
        else:
            return outputs, Pi.T
    
    def backward_step_shock(self, ss, shocks, precomputed=None):
        Pi = Markov(ss[self.markov_name], self.index)
        outputs = {k: Pi @ shocks[k] for k in self.backward_outputs if k in shocks}

        if self.markov_name in shocks:
            dPi = Markov(shocks[self.markov_name], self.index)
            for k in self.backward_outputs:
                if k in outputs:
                    outputs[k] += dPi @ ss[k]
                else:
                    outputs[k] = dPi @ ss[k]
            return outputs, dPi.T
        else:
            return outputs, None


class LogitChoice(Stage):
    """Stage that does endogenous discrete choice with type 1 extreme value taste shocks"""
    def __init__(self, value, backward, index, taste_shock_scale, f=None, name=None, hetoutputs=None):
        # flow utility function, if present, should return a single output
        if f is not None:
            f = ExtendedFunction(f)
            if not len(f.outputs) == 1:
                raise ValueError(f'Flow utility function {f.name} returning multiple outputs {f.outputs}')
            self.f = f
        else:
            self.f = None

        # other subclass-specific attributes
        self.index = index
        self.value = value
        self.backward = OrderedSet(make_tuple(backward))
        self.taste_shock_scale = taste_shock_scale

        # attributes needed for any stage
        if name is None:
            name = self.f.name
        self.name = name
        self.backward_outputs = self.backward | [value]
        self.report = OrderedSet([])
        self.inputs = self.backward | [value, taste_shock_scale]
        if f is not None:
            self.inputs |= f.inputs

        super().__init__(hetoutputs)

    def __repr__(self):
        return f"<Stage-Discrete '{self.name}'>"

    def backward_step(self, inputs, lawofmotion=False):
        # start with value we're given
        V_next = inputs[self.value]

        # add dimension at beginning to allow for choice, then swap (today's choice determines next stages's state)
        V = V_next[np.newaxis, ...]
        V = np.swapaxes(V, 0, self.index+1)

        # call f if we have it to get flow utility
        if self.f is not None:
            flow_u = self.f(inputs)
            flow_u = next(iter(flow_u.values()))
        else:
            # create phantom state variable, convenient but bit wasteful
            nchoice = V.shape[0]
            flow_u = np.zeros((nchoice,) + V_next.shape)

        V = flow_u + V
        
        # calculate choice probabilities and expected value
        P, EV = logit_choice(V, inputs[self.taste_shock_scale])
        
        # make law of motion, use it to take expectations of everything else
        lom = DiscreteChoice(P, self.index)

        # take expectations
        outputs = {k: lom.T @ inputs[k] for k in self.backward}
        outputs[self.value] = EV

        if not lawofmotion:
            return outputs
        else:
            return outputs, lom

    def backward_step_shock(self, ss, shocks, precomputed):
        """See 'discrete choice math' note for background. Note that scale is inverse of 'c' in that note."""
        f, lom = precomputed

        # this part parallel to backward_step, just with derivatives...
        # (missing value shock means zero, e.g. when the shocked input
        # has not yet propagated to this stage's backward inputs)
        dV_next = shocks.get(self.value, np.zeros_like(ss[self.value]))
        dV = dV_next[np.newaxis, ...]
        dV = np.swapaxes(dV, 0, self.index+1)

        if f is not None:
            dflow_u = f.diff(shocks)
            dflow_u = next(iter(dflow_u.values()))
            dflow_u = np.nan_to_num(dflow_u)  # -inf - (-inf) = nan, want zeros
        else:
            dflow_u = np.zeros_like(lom.P)
        
        dV = dflow_u + dV

        # simply take expectations to get shock to expected value function (envelope result)
        dEV = np.sum(lom.P * dV, axis=0)

        # calculate shocks to choice probabilities (note nifty broadcasting of dEV)
        scale = ss[self.taste_shock_scale]
        dP = lom.P * (dV - dEV) / scale
        dlom = DiscreteChoice(dP, self.index)

        # find shocks to outputs, aggregate everything of interest
        doutputs = {self.value: dEV}
        for k in self.backward:
            doutputs[k] = dlom.T @ ss[k]
            if k in shocks:
                doutputs[k] += lom.T @ shocks[k]
        
        return doutputs, dlom

    def precompute(self, ss, ss_lawofmotion):
        f = self.f.differentiable(ss) if self.f is not None else None
        return f, ss_lawofmotion

class KTChoice(LogitChoice):
    """Khan-Thomas random-fixed-cost binary choice stage (2026-07-21,
    GSZ_LITE_DERIVATION sec. 12). Replaces the logit taste shock
    with an iid REAL fixed cost kappa ~ U[0, kbar] (labor units),
    paid on adjustment (choice index 0). The threshold kstar =
    (V0 - V1)/w gives
        P0 = clip(kstar/kbar, 0, 1)
        EV = V1 + P0*(V0 - V1) - (w*kbar/2)*P0**2
    (the last term = the expected paid cost; dEV/dV0 = P0 and
    dEV/dV1 = 1 - P0 by the threshold envelope — verified by hand,
    sec. 12 — and dEV/dw = -(kbar/2)*P0^2).
    Inputs: `wage_input` (default 'w') and `kbar_input` (default
    'kt_kbar') must be block inputs. Binary choice only (nchoice=2).
    The lom contract (P applies to every backward variable) is kept,
    so downstream carry designs (the pi-grid) work verbatim."""

    def __init__(self, *args, wage_input='w', kbar_input='kt_kbar',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.wage_input = wage_input
        self.kbar_input = kbar_input
        self.inputs = (self.inputs | {wage_input, kbar_input}) \
            - {self.taste_shock_scale}

    def __repr__(self):
        return f"<Stage-KTChoice '{self.name}'>"

    def _pev(self, V, w, kbar):
        assert V.shape[0] == 2, "KTChoice is a binary-choice stage"
        dV = V[0] - V[1]
        kstar = dV / (w * kbar)
        P0 = np.clip(kstar, 0.0, 1.0)
        EV = V[1] + P0 * dV - 0.5 * (w * kbar) * P0 ** 2
        interior = (kstar > 0.0) & (kstar < 1.0)
        return P0, EV, dV, interior

    def backward_step(self, inputs, lawofmotion=False):
        V_next = inputs[self.value]
        V = V_next[np.newaxis, ...]
        V = np.swapaxes(V, 0, self.index + 1)
        if self.f is not None:
            flow_u = self.f(inputs)
            flow_u = next(iter(flow_u.values()))
        else:
            nchoice = V.shape[0]
            flow_u = np.zeros((nchoice,) + V_next.shape)
        V = flow_u + V
        w = inputs[self.wage_input]
        kbar = inputs[self.kbar_input]
        P0, EV, _, _ = self._pev(V, w, kbar)
        P = np.stack([P0, 1.0 - P0], axis=0)
        lom = DiscreteChoice(P, self.index)
        outputs = {k: lom.T @ inputs[k] for k in self.backward}
        outputs[self.value] = EV
        if not lawofmotion:
            return outputs
        return outputs, lom

    def backward_step_shock(self, ss, shocks, precomputed):
        f, lom, w, kbar, dVlev, interior = precomputed
        dV_next = shocks.get(self.value, np.zeros_like(ss[self.value]))
        dV = dV_next[np.newaxis, ...]
        dV = np.swapaxes(dV, 0, self.index + 1)
        if f is not None:
            dflow_u = f.diff(shocks)
            dflow_u = next(iter(dflow_u.values()))
            dflow_u = np.nan_to_num(dflow_u)
        else:
            dflow_u = np.zeros_like(lom.P)
        dV = dflow_u + dV
        P0 = lom.P[0]
        dw = shocks.get(self.wage_input, 0.0)
        ddV = dV[0] - dV[1]
        # the envelope: dEV = P0 dV0 + (1-P0) dV1 - (kbar/2) P0^2 dw
        dEV = P0 * dV[0] + (1.0 - P0) * dV[1] \
            - 0.5 * kbar * P0 ** 2 * dw
        # dP0 = 1{interior} [ ddV/(w kbar) - P0 dw/w ]
        dP0 = np.where(interior,
                       ddV / (w * kbar) - P0 * dw / w, 0.0)
        dP = np.stack([dP0, -dP0], axis=0)
        dlom = DiscreteChoice(dP, self.index)
        doutputs = {self.value: dEV}
        for k in self.backward:
            doutputs[k] = dlom.T @ ss[k]
            if k in shocks:
                doutputs[k] += lom.T @ shocks[k]
        return doutputs, dlom

    def precompute(self, ss, ss_lawofmotion):
        f = self.f.differentiable(ss) if self.f is not None else None
        V_next = ss[self.value]
        V = V_next[np.newaxis, ...]
        V = np.swapaxes(V, 0, self.index + 1)
        if self.f is not None:
            flow_u = self.f(ss)
            flow_u = next(iter(flow_u.values()))
            V = flow_u + V
        w = ss[self.wage_input]
        kbar = ss[self.kbar_input]
        _, _, dVlev, interior = self._pev(V, w, kbar)
        return f, ss_lawofmotion, w, kbar, dVlev, interior

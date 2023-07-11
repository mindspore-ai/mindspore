# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""line search"""
from typing import NamedTuple
from mindspore import ops
from ... import nn
from ... import numpy as mnp
from ...common import dtype as mstype
from ...common import Tensor
from ..utils import _to_scalar, _to_tensor, grad


class _LineSearchResults(NamedTuple):
    """Results of line search results.

    Args:
        failed (bool): `True`` if the strong Wolfe criteria were satisfied
        nit (int): number of iterations
        nfev (int): number of functions evaluations
        ngev (int): number of gradients evaluations
        k (int): number of iterations
        a_k (float): step size
        f_k (float): final function value
        g_k (Tensor): final gradient value
        status (int): end status
    """
    failed: bool
    nit: int
    nfev: int
    ngev: int
    k: int
    a_k: float
    f_k: float
    g_k: Tensor
    status: int


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    """Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
    """
    db = b - a
    dc = c - a
    denom = (db * dc) ** 2 * (db - dc)

    d1 = mnp.zeros((2, 2))
    d1[0, 0] = dc ** 2
    d1[0, 1] = -db ** 2
    d1[1, 0] = -dc ** 3
    d1[1, 1] = db ** 3

    d2 = mnp.zeros((2,))
    d2[0] = fb - fa - fpa * db
    d2[1] = fc - fa - fpa * dc

    a2, b2 = mnp.dot(d1, d2) / denom
    radical = b2 * b2 - 3. * a2 * fpa
    xmin = a + (-b2 + mnp.sqrt(radical)) / (3. * a2)
    return xmin


def _quadmin(a, fa, fpa, b, fb):
    """Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa.
    """
    db = b - a
    b2 = (fb - fa - fpa * db) / (db ** 2)
    xmin = a - fpa / (2. * b2)
    return xmin


def _zoom(fn, a_low, phi_low, dphi_low, a_high, phi_high, dphi_high, phi_0, g_0, dphi_0, c1, c2, is_run):
    """Implementation of zoom algorithm.
    Algorithm 3.6 from Wright and Nocedal, 'Numerical Optimization', 1999, pg. 59-61.
    Tries cubic, quadratic, and bisection methods of zooming.
    """
    # Constant tensors which avoid loop unrolling
    const_float_one = _to_tensor(1., dtype=a_low.dtype)
    const_bool_false = _to_tensor(False)
    const_int_zero = _to_tensor(0)
    state = {
        "done": const_bool_false,
        "failed": const_bool_false,
        "j": const_int_zero,
        "a_low": a_low,
        "phi_low": phi_low,
        "dphi_low": dphi_low,
        "a_high": a_high,
        "phi_high": phi_high,
        "dphi_high": dphi_high,
        "a_rec": (a_low + a_high) / 2.,
        "phi_rec": (phi_low + phi_high) / 2.,
        "a_star": const_float_one,
        "phi_star": phi_low,
        "dphi_star": dphi_low,
        "g_star": g_0,
        "nfev": const_int_zero,
        "ngev": const_int_zero,
    }

    if mnp.logical_not(is_run):
        return state

    delta1 = 0.2
    delta2 = 0.1
    maxiter = 10  # scipy: 10 jax: 30
    while mnp.logical_not(state["done"]) and state["j"] < maxiter:
        dalpha = state["a_high"] - state["a_low"]
        a = mnp.minimum(state["a_low"], state["a_high"])
        b = mnp.maximum(state["a_low"], state["a_high"])

        cchk = delta1 * dalpha
        qchk = delta2 * dalpha

        a_j_cubic = _cubicmin(state["a_low"], state["phi_low"], state["dphi_low"], state["a_high"],
                              state["phi_high"], state["a_rec"], state["phi_rec"])
        use_cubic = state["j"] > 0 and mnp.isfinite(a_j_cubic) and \
                    mnp.logical_and(a_j_cubic > a + cchk, a_j_cubic < b - cchk)

        a_j_quad = _quadmin(state["a_low"], state["phi_low"], state["dphi_low"], state["a_high"],
                            state["phi_high"])
        use_quad = mnp.logical_not(use_cubic) and mnp.isfinite(a_j_quad) and \
                   mnp.logical_and(a_j_quad > a + qchk, a_j_quad < b - qchk)

        a_j_bisection = (state["a_low"] + state["a_high"]) / 2.0
        use_bisection = mnp.logical_not(use_cubic) and mnp.logical_not(use_quad)

        a_j = mnp.where(use_cubic, a_j_cubic, state["a_rec"])
        a_j = mnp.where(use_quad, a_j_quad, a_j)
        a_j = mnp.where(use_bisection, a_j_bisection, a_j)

        phi_j, g_j, dphi_j = fn(a_j)
        state["nfev"] += 1
        state["ngev"] += 1

        j_to_high = (phi_j > phi_0 + c1 * a_j * dphi_0) or (phi_j >= state["phi_low"])
        state["a_rec"] = mnp.where(j_to_high, state["a_high"], state["a_rec"])
        state["phi_rec"] = mnp.where(j_to_high, state["phi_high"], state["phi_rec"])
        state["a_high"] = mnp.where(j_to_high, a_j, state["a_high"])
        state["phi_high"] = mnp.where(j_to_high, phi_j, state["phi_high"])
        state["dphi_high"] = mnp.where(j_to_high, dphi_j, state["dphi_high"])

        j_to_star = mnp.logical_not(j_to_high) and mnp.abs(dphi_j) <= ops.negative(ops.add(c2, Tensor(0))) * dphi_0
        state["done"] = j_to_star
        state["a_star"] = mnp.where(j_to_star, a_j, state["a_star"])
        state["phi_star"] = mnp.where(j_to_star, phi_j, state["phi_star"])
        state["g_star"] = mnp.where(j_to_star, g_j, state["g_star"])
        state["dphi_star"] = mnp.where(j_to_star, dphi_j, state["dphi_star"])

        low_to_high = mnp.logical_not(j_to_high) and mnp.logical_not(j_to_star) and \
                      dphi_j * (state["a_high"] - state["a_low"]) >= 0.
        state["a_rec"] = mnp.where(low_to_high, state["a_high"], state["a_rec"])
        state["phi_rec"] = mnp.where(low_to_high, state["phi_high"], state["phi_rec"])
        state["a_high"] = mnp.where(low_to_high, a_low, state["a_high"])
        state["phi_high"] = mnp.where(low_to_high, phi_low, state["phi_high"])
        state["dphi_high"] = mnp.where(low_to_high, dphi_low, state["dphi_high"])

        j_to_low = mnp.logical_not(j_to_high) and mnp.logical_not(j_to_star)
        state["a_rec"] = mnp.where(j_to_low, state["a_low"], state["a_rec"])
        state["phi_rec"] = mnp.where(j_to_low, state["phi_low"], state["phi_rec"])
        state["a_low"] = mnp.where(j_to_low, a_j, state["a_low"])
        state["phi_low"] = mnp.where(j_to_low, phi_j, state["phi_low"])
        state["dphi_low"] = mnp.where(j_to_low, dphi_j, state["dphi_low"])

        state["j"] += 1

    state["failed"] = state["j"] == maxiter
    return state


class LineSearch(nn.Cell):
    """Line Search that satisfies strong Wolfe conditions."""

    def __init__(self, func, jac):
        """Initialize LineSearch."""
        super(LineSearch, self).__init__()
        self.func = func
        self.jac = jac

    def construct(self, xk, pk, old_fval=None, old_old_fval=None, gfk=None, c1=1e-4, c2=0.9, maxiter=20):
        def fval_and_grad(alpha):
            xkk = xk + alpha * pk
            fkk = self.func(xkk)
            gkk = self.jac(xkk)
            return fkk, gkk, mnp.dot(gkk, pk)

        # Constant tensors which avoid loop unrolling
        const_float_zero = _to_tensor(0., dtype=xk.dtype)
        const_float_one = _to_tensor(1., dtype=xk.dtype)
        const_bool_false = _to_tensor(False)
        const_int_zero = _to_tensor(0)
        const_int_one = _to_tensor(1)

        if old_fval is None or gfk is None:
            nfev, ngev = const_int_one, const_int_one
            phi_0, g_0, dphi_0 = fval_and_grad(const_float_zero)
        else:
            nfev, ngev = const_int_zero, const_int_zero
            phi_0, g_0 = old_fval, gfk
            dphi_0 = mnp.dot(g_0, pk)

        if old_old_fval is None:
            start_value = const_float_one
        else:
            old_phi0 = old_old_fval
            candidate_start_value = 1.01 * 2 * (phi_0 - old_phi0) / dphi_0
            start_value = mnp.where(
                mnp.isfinite(candidate_start_value),
                mnp.minimum(candidate_start_value, const_float_one),
                const_float_one
            )

        state = {
            "done": const_bool_false,
            "failed": const_bool_false,
            "i": const_int_one,
            "a_i": const_float_zero,
            "phi_i": phi_0,
            "dphi_i": dphi_0,
            "nfev": nfev,
            "ngev": ngev,
            "a_star": const_float_zero,
            "phi_star": phi_0,
            "dphi_star": dphi_0,
            "g_star": g_0,
        }

        while mnp.logical_not(state["done"]) and state["i"] <= maxiter:
            a_i = mnp.where(state["i"] > 1, state["a_i"] * 2.0, start_value)
            phi_i, g_i, dphi_i = fval_and_grad(a_i)
            state["nfev"] += 1
            state["ngev"] += 1

            # Armijo condition
            cond1 = (phi_i > phi_0 + c1 * a_i * dphi_0) or \
                    (phi_i >= state["phi_i"] and state["i"] > 1)
            zoom1 = _zoom(fval_and_grad, state["a_i"], state["phi_i"], state["dphi_i"],
                          a_i, phi_i, dphi_i, phi_0, g_0, dphi_0, c1, c2, cond1)
            state["nfev"] += zoom1["nfev"]
            state["ngev"] += zoom1["ngev"]
            state["done"] = cond1
            state["failed"] = cond1 and zoom1["failed"]
            state["a_star"] = mnp.where(cond1, zoom1["a_star"], state["a_star"])
            state["phi_star"] = mnp.where(cond1, zoom1["phi_star"], state["phi_star"])
            state["g_star"] = mnp.where(cond1, zoom1["g_star"], state["g_star"])
            state["dphi_star"] = mnp.where(cond1, zoom1["dphi_star"], state["dphi_star"])

            # Curvature condition
            cond2 = mnp.logical_not(cond1) and mnp.abs(dphi_i) <= -c2 * dphi_0
            state["done"] = state["done"] or cond2
            state["a_star"] = mnp.where(cond2, a_i, state["a_star"])
            state["phi_star"] = mnp.where(cond2, phi_i, state["phi_star"])
            state["g_star"] = mnp.where(cond2, g_i, state["g_star"])
            state["dphi_star"] = mnp.where(cond2, dphi_i, state["dphi_star"])

            # Satisfying the strong wolf conditions
            cond3 = mnp.logical_not(cond1) and mnp.logical_not(cond2) and dphi_i >= 0.
            zoom2 = _zoom(fval_and_grad, a_i, phi_i, dphi_i, state["a_i"], state["phi_i"],
                          state["dphi_i"], phi_0, g_0, dphi_0, c1, c2, cond3)
            state["nfev"] += zoom2["nfev"]
            state["ngev"] += zoom2["ngev"]
            state["done"] = state["done"] or cond3
            state["failed"] = state["failed"] or (cond3 and zoom2["failed"])
            state["a_star"] = mnp.where(cond3, zoom2["a_star"], state["a_star"])
            state["phi_star"] = mnp.where(cond3, zoom2["phi_star"], state["phi_star"])
            state["g_star"] = mnp.where(cond3, zoom2["g_star"], state["g_star"])
            state["dphi_star"] = mnp.where(cond3, zoom2["dphi_star"], state["dphi_star"])

            state["i"] += 1
            state["a_i"] = a_i
            state["phi_i"] = phi_i
            state["dphi_i"] = dphi_i

        state["status"] = mnp.where(
            state["failed"],
            1,  # zoom failed
            mnp.where(
                state["i"] > maxiter,
                3,  # maxiter reached
                0,  # passed (should be)
            ),
        )
        state["a_star"] = mnp.where(
            _to_tensor(state["a_star"].dtype != mstype.float64)
            and (mnp.abs(state["a_star"]) < 1e-8),
            mnp.sign(state["a_star"]) * 1e-8,
            state["a_star"],
        )
        return state


def line_search(f, xk, pk, jac=None, gfk=None, old_fval=None, old_old_fval=None, c1=1e-4, c2=0.9, maxiter=20):
    """Inexact line search that satisfies strong Wolfe conditions.

    Algorithm 3.5 from Wright and Nocedal, 'Numerical Optimization', 1999, pg. 59-61

    Note:
        `line_search` is not supported on Windows platform yet.

    Args:
        f (function): function of the form f(x) where x is a flat Tensor and returns a real
            scalar. The function should be composed of operations with vjp defined.
        gf (function): the gradient function at x where x is a flat Tensor and returns a Tensor.
            The function can be None if you want to use automatic credits.
        xk (Tensor): initial guess.
        pk (Tensor): direction to search in. Assumes the direction is a descent direction.
        gfk (Tensor): initial value of value_and_gradient as position. Default: ``None`` .
        old_fval (Tensor): The same as `gfk`. Default: ``None`` .
        old_old_fval (Tensor): unused argument, only for scipy API compliance. Default: ``None`` .
        c1 (float): Wolfe criteria constant, see ref. Default: ``1e-4`` .
        c2 (float): The same as `c1`. Default: ``0.9`` .
        maxiter (int): maximum number of iterations to search. Default: ``20`` .

    Returns:
        LineSearchResults, results of line search results.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore.scipy.optimize import line_search
        >>> from mindspore import Tensor
        >>> x0 = Tensor(onp.ones(2).astype(onp.float32))
        >>> p0 = Tensor(onp.array([-1, -1]).astype(onp.float32))
        >>> def func(x):
        ...     return x[0] ** 2 - x[1] ** 3
        >>> res = line_search(func, x0, p0)
        >>> print(res.a_k)
        1.0
    """
    if jac is None:
        jac = grad(f)

    state = LineSearch(f, jac)(xk, pk, old_fval, old_old_fval, gfk, c1, c2, maxiter)
    # If running in graph mode, the state is a tuple.
    if isinstance(state, tuple):
        state = _LineSearchResults(failed=_to_scalar(state[1] or not state[0]),
                                   nit=_to_scalar(state[2] - 1),
                                   nfev=_to_scalar(state[6]),
                                   ngev=_to_scalar(state[7]),
                                   k=_to_scalar(state[2]),
                                   a_k=_to_scalar(state[8]),
                                   f_k=_to_scalar(state[9]),
                                   g_k=state[11],
                                   status=_to_scalar(state[12]))
    else:
        state = _LineSearchResults(failed=_to_scalar(state["failed"] or not state["done"]),
                                   nit=_to_scalar(state["i"] - 1),
                                   nfev=_to_scalar(state["nfev"]),
                                   ngev=_to_scalar(state["ngev"]),
                                   k=_to_scalar(state["i"]),
                                   a_k=_to_scalar(state["a_star"]),
                                   f_k=_to_scalar(state["phi_star"]),
                                   g_k=state["g_star"],
                                   status=_to_scalar(state["status"]))

    return state

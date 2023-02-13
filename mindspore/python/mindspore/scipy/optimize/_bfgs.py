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
"""BFGS"""
from typing import NamedTuple
from ... import nn
from ... import numpy as mnp
from ...common import Tensor
from .line_search import LineSearch
from ..utils import _to_scalar, _to_tensor, grad, _norm


class _BFGSResults(NamedTuple):
    """Results from BFGS optimization.

    Arg
        converged (bool): `True`` if minimization converged.
        failed (bool): `True`` if line search failed.
        k (int): the number of iterations of the BFGS update.
        nfev (int): total number of objective evaluations performed.
        ngev (int): total number of jacobian evaluations
        nhev (int): total number of hessian evaluations
        x_k (Tensor): containing the last argument value found during the search. If
            the search converged, then this value is the argmin of the objective
            function.
        f_k (float): containing the value of the objective function at `x_k`. If the
            search converged, then this is the (local) minimum of the objective
            function.
        g_k (Tensor): containing the gradient of the objective function at `x_k`. If
            the search converged the l2-norm of this Tensor should be below the
            tolerance.
        H_k (Tensor): containing the inverse of the estimated Hessian.
        old_old_fval (float): Function value for the point preceding x=x_k.
        status (int): describing end state.
        line_search_status (int): describing line search end state (only means
            something if line search fails).
    """
    converged: bool
    failed: bool
    k: int
    nfev: int
    ngev: int
    nhev: int
    x_k: Tensor
    f_k: float
    g_k: Tensor
    H_k: Tensor
    old_old_fval: float
    status: int
    line_search_status: int


class MinimizeBfgs(nn.Cell):
    """minimize bfgs"""

    def __init__(self, func, jac):
        """Initialize MinimizeBfgs."""
        super(MinimizeBfgs, self).__init__()
        self.func = func
        self.jac = jac
        self.line_search = LineSearch(func, jac)

    def construct(self, x0, maxiter=None, norm=mnp.inf, gtol=1e-5, line_search_maxiter=10):
        # Constant tensors which avoid loop unrolling
        const_bool_false = _to_tensor(False)
        const_int_zero = _to_tensor(0)
        const_int_one = _to_tensor(1)

        if maxiter is None:
            maxiter = mnp.size(x0) * 200

        d = x0.shape[0]
        identity = mnp.eye(d, dtype=x0.dtype)
        f_0 = self.func(x0)
        g_0 = self.jac(x0)

        state = {
            "converged": _norm(g_0, ord_=mnp.inf) < gtol,
            "failed": const_bool_false,
            "k": const_int_zero,
            "nfev": const_int_one,
            "ngev": const_int_one,
            "nhev": const_int_zero,
            "x_k": x0,
            "f_k": f_0,
            "g_k": g_0,
            "H_k": identity,
            "old_old_fval": f_0 + _norm(g_0) / 2,
            "status": const_int_zero,
            "line_search_status": const_int_zero
        }

        while state["k"] < maxiter:
            p_k = -1 * mnp.dot(state["H_k"], state["g_k"])
            line_search_results = self.line_search(state["x_k"],
                                                   p_k,
                                                   old_fval=state["f_k"],
                                                   old_old_fval=state["old_old_fval"],
                                                   gfk=state["g_k"],
                                                   maxiter=line_search_maxiter)
            state["nfev"] += line_search_results["nfev"]
            state["ngev"] += line_search_results["ngev"]
            state["failed"] = line_search_results["failed"] or mnp.logical_not(line_search_results["done"])
            state["line_search_status"] = line_search_results["status"]

            if state["failed"]:
                break

            s_k = line_search_results["a_star"] * p_k
            x_kp1 = state["x_k"] + s_k
            f_kp1 = line_search_results["phi_star"]
            g_kp1 = line_search_results["g_star"]
            y_k = g_kp1 - state["g_k"]

            state["old_old_fval"] = state["f_k"]
            state["converged"] = _norm(g_kp1, ord_=norm) < gtol
            state["x_k"] = x_kp1
            state["f_k"] = f_kp1
            state["g_k"] = g_kp1

            if state["converged"]:
                break

            rho_k = mnp.reciprocal(mnp.dot(y_k, s_k))
            sy_k = mnp.expand_dims(s_k, axis=1) * mnp.expand_dims(y_k, axis=0)
            term1 = rho_k * sy_k
            ys_k = mnp.expand_dims(y_k, axis=1) * mnp.expand_dims(s_k, axis=0)
            term2 = rho_k * ys_k
            term3 = mnp.matmul(identity - term1, state["H_k"])
            term4 = mnp.matmul(term3, identity - term2)
            term5 = rho_k * (mnp.expand_dims(s_k, axis=1) * mnp.expand_dims(s_k, axis=0))
            hess_kp1 = term4 + term5
            state["H_k"] = hess_kp1

            # next iteration
            state["k"] = state["k"] + 1

        status = mnp.where(
            state["converged"],
            0,  # converged
            mnp.where(
                state["k"] == maxiter,
                1,  # max iters reached
                mnp.where(
                    state["failed"],
                    2 + state["line_search_status"],  # ls failed (+ reason)
                    -1,  # undefined
                )
            )
        )
        state["status"] = status
        return state


def minimize_bfgs(func, x0, jac=None, maxiter=None, norm=mnp.inf, gtol=1e-5, line_search_maxiter=10):
    """Minimize a function using BFGS.

    Implements the BFGS algorithm from
        Algorithm 6.1 from Wright and Nocedal, 'Numerical Optimization', 1999, pg.
        136-143.

    Args:
        fun (Callable): function of the form f(x) where x is a flat Tensor and returns a real
            scalar. The function should be composed of operations with vjp defined.
        x0 (Tensor): initial guess.
        jac (Callable, optional): method for computing the gradient vector.
        if it is None, the gradient will be estimated with gradient of ``func``.
        if it is a callable, it should be a function that returns the gradient vector:
          jac(x, *args) -> array_like, shape (n,)
          where x is an array with shape (n,) and args is a tuple with the fixed parameters.
        maxiter (int, optional): maximum number of iterations.
        norm (float): order of norm for convergence check. Default inf.
        gtol (float): terminates minimization when |grad|_norm < g_tol.
        line_search_maxiter (int): maximum number of linesearch iterations.

    Returns:
        BFGSResults, results from BFGS optimization.

    Supported Platforms:
        ``GPU`` ``CPU``
    """
    if jac is None:
        jac = grad(func)

    if maxiter is None:
        maxiter = mnp.size(x0) * 200

    state = MinimizeBfgs(func, jac)(x0, maxiter, norm, gtol, line_search_maxiter)
    # If running in graph mode, the state is a tuple.
    if isinstance(state, tuple):
        state = _BFGSResults(converged=_to_scalar(state[0]),
                             failed=_to_scalar(state[1]),
                             k=_to_scalar(state[2]),
                             nfev=_to_scalar(state[3]),
                             ngev=_to_scalar(state[4]),
                             nhev=_to_scalar(state[5]),
                             x_k=state[6],
                             f_k=_to_scalar(state[7]),
                             g_k=state[8],
                             H_k=state[9],
                             old_old_fval=_to_scalar(state[10]),
                             status=_to_scalar(state[11]),
                             line_search_status=_to_scalar(state[12]))
    else:
        state = _BFGSResults(converged=_to_scalar(state["converged"]),
                             failed=_to_scalar(state["failed"]),
                             k=_to_scalar(state["k"]),
                             nfev=_to_scalar(state["nfev"]),
                             ngev=_to_scalar(state["ngev"]),
                             nhev=_to_scalar(state["nhev"]),
                             x_k=state["x_k"],
                             f_k=_to_scalar(state["f_k"]),
                             g_k=state["g_k"],
                             H_k=state["H_k"],
                             old_old_fval=_to_scalar(state["old_old_fval"]),
                             status=_to_scalar(state["status"]),
                             line_search_status=_to_scalar(state["line_search_status"]))

    return state

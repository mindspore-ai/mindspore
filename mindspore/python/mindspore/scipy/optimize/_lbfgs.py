# Copyright 2022 Huawei Technologies Co., Ltd
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
"""LBFGS"""
from typing import NamedTuple
from ... import nn
from ... import numpy as mnp
from ...common import Tensor
from .line_search import LineSearch
from ..utils import _to_scalar, _to_tensor, grad, _norm


class _LBFGSResults(NamedTuple):
    """Results from LBFGS optimization.

    Arg
        converged (bool): `True`` if minimization converged.
        failed (bool): `True`` if line search failed.
        k (int): the number of iterations of the LBFGS update.
        nfev (int): total number of objective evaluations performed.
        ngev (int): total number of jacobian evaluations
        x_k (Tensor): containing the last argument value found during the search. If
            the search converged, then this value is the argmin of the objective
            function.
        f_k (float): containing the value of the objective function at `x_k`. If the
            search converged, then this is the (local) minimum of the objective
            function.
        g_k (Tensor): containing the gradient of the objective function at `x_k`. If
            the search converged the l2-norm of this Tensor should be below the
            tolerance.
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
    x_k: Tensor
    f_k: float
    g_k: Tensor
    old_old_fval: float
    status: int
    line_search_status: int


class MinimizeLbfgs(nn.Cell):
    """minimize LBFGS"""

    def __init__(self, func, jac):
        """Initialize MinimizeLbfgs."""
        super(MinimizeLbfgs, self).__init__()
        self.func = func
        self.jac = jac
        self.line_search = LineSearch(func, jac)

    def update_grad_and_curvature(self, g, s_list, y_list, rho_list):
        """calculate alpha using curature info(s, y)
           used to calculate the final search direction vector
           by updating approxicate inv hessian
        """
        alpha_list = []
        for index in range(len(s_list) - 1, -1, -1):
            alpha = rho_list[index] * mnp.dot(s_list[index], g)
            alpha_list.insert(0, alpha)
            g = g - alpha * y_list[index]
        return g, alpha_list

    def update_appr_inv_hessian(self, r, s_list, y_list, rho_list, alpha_list):
        """calculate the final search direction vector
           by updating approxicate inv hessian
        """
        for index in range(len(s_list)):
            beta = rho_list[index] * mnp.dot(y_list[index], r)
            r = r + (alpha_list[index] - beta) * s_list[index]
        return r

    def calc_search_direction_vector(self, s, y, s_list, y_list, rho_list, g, h_diag):
        """calculate the final search direction vector"""
        rho = mnp.reciprocal(mnp.dot(y, s))
        s_list.append(s)
        y_list.append(y)
        rho_list.append(rho)

        q, alpha_list = self.update_grad_and_curvature(g, s_list, y_list, rho_list)

        r = mnp.matmul(h_diag, q)
        r = self.update_appr_inv_hessian(r, s_list, y_list, rho_list, alpha_list)
        return r, s_list, y_list, rho_list

    def construct(self, x0, history_size, maxiter=None, norm=mnp.inf, gtol=1e-5, line_search_maxiter=10):
        # Constant tensors which avoid loop unrolling
        const_bool_false = _to_tensor(False)
        const_int_zero = _to_tensor(0)
        const_int_one = _to_tensor(1)

        if maxiter is None:
            maxiter = mnp.size(x0) * 200

        identity = mnp.eye(x0.shape[0], dtype=x0.dtype)
        f = self.func(x0)
        g = self.jac(x0)
        r = mnp.dot(identity, g)
        # the buffer used to store data to calc approximate hessian
        # and the final direction vector
        s_list = []
        y_list = []
        rho_list = []

        state = {
            "converged": _norm(g, ord_=mnp.inf) < gtol,
            "failed": const_bool_false,
            "k": const_int_zero,
            "nfev": const_int_one,
            "ngev": const_int_one,
            "x_k": x0,
            "f_k": f,
            "g_k": g,
            "old_old_fval": f + _norm(g) / 2,
            "status": const_int_zero,
            "line_search_status": const_int_zero
        }

        while state["k"] < maxiter:
            # search direction vector
            sdv = -1 * r
            line_search_results = self.line_search(state["x_k"],
                                                   sdv,
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

            s = line_search_results["a_star"] * sdv
            x = state["x_k"] + s
            f = line_search_results["phi_star"]
            g = line_search_results["g_star"]
            y = g - state["g_k"]

            state["old_old_fval"] = state["f_k"]
            state["converged"] = _norm(g, ord_=norm) < gtol
            state["x_k"] = x
            state["f_k"] = f
            state["g_k"] = g

            if state["converged"]:
                break

            # counter of iteration
            state["k"] = state["k"] + 1

            r = g
            ys = mnp.dot(y, s)
            if ys <= 1e-10:
                continue

            s_list_tmp = []
            y_list_tmp = []
            rho_list_tmp = []
            if history_size == len(s_list):
                s_list_tmp = s_list[1:]
                y_list_tmp = y_list[1:]
                rho_list_tmp = rho_list[1:]
            else:
                s_list_tmp = s_list
                y_list_tmp = y_list
                rho_list_tmp = rho_list

            gamma = ys * mnp.reciprocal(mnp.dot(y, y))
            h_diag = gamma * identity
            r, s_list, y_list, rho_list = self.calc_search_direction_vector(s,
                                                                            y,
                                                                            s_list_tmp,
                                                                            y_list_tmp,
                                                                            rho_list_tmp,
                                                                            g,
                                                                            h_diag)

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


def minimize_lbfgs(func, x0, jac=None, history_size=20, maxiter=None, norm=mnp.inf, gtol=1e-5, line_search_maxiter=10):
    """Minimize a function using LBFGS.

    Implements the LBFGS algorithm

    Args:
        fun (Callable): function of the form f(x) where x is a flat Tensor and returns a real
            scalar. The function should be composed of operations with vjp defined.
        x0 (Tensor): initial guess.
        jac (Callable, optional): method for computing the gradient vector.
        if it is None, the gradient will be estimated with gradient of func.
        if it is a callable, it should be a function that returns the gradient vector:
          jac(x, *args) -> array_like, shape (n,)
          where x is an array with shape (n,) and args is a tuple with the fixed parameters.
        history_size (int, optional): size of buffer used to help to update inv hessian, Default: 20.
        maxiter (int, optional): maximum number of iterations.
        norm (float): order of norm for convergence check. Default inf.
        gtol (float): terminates minimization when |grad|_norm < g_tol.
        line_search_maxiter (int): maximum number of linesearch iterations.

    Returns:
        LBFGSResults, results from LBFGS optimization.

    Supported Platforms:
        ``GPU`` ``CPU``
    """
    if jac is None:
        jac = grad(func)

    if maxiter is None:
        maxiter = mnp.size(x0) * 200

    state = MinimizeLbfgs(func, jac)(x0, history_size, maxiter, norm, gtol, line_search_maxiter)
    # If running in graph mode, the state is a tuple.
    if isinstance(state, tuple):
        state = _LBFGSResults(converged=_to_scalar(state[0]),
                              failed=_to_scalar(state[1]),
                              k=_to_scalar(state[2]),
                              nfev=_to_scalar(state[3]),
                              ngev=_to_scalar(state[4]),
                              x_k=state[5],
                              f_k=_to_scalar(state[6]),
                              g_k=state[7],
                              old_old_fval=_to_scalar(state[8]),
                              status=_to_scalar(state[9]),
                              line_search_status=_to_scalar(state[10]))
    else:
        state = _LBFGSResults(converged=_to_scalar(state["converged"]),
                              failed=_to_scalar(state["failed"]),
                              k=_to_scalar(state["k"]),
                              nfev=_to_scalar(state["nfev"]),
                              ngev=_to_scalar(state["ngev"]),
                              x_k=state["x_k"],
                              f_k=_to_scalar(state["f_k"]),
                              g_k=state["g_k"],
                              old_old_fval=_to_scalar(state["old_old_fval"]),
                              status=_to_scalar(state["status"]),
                              line_search_status=_to_scalar(state["line_search_status"]))

    return state

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
"""GradientOptimizationAlgorithm"""
from abc import abstractmethod
from typing import NamedTuple
from mindspore import numpy as mnp
from mindspore.common import Tensor
from mindspore.scipy.optimize.line_search import LineSearch
from mindspore.scipy.utils import _norm


class GradientOptimizationResults(NamedTuple):
    """Results from gradient optimization.

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


class GradientOptimizationAlgorithm():
    """Base class of gradient optimization algorithms"""

    def __init__(self, func, jac):
        """Initialize GradientOptimizationAlgorithm."""
        self.func = func
        self.jac = jac
        self.line_search = LineSearch(func, jac)

    @abstractmethod
    def gen_search_vector(self, s_k, y_k, state):
        """Calculate the search vector for the next line search"""

    def construct(self, x_0, maxiter=None, norm=mnp.inf, gtol=1e-5, line_search_maxiter=10):
        """Calculate the minimum value of the func by the gradient optimization algorithm"""
        if maxiter is None:
            maxiter = mnp.size(x_0) * 200

        f_0 = self.func(x_0)
        g_0 = self.jac(x_0)

        state = {
            "converged": _norm(g_0, ord_=mnp.inf) < gtol,
            "failed": False,
            "k": 0,
            "nfev": 1,
            "ngev": 1,
            "nhev": 0,
            "x_k": x_0,
            "f_k": f_0,
            "g_k": g_0,
            "search_vector": g_0,
            "H_k": None,
            "old_old_fval": f_0 + _norm(g_0) / 2,
            "a_star": None,
            "phi_star": None,
            "g_star": None,
            "status": 0,
            "line_search_status": 0
        }

        while state.get("k") < maxiter:
            # search direction vector
            sdv = -1 * state.get("search_vector")
            line_search_results = self.line_search(state.get("x_k"),
                                                   sdv,
                                                   old_fval=state.get("f_k"),
                                                   old_old_fval=state.get("old_old_fval"),
                                                   gfk=state.get("g_k"),
                                                   maxiter=line_search_maxiter)
            if isinstance(line_search_results, tuple):
                state["failed"] = mnp.logical_not(line_search_results[0]) or line_search_results[1]
                state["nfev"] += line_search_results[6]
                state["ngev"] += line_search_results[7]
                state["a_star"] = line_search_results[8]
                state["phi_star"] = line_search_results[9]
                state["g_star"] = line_search_results[11]
                state["line_search_status"] = line_search_results[12]
            else:
                state["failed"] = mnp.logical_not(line_search_results.get("done")) or line_search_results.get("failed")
                state["nfev"] += line_search_results.get("nfev")
                state["ngev"] += line_search_results.get("ngev")
                state["a_star"] = line_search_results.get("a_star")
                state["phi_star"] = line_search_results.get("phi_star")
                state["g_star"] = line_search_results.get("g_star")
                state["line_search_status"] = line_search_results.get("status")

            if state.get("failed"):
                break

            s_k = state.get("a_star") * sdv
            x_k = state.get("x_k") + s_k
            f_k = state.get("phi_star")
            g_k = state.get("g_star")
            y_k = g_k - state.get("g_k")

            state["old_old_fval"] = state.get("f_k")
            state["converged"] = _norm(g_k, ord_=norm) < gtol
            state["x_k"] = x_k
            state["f_k"] = f_k
            state["g_k"] = g_k

            if state.get("converged"):
                break

            self.gen_search_vector(s_k, y_k, state)

            # counter of iteration
            state["k"] = state.get("k") + 1

        status = mnp.where(
            state.get("converged"),
            0,  # converged
            mnp.where(
                state.get("k") == maxiter,
                1,  # max iters reached
                mnp.where(
                    state.get("failed"),
                    2 + state.get("line_search_status"),  # ls failed (+ reason)
                    -1,  # undefined
                )
            )
        )
        state["status"] = status
        return state

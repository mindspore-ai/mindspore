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
from mindspore import ops, Parameter, numpy as mnp
from mindspore.nn import Cell
from mindspore.scipy.utils import grad, _to_scalar
from mindspore.scipy.optimize.gradient_optimization_algorithm import GradientOptimizationAlgorithm
from mindspore.scipy.optimize.gradient_optimization_algorithm import GradientOptimizationResults


class SearchVector(Cell):
    """Calculate the search vector"""
    def __init__(self, shape, dtype):
        """Initialize SearchVector."""
        super(SearchVector, self).__init__()
        # create buffer to save history data to compute new search vector
        zeros = ops.Zeros()
        self._tensor_s = Parameter(zeros(shape, dtype))
        self._tensor_y = Parameter(zeros(shape, dtype))
        self._alpha_rate = Parameter(zeros(shape, dtype))
        self._beta_rate = Parameter(zeros(shape, dtype))

    def construct(self, tensor_q, s_k, y_k):
        """Building a graph to computes the search vector"""
        # If the change of curvature is too small, use the gradient as the search vector directly.
        y_s = self._dot(y_k, s_k)
        if y_s <= 1e-10:
            return tensor_q

        # save data as history, use to compute search vector
        self._tensor_s = self._save_history_data(self._tensor_s, s_k)
        self._tensor_y = self._save_history_data(self._tensor_y, y_k)
        rho = y_s.inv()
        self._alpha_rate = self._save_history_data(self._alpha_rate, rho * s_k)
        self._beta_rate = self._save_history_data(self._beta_rate, rho * y_k)

        alpha_tensor = ops.ZerosLike()(self._alpha_rate)
        history_size = self._tensor_s.shape[0]
        for index in range(history_size - 1, -1, -1):
            alpha = self._dot(self._alpha_rate[index], tensor_q)
            tensor_q = tensor_q - alpha * self._tensor_y[index]
            alpha_tensor[index] = alpha

        gamma = y_s / self._dot(y_k, y_k)
        tensor_r = gamma * tensor_q
        alpha_s = ops.mul(alpha_tensor, self._tensor_s)
        for index in range(history_size):
            beta = self._dot(self._beta_rate[index], tensor_r)
            tensor_r = tensor_r + alpha_s[index] - beta * self._tensor_s[index]
        return tensor_r

    def _save_history_data(self, history_buffer, data):
        """Remove the first history data, and move the other history data forward one position
           put the new data into the last position of the history buffer.
        """
        for index in range(history_buffer.shape[0] - 1):
            history_buffer[index] = history_buffer[index + 1]
        history_buffer[-1] = data
        return history_buffer

    def _dot(self, vector_a, vector_b):
        shape = vector_a.shape + (1,)
        res = ops.MatMul(transpose_a=True)(vector_a.reshape(shape), vector_b.reshape(shape))
        return res.reshape(res.shape[:-1])


class AlgorithmLbfgs(GradientOptimizationAlgorithm):
    """minimize AlgorithmLbfgs"""

    def __init__(self, func, jac, history_size):
        """Initialize AlgorithmLbfgs."""
        super(AlgorithmLbfgs, self).__init__(func, jac)
        self.history_size = history_size
        self.search_vector = None

    def gen_search_vector(self, s_k, y_k, state):
        """Calculate the search vector for the next line search"""
        state["search_vector"] = state.get("g_star")

        if self.search_vector is None:
            shape = (self.history_size,) + s_k.shape
            self.search_vector = SearchVector(shape, s_k.dtype)

        state["search_vector"] = self.search_vector(state.get("search_vector"), s_k, y_k)


def minimize_lbfgs(func, x_0, jac=None, history_size=20, maxiter=None, norm=mnp.inf, gtol=1e-5, line_search_maxiter=10):
    """Minimize a function using LBFGS.

    Implements the LBFGS algorithm

    Args:
        fun (Callable): function of the form f(x) where x is a flat Tensor and returns a real
            scalar. The function should be composed of operations with vjp defined.
        x_0 (Tensor): initial guess.
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
        maxiter = ops.size(x_0) * 200

    state = AlgorithmLbfgs(func, jac, history_size).construct(x_0, maxiter, norm, gtol, line_search_maxiter)
    results = GradientOptimizationResults(converged=_to_scalar(state.get("converged")),
                                          failed=_to_scalar(state.get("failed")),
                                          k=_to_scalar(state.get("k")),
                                          nfev=_to_scalar(state.get("nfev")),
                                          ngev=_to_scalar(state.get("ngev")),
                                          nhev=_to_scalar(state.get("nhev")),
                                          x_k=state.get("x_k"),
                                          f_k=_to_scalar(state.get("f_k")),
                                          g_k=state.get("g_k"),
                                          H_k=state.get("H_k"),
                                          old_old_fval=_to_scalar(state.get("old_old_fval")),
                                          status=_to_scalar(state.get("status")),
                                          line_search_status=_to_scalar(state.get("line_search_status")))

    return results

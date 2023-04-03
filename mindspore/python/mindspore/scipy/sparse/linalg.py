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
"""Sparse linear algebra submodule"""
from ... import nn
from ... import numpy as mnp
from ...ops import functional as F
from ...common import Tensor, CSRTensor
from ...ops.composite.multitype_ops.zeros_like_impl import zeros_like
from ..utils import _to_tensor, _norm, _type_check, _value_check, \
    _sparse_check, _matvec
from ..utils_const import is_within_graph


def _cg(A, b, x0, tol, atol, maxiter, M):
    """
    Figure 2.5 from Barrett R, et al. 'Templates for the sulution of linear systems:
    building blocks for iterative methods', 1994, pg. 12-14
    """
    # Constant tensor which avoids loop unrolling
    const_int_zero = _to_tensor(0)
    atol_ = mnp.maximum(atol, tol * _norm(b))

    r = b - _matvec(A, x0)
    z = p = _matvec(M, r)
    rho = mnp.dot(r, z)
    k = const_int_zero
    x = x0
    while k < maxiter and _norm(r) > atol_:
        q = _matvec(A, p)
        alpha = rho / mnp.dot(p, q)
        x = x + alpha * p
        r = r - alpha * q

        z = _matvec(M, r)
        rho_ = mnp.dot(r, z)
        beta = rho_ / rho
        p = z + beta * p
        rho = rho_

        k += 1

    return x, F.select(_norm(r) > atol_, k, const_int_zero)


class CG(nn.Cell):
    """Use Conjugate Gradient iteration to solve the linear system:

    .. math::
        A x = b
    """

    def __init__(self, A, M):
        super(CG, self).__init__()
        self.A = A
        self.M = M

    def construct(self, b, x0, tol, atol, maxiter):
        return _cg(self.A, b, x0, tol, atol, maxiter, self.M)


class CGv2(nn.Cell):
    """
    This is a new version of CG, which contains all parameters in a graph.
    """

    def __init__(self):
        super(CGv2, self).__init__()

    def construct(self, A, b, x0, tol, atol, maxiter, M):
        return _cg(A, b, x0, tol, atol, maxiter, M)

    def bprop(self, A, b, x0, tol, atol, maxiter, M, out, dout):
        """
        Derivatives of `cg` are implemented via implicit differentiation with
        another `cg` solve, rather than by differentiating *through* the solver.
        They will be accurate only if both solves converge.
        """
        n = b.shape[0]
        if not isinstance(M, (Tensor, CSRTensor)):
            M = F.eye(n, n, b.dtype)
        grad_b, _ = self.construct(A, dout[0], x0, tol, atol, maxiter, M)
        if isinstance(A, CSRTensor):
            grad_a_dense = -1 * F.reshape(grad_b, (n, 1)) * F.reshape(out[0], (1, n))
            values = F.csr_gather(A.indptr, A.indices, grad_a_dense, A.shape)
            grad_a = CSRTensor(A.indptr, A.indices, values, A.shape)
        else:
            grad_a = -1 * F.reshape(grad_b, (n, 1)) * F.reshape(out[0], (1, n))
        return grad_a, grad_b, zeros_like(x0), zeros_like(tol), zeros_like(atol), zeros_like(maxiter), zeros_like(M)


def cg(A, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None, M=None, callback=None):
    """Use Conjugate Gradient iteration to solve the linear system:

    .. math::
        A x = b

    The numerics of MindSpore's `cg` should exact match SciPy's `cg` (up to
    numerical precision).

    Derivatives of `cg` are implemented via implicit differentiation with
    another `cg` solve, rather than by differentiating *through* the solver.
    They will be accurate only if both solves converge.

    Note:
        - Input `A` must represent a hermitian, positive definite matrix. If not,
          the output is wrong and inconsistent with scipy.

        - `cg` is not supported on Windows platform yet.

        - Currently, when `A` is a CSRTensor, the derivatives of `cg` is not supported on PyNative mode.

    Args:
        A (Union[Tensor, CSRTensor, function]): 2D Tensor, CSRTensor or function that calculates the linear
            map (matrix-vector product) :math:`Ax` when called like :math:`A(x)`.
            As function, `A` must return Tensor with the same structure and shape as its input matrix.
        b (Tensor): Right hand side of the linear system representing a single vector. Can be
            stored as a Tensor.
        x0 (Tensor): Starting guess for the solution. Must have the same structure as `b`. Default: None.
        tol (float, optional): Tolerances for convergence, :math:`norm(residual) <= max(tol*norm(b), atol)`.
            We do not implement SciPy's "legacy" behavior, so MindSpore's tolerance will
            differ from SciPy unless you explicitly pass `atol` to SciPy's `cg`. Default: 1e-5.
        atol (float, optional): The same as `tol`. Default: 0.0.
        maxiter (int): Maximum number of iterations.  Iteration will stop after maxiter
            steps even if the specified tolerance has not been achieved. Default: None.
        M (Union[Tensor, CSRTensor, function]): Preconditioner for A.  The preconditioner should approximate the
            inverse of A. Effective preconditioning dramatically improves the
            rate of convergence, which implies that fewer iterations are needed
            to reach a given error tolerance. Default: None.
        callback (function, optional): User-supplied function to call after each iteration.
            It is called as callback(xk), where xk is the current solution vector. Default: None.

    Returns:
        - Tensor, the converged solution. Has the same structure as `b`.
        - Tensor, placeholder for convergence information: 0 : successful exit.
          >0 : convergence to tolerance not achieved, number of iterations. <0 : illegal input or breakdown.

    Raises:
        TypeError: If `tol` is not float.
        TypeError: If `atol` is not float.
        TypeError: If `maxiter` is not int.
        ValueError: If `callback` is not None.
        TypeError: If `A` is not Tensor, CSRTensor, or Function.
        TypeError: If `M` is not None, Tensor, CSRTensor, or Function.
        TypeError: If `b` is not Tensor.
        TypeError: If `x0` is not None or Tensor.
        ValueError: If `b` is not 1 or 2 dimension.
        ValueError: If `x0` and `b` don't have the same structure and type.
        ValueError: If `A` is a square matrix.
        ValueError: If `M` is a square matrix when `M` is not a function.
        TypeError: If `A` and `b` don't have the same data types.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore.common import Tensor
        >>> from mindspore.scipy.sparse.linalg import cg
        >>> A = Tensor(onp.array([[1, 2], [2, 1]], dtype='float32'))
        >>> b = Tensor(onp.array([1, -1], dtype='float32'))
        >>> result, info = cg(A, b)
        >>> print(result)
        [-1.  1.]
        >>> print(info)
        0
    """
    func_name = 'cg'
    A, M, b, x0 = _sparse_check(func_name, A, M, b, x0)
    if maxiter is None:
        maxiter = 10 * b.size  # copied from scipy
    _type_check(func_name, tol, float, 'tol')
    _type_check(func_name, atol, float, 'atol')
    _type_check(func_name, maxiter, int, 'maxiter')
    _value_check(func_name, callback, None, 'callback', op='is', fmt='todo')

    if not is_within_graph(b):
        x, info = CG(A, M)(b, x0, tol, atol, maxiter)
    else:
        x, info = CGv2()(A, b, x0, tol, atol, maxiter, M)
    return x, info

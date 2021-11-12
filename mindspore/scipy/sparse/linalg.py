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
from ...common import Tensor

from ..utils import _INT_ZERO, _normalize_matvec


class CG(nn.Cell):
    """Figure 2.5 from Barrett R, et al. 'Templates for the sulution of linear systems:
    building blocks for iterative methods', 1994, pg. 12-14
    """

    def __init__(self, A, M):
        super(CG, self).__init__()
        self.A = A
        self.M = M

    def construct(self, b, x0, tol, atol, maxiter):
        A = _normalize_matvec(self.A)
        M = _normalize_matvec(self.M)

        def _my_norm(x, ord_=None):
            if ord_ == mnp.inf:
                res = mnp.max(mnp.abs(x))
            else:
                res = mnp.sqrt(mnp.sum(x ** 2))
            return res

        atol_ = mnp.maximum(atol, tol * _my_norm(b))

        r = b - A(x0)
        z = p = M(r)
        rho = mnp.dot(r, z)
        k = _INT_ZERO
        x = x0
        while k < maxiter and _my_norm(r) > atol_:
            q = A(p)
            alpha = rho / mnp.dot(p, q)
            x = x + alpha * p
            r = r - alpha * q

            z = M(r)
            rho_ = mnp.dot(r, z)
            beta = rho_ / rho
            p = z + beta * p
            rho = rho_.copy()

            k += 1

        return x


def cg(A, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None, M=None) -> (Tensor, None):
    """Use Conjugate Gradient iteration to solve ``Ax = b``.

    The numerics of MindSpore's ``cg`` should exact match SciPy's ``cg`` (up to
    numerical precision).

    Derivatives of ``cg`` are implemented via implicit differentiation with
    another ``cg`` solve, rather than by differentiating *through* the solver.
    They will be accurate only if both solves converge.

    Args:
        A (Tensor or function): 2D Tensor or function that calculates the linear
            map (matrix-vector product) ``Ax`` when called like ``A(x)``.
            ``A`` must return Tensor with the same structure and shape as its argument.
        b (Tensor): Right hand side of the linear system representing a single vector. Can be
            stored as a Tensor.

    Returns:
        x (Tensor): The converged solution. Has the same structure as ``b``.
        info (None): Placeholder for convergence information. In the future, MindSpore will report
            the number of iterations when convergence is not achieved, like SciPy.

    Other Parameters:
        x0 (Tensor): Starting guess for the solution. Must have the same structure as ``b``.
        tol, atol (float, optional): Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
            We do not implement SciPy's "legacy" behavior, so MindSpore's tolerance will
            differ from SciPy unless you explicitly pass ``atol`` to SciPy's ``cg``.
        maxiter (int): Maximum number of iterations.  Iteration will stop after maxiter
            steps even if the specified tolerance has not been achieved.
        M (Tensor): Preconditioner for A.  The preconditioner should approximate the
            inverse of A.  Effective preconditioning dramatically improves the
            rate of convergence, which implies that fewer iterations are needed
            to reach a given error tolerance.

    Supported Platforms:
        ``CPU`` ``GPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore.common import Tensor
        >>> from mindspore.scipy.sparse.linalg import cg
        >>> A = Tensor(onp.array([[1, 2], [2, 1]], dtype='float32'))
        >>> b = Tensor(onp.array([1, -1], dtype='float32'))
        >>> cg(A, b)
        [-1.  1.]
    """
    if x0 is None:
        x0 = mnp.zeros_like(b)

    if maxiter is None:
        maxiter = 10 * b.shape[0]

    if M is None:
        def identity(x):
            return x

        M = identity

    if x0.shape != b.shape:
        raise ValueError(
            'Tensor in x0 and b must have matching shapes: ', x0.shape, " vs ", b.shape)

    x = CG(A, M)(b, x0, tol, atol, maxiter)
    return x, None

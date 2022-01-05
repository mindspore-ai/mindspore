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
from ... import nn, ms_function
from ... import numpy as mnp
from ...ops import functional as F
from ..linalg import solve_triangular
from ..linalg import cho_factor, cho_solve
from ..utils import _INT_ZERO, _INT_NEG_ONE, _normalize_matvec, _to_tensor, _safe_normalize, _eps
from ..utils_const import _raise_value_error


def gram_schmidt(Q, q):
    """
    do Gram–Schmidt process to normalize vector v
    """
    h = mnp.dot(Q.T, q)
    Qh = mnp.dot(Q, h)
    q = q - Qh
    return q, h


def arnoldi_iteration(k, A, M, V, H):
    """ Performs a single (the k'th) step of the Arnoldi process."""
    v_ = V[..., k]
    v = M(A(v_))
    v, h = gram_schmidt(V, v)
    _, v_norm_0 = _safe_normalize(v)
    tol = _eps(v) * v_norm_0
    unit_v, v_norm_1 = _safe_normalize(v, tol)
    V[..., k + 1] = unit_v
    h[k + 1] = v_norm_1
    H[k, :] = h
    breakdown = v_norm_1 == 0
    return V, H, breakdown


@ms_function
def rotate_vectors(H, i, cs, sn):
    x1 = H[i]
    y1 = H[i + 1]
    x2 = cs * x1 - sn * y1
    y2 = sn * x1 + cs * y1
    H[i] = x2
    H[i + 1] = y2
    return H


class BatchedGmres(nn.Cell):
    """
    Implements a single restart of GMRES. The ``restart``-dimensional Krylov subspace
    This implementation solves a dense linear problem instead of building
    a QR factorization during the Arnoldi process.
    """

    def __init__(self, A, M):
        super(BatchedGmres, self).__init__()
        self.A = A
        self.M = M

    def construct(self, b, x0=None, tol=1e-5, atol=0.0, restart=20, maxiter=None):
        A = _normalize_matvec(self.A)
        M = _normalize_matvec(self.M)
        dtype = b.dtype
        _, b_norm = _safe_normalize(b)
        atol = mnp.maximum(tol * b_norm, _to_tensor(atol), dtype=dtype)
        residual = M(b - A(x0))
        unit_residual, residual_norm = _safe_normalize(residual)
        k = _INT_ZERO
        x = x0
        while k < maxiter and residual_norm > atol:
            pad_width = ((0, 0),) * unit_residual.ndim + ((0, restart),)
            V = mnp.pad(unit_residual[..., None], pad_width=pad_width)
            H = mnp.eye(restart, restart + 1, dtype=dtype)
            k_iter = _INT_ZERO
            breakdown = _to_tensor(False)
            while k_iter < restart and mnp.logical_not(breakdown):
                V, H, breakdown = arnoldi_iteration(k_iter, A, M, V, H)
                k_iter += 1
            beta_vec = mnp.zeros((restart + 1,), dtype=dtype)
            beta_vec[0] = residual_norm
            a2 = mnp.dot(H, H.T)
            b2 = mnp.dot(H, beta_vec)
            c, lower = cho_factor(a2, lower=False)
            factor = (c, lower)
            y = cho_solve(factor, b2)
            dx = mnp.dot(V[..., :-1], y)
            x = x + dx
            residual = b - A(x)
            unit_residual, residual_norm = _safe_normalize(residual)
            k += 1
        return x


class IterativeGmres(nn.Cell):
    """
    Implements a iterative GMRES. While building the ``restart``-dimensional
    Krylov subspace iteratively using Givens Rotation method, the algorithm
    constructs a Triangular matrix R which could be more easily solved.
    """

    def __init__(self, A, M):
        super(IterativeGmres, self).__init__()
        self.A = A
        self.M = M

    def construct(self, b, x0, tol, atol, restart, maxiter):
        A = _normalize_matvec(self.A)
        M = _normalize_matvec(self.M)

        _, b_norm = _safe_normalize(b)
        atol = mnp.maximum(tol * b_norm, atol)

        Mb = M(b)
        _, Mb_norm = _safe_normalize(Mb)
        ptol = Mb_norm * mnp.minimum(1.0, atol / b_norm)

        r = M(b - A(x0))
        r, r_norm = _safe_normalize(r)

        iters = _INT_ZERO
        while iters < maxiter and r_norm > atol:
            V = mnp.pad(r[..., None], ((0, 0),) * r.ndim + ((0, restart),))
            dtype = mnp.result_type(b)
            # use eye() to avoid constructing a singular matrix in case of early
            # termination
            R = mnp.eye(restart, restart + 1, dtype=dtype)
            givens = mnp.zeros((restart, 2), dtype=dtype)
            beta_vec = mnp.zeros((restart + 1), dtype=dtype)
            beta_vec[0] = r_norm

            k = _INT_ZERO
            err = r_norm
            while mnp.logical_and(mnp.less(k, restart), mnp.less(ptol, err)):
                V, R, _ = arnoldi_iteration(k, A, M, V, R)
                # givens rotation
                row_k = R[k, :].copy()
                i = _INT_ZERO
                while i < k:
                    row_k = rotate_vectors(row_k, i, givens[i, 0], givens[i, 1])
                    i += 1

                if row_k[k + 1] == 0:
                    givens[k, 0] = 1
                    givens[k, 1] = 0
                else:
                    increase = mnp.absolute(row_k[k]) < mnp.absolute(row_k[k + 1])
                    t = mnp.where(increase, -row_k[k] / row_k[k + 1], -row_k[k + 1] / row_k[k])
                    r = 1 / F.sqrt(1 + mnp.absolute(t) ** 2)
                    givens[k, 0] = mnp.where(increase, r * t, r)
                    givens[k, 1] = mnp.where(increase, r, r * t)

                R[k, :] = rotate_vectors(row_k, k, givens[k, 0], givens[k, 1])
                beta_vec = rotate_vectors(beta_vec, k, givens[k, 0], givens[k, 1])
                err = mnp.absolute(beta_vec[k + 1])
                k += 1

            y = solve_triangular(R[:, :-1], beta_vec[:-1], trans='T', lower=True)
            dx = mnp.dot(V[:, :-1], y)

            x = x0 + dx
            r = M(b - A(x))
            r, r_norm = _safe_normalize(r)
            x0 = x
            iters += 1

        return x0


def gmres(A, b, x0=None, *, tol=1e-5, atol=0.0, restart=20, maxiter=None,
          M=None, solve_method='batched'):
    """
    GMRES solves the linear system :math:`A x = b` for x, given A and b.

    A is specified as a function performing A(vi) -> vf = A @ vi, and in principle
    need not have any particular special properties, such as symmetry. However,
    convergence is often slow for nearly symmetric operators.

    Note:
        In the future, MindSpore will report the number of iterations when convergence
        is not achieved, like SciPy. Currently it is None, as a Placeholder.

    Args:
        A (Union[Tensor, function]): 2D Tensor or function that calculates the linear
            map (matrix-vector product) :math:`Ax` when called like :math:`A(x)`.
            As function, `A` must return Tensor with the same structure and shape as its input matrix.
        b (Tensor): Right hand side of the linear system representing a single vector.
            Can be stored as a Tensor.
        x0 (Tensor, optional): Starting guess for the solution. Must have the same structure
            as `b`. If this is unspecified, zeroes are used. Default: None.
        tol (float, optional): Tolerances for convergence,
            :math:`norm(residual) <= max(tol*norm(b), atol)`. We do not implement SciPy's
            "legacy" behavior, so MindSpore's tolerance will differ from SciPy unless you
            explicitly pass `atol` to SciPy's `gmres`. Default: 1e-5.
        atol (float, optional): The same as `tol`. Default: 0.0.
        restart (integer, optional): Size of the Krylov subspace ("number of iterations")
            built between restarts. GMRES works by approximating the true solution x as its
            projection into a Krylov space of this dimension - this parameter
            therefore bounds the maximum accuracy achievable from any guess
            solution. Larger values increase both number of iterations and iteration
            cost, but may be necessary for convergence. The algorithm terminates
            early if convergence is achieved before the full subspace is built.
            Default: 20.
        maxiter (int): Maximum number of times to rebuild the size-`restart`
            Krylov space starting from the solution found at the last iteration. If GMRES
            halts or is very slow, decreasing this parameter may help.
            Default: None.
        M (Union[Tensor, function]): Preconditioner for A.  The preconditioner should approximate the
            inverse of A.  Effective preconditioning dramatically improves the
            rate of convergence, which implies that fewer iterations are needed
            to reach a given error tolerance. Default: None.
        solve_method (str): There are two kinds of solve methods,'incremental' or 'batched'. Default: "batched".

            - incremental: builds a QR decomposition for the Krylov subspace incrementally during
              the GMRES process using Givens rotations. This improves numerical stability and gives
              a free estimate of the residual norm that allows for early termination within a single "restart".
            - batched: solve the least squares problem from scratch at the end of each GMRES
              iteration. It does not allow for early termination, but has much less overhead on GPUs.

    Returns:
        - Tensor, the converged solution. Has the same structure as `b`.
        - None, placeholder for convergence information.

    Supported Platforms:
        ``CPU`` ``GPU``

    Examples:
        >>> import numpy as onp
        >>> import mindspore.numpy as mnp
        >>> from mindspore.common import Tensor
        >>> from mindspore.scipy.sparse.linalg import gmres
        >>> A = Tensor(mnp.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=mnp.float32))
        >>> b = Tensor(mnp.array([2, 4, -1], dtype=mnp.float32))
        >>> x, exitCode = gmres(A, b)
        >>> exitCode            # 0 indicates successful convergence
        0
        >>> onp.allclose(mnp.dot(A,x).asnumpy(), b.asnumpy())
        True
    """

    if x0 is None:
        x0 = mnp.zeros_like(b)
    size = b.size
    if maxiter is None:
        maxiter = 10 * size  # copied from scipy
    if restart > size:
        restart = size

    if M is None:
        M = lambda x: x

    if solve_method == 'incremental':
        x = IterativeGmres(A, M)(b, x0, tol, atol, restart, maxiter)
    elif solve_method == 'batched':
        x = BatchedGmres(A, M)(b, x0, tol, atol, restart, maxiter)
    else:
        _raise_value_error("solve_method should be in ('incremental' or 'batched'), but got {}."
                           .format(solve_method))
    _, x_norm = _safe_normalize(x)
    info = mnp.where(mnp.isnan(x_norm), _INT_NEG_ONE, _INT_ZERO)
    return x, info


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
            rho = rho_

            k += 1

        return x


def cg(A, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None, M=None):
    """Use Conjugate Gradient iteration to solve :math:`Ax = b`.

    The numerics of MindSpore's `cg` should exact match SciPy's `cg` (up to
    numerical precision).

    Derivatives of `cg` are implemented via implicit differentiation with
    another `cg` solve, rather than by differentiating *through* the solver.
    They will be accurate only if both solves converge.

    Note:
        In the future, MindSpore will report the number of iterations when convergence
        is not achieved, like SciPy. Currently it is None, as a Placeholder.

    Args:
        A (Union[Tensor, function]): 2D Tensor or function that calculates the linear
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
        M (Union[Tensor, function]): Preconditioner for A.  The preconditioner should approximate the
            inverse of A. Effective preconditioning dramatically improves the
            rate of convergence, which implies that fewer iterations are needed
            to reach a given error tolerance. Default: None.

    Returns:
        - Tensor, the converged solution. Has the same structure as `b`.
        - None, placeholder for convergence information.

    Supported Platforms:
        ``CPU`` ``GPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore.common import Tensor
        >>> from mindspore.scipy.sparse.linalg import cg
        >>> A = Tensor(onp.array([[1, 2], [2, 1]], dtype='float32'))
        >>> b = Tensor(onp.array([1, -1], dtype='float32'))
        >>> result, _ = cg(A, b)
        >>> result
        Tensor(shape=[2], dtype=Float32, value= [-1.00000000e+00,  1.00000000e+00])
    """
    if x0 is None:
        x0 = mnp.zeros_like(b)

    if maxiter is None:
        maxiter = 10 * b.shape[0]

    if M is None:
        M = lambda x: x

    if x0.shape != b.shape:
        _raise_value_error(
            'Tensor in x0 and b must have matching shapes: {} vs {}'.format(x0.shape, b.shape))

    x = CG(A, M)(b, x0, tol, atol, maxiter)
    return x, None

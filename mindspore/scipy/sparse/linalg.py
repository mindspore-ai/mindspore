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
from ... import nn, Tensor, ops, ms_function
from ... import numpy as mnp
from ...ops import functional as F
from ..linalg import solve_triangular

from ..utils import _INT_ZERO, _normalize_matvec, _INT_ONE, _safe_normalize, _SafeNormalize


class ArnoldiIteration(nn.Cell):
    """ do the Arnoldi iteration"""

    def __init__(self):
        super(ArnoldiIteration, self).__init__()
        self.safe_normalize = _SafeNormalize()
        self.eps = ops.Eps()
        self.sqrt_2 = F.pows(Tensor(2.0), 1/2.0)
        self.matmul_t = ops.MatMul(True, False)

    def construct(self, k, V, v, H):
        v, h = self._gram_schmidt(V, v)

        eps_v = self.eps(v[(0,) * v.ndim])
        _, v_norm_0 = self.safe_normalize(v)
        tol = eps_v * v_norm_0
        unit_v, v_norm_1 = self.safe_normalize(v, tol)
        V[..., k + 1] = unit_v

        h[k + 1] = v_norm_1
        H[k, :] = h
        return V, H

    def _gram_schmidt(self, Q, q):
        """
        do Gram–Schmidt process to normalize vector v
        """
        # transpose is not support float64 yet,
        # so the following code is the same as h = mnp.dot(Q.T, q)
        h = self.matmul_t(Q, q.reshape((q.shape[0], 1))).flatten()
        Qh = mnp.dot(Q, h)
        q = q - Qh

        return q, h


@ms_function
def rotate_vectors(H, i, cs, sn):
    x1 = H[i]
    y1 = H[i + 1]
    x2 = cs * x1 - sn * y1
    y2 = sn * x1 + cs * y1
    H[i] = x2
    H[i + 1] = y2
    return H


class GivensRotation(nn.Cell):
    """ do the Givens Rotation"""

    def __init__(self):
        super(GivensRotation, self).__init__()
        self.tensor_0 = Tensor(0.0)
        self.tensor_1 = Tensor(1.0)

    def construct(self, H_row, givens, k):
        """
        Appliy each of the Givens rotations stored in givens[:, :k] to H_row.

        Args:
            H_row (Tensor): The kth row in (n, n+1) Matrix H
            givens (Tensor): a (n, 2) Matrix which stores cs, sn for Givens Rotation
            k (Tensor): the row number, must smaller than n

        Returns:
            R_row (Tensor): Rotated Vector from H_row
            givens (Tensor): a (n, 2) Matrix which stores the kth cs, sn values
        """
        i = _INT_ZERO

        while i < k:
            H_row = rotate_vectors(H_row, i, givens[i, 0], givens[i, 1])
            i = i + _INT_ONE

        if H_row[k + 1] == self.tensor_0:
            givens[k, 0] = self.tensor_1
            givens[k, 1] = self.tensor_0
        else:
            increase = mnp.absolute(H_row[k]) < mnp.absolute(H_row[k + 1])
            t = mnp.where(increase, -H_row[k] /
                          H_row[k + 1], -H_row[k + 1] / H_row[k])
            r = 1 / F.sqrt(1 + mnp.absolute(t) ** 2)
            givens[k, 0] = mnp.where(increase, r * t, r)
            givens[k, 1] = mnp.where(increase, r, r * t)

        R_row = rotate_vectors(H_row, k, givens[k, 0], givens[k, 1])
        return R_row, givens


kth_arnoldi_iteration = ArnoldiIteration()
givens_rotation = GivensRotation()


def gmres_iter(A_mat_func, b, x0, r, r_norm, ptol, restart, M_mat_func):
    """
    Single iteration for Gmres Algorithm with restart
    """
    V = mnp.pad(r[..., None], ((0, 0),) * r.ndim + ((0, restart),))
    dtype = mnp.result_type(b)
    # use eye() to avoid constructing a singular matrix in case of early
    # termination
    R = mnp.eye(restart, restart + 1, dtype=dtype)
    givens = mnp.zeros((restart, 2), dtype=dtype)
    beta_vec = mnp.zeros((restart + 1), dtype=dtype)
    beta_vec[0] = r_norm

    k = 0
    err = r_norm
    while mnp.logical_and(mnp.less(k, restart), mnp.less(ptol, err)):
        v = M_mat_func(A_mat_func(V[:, k]))
        V, H = kth_arnoldi_iteration(k, V, v, R)
        R[k, :], givens = givens_rotation(H[k, :], givens, k)
        beta_vec = rotate_vectors(beta_vec, k, givens[k, 0], givens[k, 1])
        err = mnp.absolute(beta_vec[k + 1])
        k = k + 1

    y = solve_triangular(R[:, :-1], beta_vec[:-1], trans='T', lower=True)
    dx = mnp.dot(V[:, :-1], y)

    x = x0 + dx
    r = M_mat_func(b - A_mat_func(x))
    r, r_norm = _safe_normalize(r)
    return x, r, r_norm


def gmres(A, b, x0=None, *, tol=1e-5, atol=0.0, restart=20, maxiter=None,
          M=None, solve_method='batched') -> (Tensor, int):
    """
    GMRES solves the linear system A x = b for x, given A and b.

    A is specified as a function performing A(vi) -> vf = A @ vi, and in principle
    need not have any particular special properties, such as symmetry. However,
    convergence is often slow for nearly symmetric operators.

    Args:
        A (Tensor or function): 2D Tensor or function that calculates the linear
            map (matrix-vector product) ``Ax`` when called like ``A(x)``.
            ``A`` must return Tensor with the same structure and shape as its argument.
        b (Tensor): Right hand side of the linear system representing a single vector.
            Can be stored as a Tensor

    Returns:
        x (Tensor): The converged solution. Has the same structure as ``b``.
        info (None): Placeholder for convergence information. In the future, MindSpore
            will report the number of iterations when convergence is not achieved, like SciPy.

    Other Parameters:
        x0 (Tensor, optional): Starting guess for the solution. Must have the same structure
            as ``b``. If this is unspecified, zeroes are used.
        tol, atol (float, optional): Tolerances for convergence,
            ``norm(residual) <= max(tol*norm(b), atol)``. We do not implement SciPy's
            "legacy" behavior, so MindSpore's tolerance will differ from SciPy unless you
            explicitly pass ``atol`` to SciPy's ``gmres``.
        restart (integer, optional): Size of the Krylov subspace ("number of iterations")
            built between restarts. GMRES works by approximating the true solution x as its
            projection into a Krylov space of this dimension - this parameter
            therefore bounds the maximum accuracy achievable from any guess
            solution. Larger values increase both number of iterations and iteration
            cost, but may be necessary for convergence. The algorithm terminates
            early if convergence is achieved before the full subspace is built.
            Default is 20.
        maxiter (integer): Maximum number of times to rebuild the size-``restart``
            Krylov space starting from the solution found at the last iteration. If GMRES
            halts or is very slow, decreasing this parameter may help.
            Default is infinite.
        M (Tensor): Preconditioner for A.  The preconditioner should approximate the
            inverse of A.  Effective preconditioning dramatically improves the
            rate of convergence, which implies that fewer iterations are needed
            to reach a given error tolerance.
        solve_method ('incremental' or 'batched'): The 'incremental' solve method
            builds a QR decomposition for the Krylov subspace incrementally during
            the GMRES process using Givens rotations.
            This improves numerical stability and gives a free estimate of the
            residual norm that allows for early termination within a single "restart".
            In contrast, the 'batched' solve method solves the least squares problem
            from scratch at the end of each GMRES iteration. It does not allow for
            early termination, but has much less overhead on GPUs.

    Supported Platforms:
        ``CPU`` ``GPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore.common import Tensor
        >>> from mindspore.numpy as mnp
        >>> from mindspore.scipy.sparse import csc_matrix
        >>> from mindspore.scipy.sparse.linalg import gmres
        >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=np.float32)
        >>> b = Tensor(onp.array([2, 4, -1], dtype=np.float32))
        >>> x, exitCode = gmres(A, b)
        >>> print(exitCode)            # 0 indicates successful convergence
        0
        >>> np.allclose(A.matvec(x).asnumpy(), b.asnumpy())
        True
    """

    if x0 is None:
        x0 = mnp.zeros_like(b)

    if M is None:
        def M_mat_func(x):
            return x
    elif not callable(M):
        def M_mat_func(x):
            return mnp.dot(M, x)
    else:
        M_mat_func = M

    if not callable(A):
        def A_mat_func(x):
            return mnp.dot(A, x)
    else:
        A_mat_func = A

    size = b.size

    if maxiter is None:
        maxiter = 10 * size  # copied from scipy
    restart = min(restart, size)

    _, b_norm = _safe_normalize(b)
    atol = mnp.maximum(tol * b_norm, atol)

    Mb = M_mat_func(b)
    _, Mb_norm = _safe_normalize(Mb)
    ptol = Mb_norm * mnp.minimum(1.0, atol / b_norm)

    if solve_method == 'incremental':
        # iterative gmres
        r = M_mat_func(b - A_mat_func(x0))
        r, r_norm = _safe_normalize(r)
        x = x0
        k = 0
        while k < maxiter and r_norm > atol:
            x, r, r_norm = gmres_iter(
                A_mat_func, b, x, r, r_norm, ptol, restart, M_mat_func)
            k += 1

        _, x_norm = _safe_normalize(x)
        info = mnp.where(mnp.isnan(x_norm), -1, 0)
    elif solve_method == 'batched':
        raise NotImplementedError("batched method not implemented yet")
    else:
        raise ValueError("solve_method should be in ('incremental' or 'batched'), but got {}."
                         .format(solve_method))

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

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
from ...ops import operations as P
from ...common import Tensor, CSRTensor, COOTensor, dtype as mstype
from ...ops.composite.multitype_ops.zeros_like_impl import zeros_like
from ..linalg import solve_triangular
from ..linalg import cho_factor, cho_solve
from ..utils import _to_tensor, _safe_normalize, _eps, _norm, _type_check, _value_check, \
    _sparse_check, _matvec
from ..utils_const import _raise_value_error, _raise_type_error, is_within_graph


def gram_schmidt(Q, q):
    """Do Gram–Schmidt process to normalize vector v"""
    h = mnp.dot(Q.T, q)
    Qh = mnp.dot(Q, h)
    q = q - Qh
    return q, h


def arnoldi_iteration(k, A, M, V, H):
    """Performs a single (the k'th) step of the Arnoldi process."""
    v_ = V[..., k]
    v = _matvec(M, _matvec(A, v_))
    v, h = gram_schmidt(V, v)
    _, v_norm_0 = _safe_normalize(v)
    tol = _eps(v) * v_norm_0
    unit_v, v_norm_1 = _safe_normalize(v, tol)
    V[..., k + 1] = unit_v
    h[k + 1] = v_norm_1
    H[k, :] = h
    breakdown = v_norm_1 == 0
    return V, H, breakdown


def rotate_vectors(H, i, cs, sn):
    """Rotate vectors."""
    x1 = H[i]
    y1 = H[i + 1]
    x2 = cs * x1 - sn * y1
    y2 = sn * x1 + cs * y1
    H[i] = x2
    H[i + 1] = y2
    return H


def _high_precision_cho_solve(a, b, data_type=mstype.float64):
    """As a core computing module of gmres, cholesky solver must explicitly cast to double precision."""
    a = a.astype(mstype.float64)
    b = b.astype(mstype.float64)
    a_a = mnp.dot(a, a.T)
    a_b = mnp.dot(a, b)
    c, lower = cho_factor(a_a, lower=False)
    factor = (c, lower)
    y = cho_solve(factor, a_b)
    return y.astype(data_type)


def _batch_gmres(A, b, x0, tol, restart, maxiter, M, atol):
    """
    batched gmres: solve the least squares problem from scratch at the end of each GMRES iteration.
    It does not allow for early termination, but has much less overhead on GPUs.
    """
    # Constant tensor which avoids loop unrolling
    const_int_zero = _to_tensor(0)
    dtype = b.dtype
    _, b_norm = _safe_normalize(b)
    atol = mnp.maximum(tol * b_norm, _to_tensor(atol), dtype=dtype)
    residual = _matvec(M, b - _matvec(A, x0))
    unit_residual, residual_norm = _safe_normalize(residual)
    k = const_int_zero
    x = x0
    while k < maxiter and residual_norm > atol:
        pad_width = ((0, 0),) * unit_residual.ndim + ((0, restart),)
        V = mnp.pad(unit_residual[..., None], pad_width=pad_width)
        H = mnp.eye(restart, restart + 1, dtype=dtype)
        k_iter = const_int_zero
        breakdown = _to_tensor(False)
        while k_iter < restart and mnp.logical_not(breakdown):
            V, H, breakdown = arnoldi_iteration(k_iter, A, M, V, H)
            k_iter += 1
        beta_vec = mnp.zeros((restart + 1,), dtype=dtype)
        beta_vec[0] = residual_norm
        y = _high_precision_cho_solve(H, beta_vec, data_type=dtype)
        dx = mnp.dot(V[..., :-1], y)
        x = x + dx
        residual = _matvec(M, b - _matvec(A, x))
        unit_residual, residual_norm = _safe_normalize(residual)
        k += 1
    return x, F.select(residual_norm > atol, k, const_int_zero)


def _incremental_gmres(A, b, x0, tol, restart, maxiter, M, atol):
    """
    incremental gmres: builds a QR decomposition for the Krylov subspace incrementally during
    the GMRES process using Givens rotations. This improves numerical stability and gives a free estimate of
    the residual norm that allows for early termination within a single "restart".
    """
    const_int_zero = _to_tensor(0)
    _, b_norm = _safe_normalize(b)
    atol = mnp.maximum(tol * b_norm, atol)

    Mb = _matvec(M, b)
    _, Mb_norm = _safe_normalize(Mb)
    ptol = Mb_norm * mnp.minimum(1.0, atol / b_norm)

    r = _matvec(M, b - _matvec(A, x0))
    r, r_norm = _safe_normalize(r)

    iters = const_int_zero
    while iters < maxiter and r_norm > atol:
        V = mnp.pad(r[..., None], ((0, 0),) * r.ndim + ((0, restart),))
        dtype = mnp.result_type(b)
        # Use eye() to avoid constructing a singular matrix in case of early
        # Termination
        R = mnp.eye(restart, restart + 1, dtype=dtype)
        givens = mnp.zeros((restart, 2), dtype=dtype)
        beta_vec = mnp.zeros((restart + 1), dtype=dtype)
        beta_vec[0] = r_norm

        k = const_int_zero
        err = r_norm
        while mnp.logical_and(mnp.less(k, restart), mnp.less(ptol, err)):
            V, R, _ = arnoldi_iteration(k, A, M, V, R)
            # Givens rotation
            row_k = R[k, :]
            i = const_int_zero
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
        r = _matvec(M, b - _matvec(A, x))
        r, r_norm = _safe_normalize(r)
        x0 = x
        iters += 1
    return x0, F.select(r_norm > atol, iters, const_int_zero)


class GMRES(nn.Cell):
    """
    Given given A and b, GMRES solves the linear system:

    .. math::
        A x = b
    """

    def __init__(self, A, M, solve_method):
        super(GMRES, self).__init__()
        self.A = A
        self.M = M
        self.solve_method = solve_method

    def construct(self, b, x0, tol, restart, maxiter, atol):
        # Constant tensor which avoids loop unrolling
        x = x0
        info = _to_tensor(0)
        if self.solve_method == 'batched':
            x, info = _batch_gmres(self.A, b, x0, tol, restart, maxiter, self.M, atol)
        elif self.solve_method == "incremental":
            x, info = _incremental_gmres(self.A, b, x0, tol, restart, maxiter, self.M, atol)
        else:
            _raise_value_error("solve_method should be in ('incremental' or 'batched'), but got ", self.solve_method,
                               ".")
        return x, info


class GMRESV2(nn.Cell):
    """
    This is a new version of GMRES, which contains all parameters in a graph.
    """

    def __init__(self, solve_method):
        super(GMRESV2, self).__init__()
        self.solve_method = solve_method

    def transpose(self, a):
        if isinstance(a, CSRTensor):
            a_coo = a.to_coo()
            row_indices = a_coo.indices[:, 0]
            col_indices = a_coo.indices[:, 1]
            coo_indices = P.Stack(1)([col_indices, row_indices])
            a_t_coo = COOTensor(coo_indices, a_coo.values, a_coo.shape)
            a_t_csr = a_t_coo.to_csr()
            return a_t_csr
        return a.T

    def construct(self, A, b, x0, tol, restart, maxiter, M, atol):
        x = x0
        info = _to_tensor(0)
        if self.solve_method == 'batched':
            x, info = _batch_gmres(A, b, x0, tol, restart, maxiter, M, atol)
        elif self.solve_method == "incremental":
            x, info = _incremental_gmres(A, b, x0, tol, restart, maxiter, M, atol)
        else:
            _raise_value_error("solve_method should be in ('incremental' or 'batched'), but got ", self.solve_method,
                               ".")
        return x, info

    def bprop(self, A, b, x0, tol, restart, maxiter, M, atol, out, dout):
        """
        Derivatives of `gmres` are implemented via implicit differentiation with
        another `gmres` solve, rather than by differentiating *through* the solver.
        They will be accurate only if both solves converge.
        """
        n = b.shape[0]
        if not isinstance(M, (Tensor, CSRTensor)):
            M = F.eye(n, n, b.dtype)
        A_T = self.transpose(A)
        grad_b, _ = self.construct(A_T, dout[0], x0, tol, restart, maxiter, M, atol)
        if isinstance(A, CSRTensor):
            grad_a_dense = -1 * F.reshape(grad_b, (n, 1)) * F.reshape(out[0], (1, n))
            values = F.csr_gather(A.indptr, A.indices, grad_a_dense, A.shape)
            grad_a = CSRTensor(A.indptr, A.indices, values, A.shape)
        else:
            grad_a = -1 * F.reshape(grad_b, (n, 1)) * F.reshape(out[0], (1, n))
        return grad_a, grad_b, zeros_like(x0), zeros_like(tol), zeros_like(atol), zeros_like(maxiter), zeros_like(M)


def gmres(A, b, x0=None, *, tol=1e-5, restart=20, maxiter=None,
          M=None, callback=None, restrt=None, atol=0.0, callback_type=None, solve_method='batched'):
    """
    Given given A and b, GMRES solves the linear system:

    .. math::
        A x = b

    A is specified as a function performing A(vi) -> vf = A @ vi, and in principle
    need not have any particular special properties, such as symmetry. However,
    convergence is often slow for nearly symmetric operators.

    Note:
        - `gmres` is not supported on Windows platform yet.

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
        restart (integer, optional): Size of the Krylov subspace ("number of iterations")
            built between restarts. GMRES works by approximating the true solution x as its
            projection into a Krylov space of this dimension - this parameter
            therefore bounds the maximum accuracy achievable from any guess
            solution. Larger values increase both number of iterations and iteration
            cost, but may be necessary for convergence. The algorithm terminates
            early if convergence is achieved before the full subspace is built. Default: 20.
        maxiter (int): Maximum number of times to rebuild the size-`restart`
            Krylov space starting from the solution found at the last iteration. If GMRES
            halts or is very slow, decreasing this parameter may help. Default: None.
        M (Union[Tensor, function]): Preconditioner for A.  The preconditioner should approximate the
            inverse of A.  Effective preconditioning dramatically improves the
            rate of convergence, which implies that fewer iterations are needed
            to reach a given error tolerance. Default: None.
        callback (function): User-supplied function to call after each iteration. It is called as callback(args),
            where args are selected by callback_type. Default: None.
        restrt (int, optional): Deprecated, use restart instead. Default: None.
        atol (float, optional): The same as `tol`. Default: 0.0.
        callback_type (str, optional): Callback function argument requested:
            Default: None.

            - x: current iterate (ndarray), called on every restart
            - pr_norm: relative (preconditioned) residual norm (float), called on every inner iteration
            - legacy (default): same as pr_norm, but also changes the meaning of ‘maxiter’ to count inner
              iterations instead of restart cycles.

        solve_method (str): There are two kinds of solve methods,'incremental' or 'batched'. Default: "batched".

            - incremental: builds a QR decomposition for the Krylov subspace incrementally during
              the GMRES process using Givens rotations. This improves numerical stability and gives
              a free estimate of the residual norm that allows for early termination within a single "restart".
            - batched: solve the least squares problem from scratch at the end of each GMRES
              iteration. It does not allow for early termination, but has much less overhead on GPUs.

    Returns:
        - Tensor, the converged solution. Has the same structure as `b`.
        - Tensor, placeholder for convergence information: 0 : successful exit.
          >0 : convergence to tolerance not achieved, number of iterations. <0 : illegal input or breakdown.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import numpy as onp
        >>> import mindspore.numpy as mnp
        >>> from mindspore.common import Tensor
        >>> from mindspore.scipy.sparse.linalg import gmres
        >>> A = Tensor(mnp.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=mnp.float32))
        >>> b = Tensor(mnp.array([2, 4, -1], dtype=mnp.float32))
        >>> x, exitCode = gmres(A, b)
        >>> print(exitCode)            # 0 indicates successful convergence
        0
        >>> print(onp.allclose(mnp.dot(A,x).asnumpy(), b.asnumpy()))
        True
    """
    func_name = "gmres"
    A, M, b, x0 = _sparse_check(func_name, A, M, b, x0)
    size = b.size
    if maxiter is None:
        maxiter = 10 * size  # copied from scipy
    _type_check(func_name, tol, float, 'tol')
    _type_check(func_name, restart, int, 'restart')
    _type_check(func_name, maxiter, int, 'maxiter')
    _type_check(func_name, solve_method, str, 'solve_method')
    _value_check(func_name, callback, None, 'callback', op='is', fmt='todo')
    _value_check(func_name, restrt, None, 'restrt', op='is', fmt='todo')
    _value_check(func_name, callback_type, None, 'callback_type', op='is', fmt='todo')
    if restart > size:
        restart = size
    if not is_within_graph(b):
        x, info = GMRES(A, M, solve_method)(b, x0, tol, restart, maxiter, atol)
    else:
        x, info = GMRESV2(solve_method)(A, b, x0, tol, restart, maxiter, M, atol)
    return x, info


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


class BiCGStab(nn.Cell):
    """Figure 2.10 from Barrett R, et al. 'Templates for the sulution of linear systems:
    building blocks for iterative methods', 1994, pg. 24-25
    """

    def __init__(self, A, M):
        super(BiCGStab, self).__init__()
        self.A = A
        self.M = M

    def construct(self, b, x0, tol, atol, maxiter):
        # Constant tensors which avoid loop unrolling
        const_int_zero = _to_tensor(0)
        const_int_neg_one = _to_tensor(-1)

        const_float_one = _to_tensor(1., dtype=b.dtype)
        atol_ = mnp.maximum(atol, tol * _norm(b))

        r = r_tilde = v = p = b - _matvec(self.A, x0)
        rho = alpha = omega = const_float_one
        k = const_int_zero
        x = x0
        while k < maxiter:
            rho_ = mnp.dot(r_tilde, r)
            if rho_ == 0. or omega == 0.:
                k = const_int_neg_one
                break

            beta = rho_ / rho * (alpha / omega)
            p = r + beta * (p - omega * v)
            p_hat = _matvec(self.M, p)
            v = _matvec(self.A, p_hat)
            alpha = rho_ / mnp.dot(r_tilde, v)
            s = r - alpha * v
            x = x + alpha * p_hat
            if _norm(s) <= atol_:
                break

            s_hat = _matvec(self.M, s)
            t = _matvec(self.A, s_hat)
            omega = mnp.dot(t, s) / mnp.dot(t, t)
            x = x + omega * s_hat
            r = s - omega * t
            if _norm(r) <= atol_:
                break

            rho = rho_
            k += 1

        return x, F.select(k == const_int_neg_one or k >= maxiter, k, const_int_zero)


def bicgstab(A, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None, M=None):
    """Use Bi-Conjugate Gradient Stable iteration to solve :math:`Ax = b`.

    The numerics of MindSpore's `bicgstab` should exact match SciPy's
    `bicgstab` (up to numerical precision).

    As with `cg`, derivatives of `bicgstab` are implemented via implicit
    differentiation with another `bicgstab` solve, rather than by
    differentiating *through* the solver. They will be accurate only if
    both solves converge.

    Note:
        - `bicgstab` is not supported on Windows platform yet.

    Args:
        A (Union[Tensor, function]): 2D Tensor or function that calculates the linear
            map (matrix-vector product) :math:`Ax` when called like :math:`A(x)`.
            As function, `A` must return Tensor with the same structure and shape as its input matrix.
        b (Tensor): Right hand side of the linear system representing a single vector. Can be
            stored as a Tensor.
        x0 (Tensor): Starting guess for the solution. Must have the same structure as `b`. Default: None.
        tol (float, optional): Tolerances for convergence, :math:`norm(residual) <= max(tol*norm(b), atol)`.
            We do not implement SciPy's "legacy" behavior, so MindSpore's tolerance will
            differ from SciPy unless you explicitly pass `atol` to SciPy's `bicgstab`. Default: 1e-5.
        atol (float, optional): The same as `tol`. Default: 0.0.
        maxiter (int): Maximum number of iterations.  Iteration will stop after maxiter
            steps even if the specified tolerance has not been achieved. Default: None.
        M (Union[Tensor, function]): Preconditioner for A.  The preconditioner should approximate the
            inverse of A. Effective preconditioning dramatically improves the
            rate of convergence, which implies that fewer iterations are needed
            to reach a given error tolerance. Default: None.

    Returns:
        - Tensor, the converged solution. Has the same structure as `b`.
        - Tensor, placeholder for convergence information: 0 : successful exit.
          >0 : convergence to tolerance not achieved, number of iterations. <0 : illegal input or breakdown.

    Raises:
        ValueError: If `x0` and `b` don't have the same structure.
        TypeError: If `A`, `x0` and `b` don't have the same float types(`mstype.float32` or `mstype.float64`).

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore.common import Tensor
        >>> from mindspore.scipy.sparse.linalg import bicgstab
        >>> A = Tensor(onp.array([[1, 2], [2, 1]], dtype='float32'))
        >>> b = Tensor(onp.array([1, -1], dtype='float32'))
        >>> result, info = bicgstab(A, b)
        >>> print(result)
        [-1.  1.]
        >>> print(info)
        0
    """
    if x0 is None:
        x0 = mnp.zeros_like(b)

    if maxiter is None:
        maxiter = 10 * b.shape[0]

    if M is None:
        M = lambda x: x

    if x0.shape != b.shape:
        _raise_value_error(
            'Input x0 and b must have matching shapes: ', x0.shape, ' vs ', b.shape)

    if (F.dtype(b) not in (mstype.float32, mstype.float64)) or (F.dtype(b) != F.dtype(x0)) or (
            F.dtype(b) != F.dtype(A)):
        _raise_type_error('Input A, x0 and b must have same float types')

    x, info = BiCGStab(A, M)(b, x0, tol, atol, maxiter)
    return x, info

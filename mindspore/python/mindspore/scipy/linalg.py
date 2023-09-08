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
"""Linear algebra submodule"""
from __future__ import absolute_import
from .ops import LU
from .ops import SolveTriangular
from .utils import _nd_transpose, _value_check, _type_check, _dtype_check, _mstype_check, _square_check, _solve_check
from .utils_const import _raise_value_error
from .. import numpy as mnp
from .. import ops
from ..common import dtype as mstype
from ..ops.operations.math_ops import Cholesky
from ..ops.operations.linalg_ops import Eigh
from ..ops import functional as F
from ..ops import operations as P

__all__ = ['block_diag', 'inv', 'cho_factor', 'cholesky', 'cho_solve', 'eigh', 'lu_factor', 'lu']


def block_diag(*arrs):
    """
    Create a block diagonal matrix from provided arrays.

    Given the list of Tensors `A`, `B`, and `C`, the output will have these
    Tensors arranged on the diagonal:

    .. code-block::

        [[A, 0, 0],
         [0, B, 0],
         [0, 0, C]]

    Note:
        `block_diag` is not supported on Windows platform yet.

    Args:
        arrs (list): up to 2-D Input Tensors.
            A 1-D Tensor or a 2-D Tensor with shape :math:`(1,n)`.

    Returns:
        Tensor with `A`, `B`, `C`, ... on the diagonal which has the same dtype as `A`.

    Raises:
        ValueError: If there are Tensors with dimensions higher than 2 in all arguments.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore import Tensor
        >>> from mindspore.scipy.linalg import block_diag
        >>> A = Tensor(onp.array([[1, 0], [0, 1]]))
        >>> B = Tensor(onp.array([[3, 4, 5], [6, 7, 8]]))
        >>> C = Tensor(onp.array([[7]]))
        >>> P = Tensor(onp.zeros((2, ), dtype='int32'))
        >>> print(block_diag(A, B, C))
        [[1 0 0 0 0 0]
         [0 1 0 0 0 0]
         [0 0 3 4 5 0]
         [0 0 6 7 8 0]
         [0 0 0 0 0 7]]
        >>> print(block_diag(A, P, B, C))
        [[1 0 0 0 0 0 0 0]
         [0 1 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0]
         [0 0 0 0 3 4 5 0]
         [0 0 0 0 6 7 8 0]
         [0 0 0 0 0 0 0 7]]
    """
    if not arrs:
        return mnp.zeros((1, 0))
    bad_shapes = [i for i, a in enumerate(arrs) if a.ndim > 2]
    if bad_shapes:
        _raise_value_error("Arguments to mindspore.scipy.linalg.block_diag must have at most 2 dimensions.")

    accum = mnp.atleast_2d(arrs[0])
    for arr in arrs[1:]:
        arr = mnp.atleast_2d(arr)
        _, c = arr.shape
        arr = ops.Pad(((0, 0), (accum.shape[-1], 0)))(arr)
        accum = ops.Pad(((0, 0), (0, c)))(accum)
        accum = mnp.concatenate([accum, arr], axis=0)
    return accum


def inv(a, overwrite_a=False, check_finite=True):
    """
    Compute the inverse of a matrix.

    Note:
        - `inv` is not supported on Windows platform yet.
        - Only `float32`, `float64`, `int32`, `int64` are supported Tensor dtypes. If Tensor with dtype `int32` or
          `int64` is passed, it will be cast to :class:`mstype.float64`.

    Args:
        a (Tensor): Square matrix to be inverted.
        overwrite_a (bool, optional): Discard data in `a` (may improve performance). Default: ``False`` .
        check_finite (bool, optional): Whether to check that the input matrix contains only finite numbers.
            Disabling may give a performance gain, but may result in problems (crashes, non-termination)
            if the inputs do contain infinities or NaNs. Default: ``True`` .

    Returns:
        Tensor, inverse of the matrix `a`.

    Raises:
        LinAlgError: If :math:`a` is singular.
        ValueError: If :math:`a` is not square, or not 2D.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore import Tensor
        >>> import mindspore.numpy as mnp
        >>> from mindspore.scipy.linalg import inv
        >>> a = Tensor(onp.array([[1., 2.], [3., 4.]]))
        >>> print(inv(a))
        [[-2.   1. ]
         [ 1.5 -0.5]]
        >>> print(mnp.dot(a, inv(a)))
        [[1.0000000e+00 0.0000000e+00]
         [8.8817842e-16 1.0000000e+00]]
    """
    func_name = "inv"
    _type_check(func_name, overwrite_a, bool, 'overwrite_a')
    _type_check(func_name, check_finite, bool, 'check_finite')
    _mstype_check(func_name, a, mstype.TensorType)
    _square_check(func_name, a)
    _dtype_check(func_name, a, [mstype.int32, mstype.int64, mstype.float32, mstype.float64])

    if F.dtype(a) in (mstype.int32, mstype.int64):
        a = F.cast(a, mstype.float64)
    matrix_inverse = P.MatrixInverse(adjoint=False)
    return matrix_inverse(a)


def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    """
    Compute the cholesky decomposition of a matrix, to use in :func:`mindspore.scipy.linalg.cho_solve`.

    Returns a matrix containing the cholesky decomposition,
    :math:`a = l l*` or :math:`a = u* u` of a Hermitian positive-definite matrix `a`.
    The return value can be directly used as the first parameter to :func:`mindspore.scipy.linalg.cho_solve`.

    Note:
        - `cho_factor` is not supported on Windows platform yet.
        - Only `float32`, `float64`, `int32`, `int64` are supported Tensor dtypes. If Tensor with dtype `int32` or
          `int64` is passed, it will be cast to :class:`mstype.float64`.

    .. warning::
        The returned matrix also contains random data in the entries not
        used by the cholesky decomposition. If you need to zero these
        entries, use the function `cholesky` instead.

    Args:
        a (Tensor): square Matrix of (M, M) to be decomposed.
        lower (bool, optional): Whether to compute the upper or lower triangular cholesky factorization.
            Default: ``False`` .
        overwrite_a(bool, optional): Whether to overwrite data in a (may improve performance). Default: ``False`` .
            in mindspore, this arg does not work right now.
        check_finite(bool, optional): Whether to check that the input matrix contains only finite numbers.
            Disabling may give a performance gain, but may result in problems
            (crashes, non-termination) if the inputs do contain infinities or NaNs. Default: ``True`` .
            in mindspore, this arg does not work right now.

    Returns:
         - Tensor, matrix whose upper or lower triangle contains the cholesky factor of `a`.
           Other parts of the matrix contain random data.
         - bool, flag indicating whether the factor is in the lower or upper triangle

    Raises:
        ValueError: If input a tensor is not a square matrix or it's dims not equal to 2D.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore import Tensor
        >>> from mindspore.scipy.linalg import cho_factor
        >>> a = Tensor(onp.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]]).astype(onp.float32))
        >>> c, low = cho_factor(a)
        >>> print(c)
        [[ 3.          1.          0.33333334  1.6666666 ]
         [ 3.          2.4494898   1.9051585  -0.2721655 ]
         [ 1.          5.          2.2933078   0.8559526 ]
         [ 5.          1.          2.          1.5541857 ]]
    """
    func_name = "cho_factor"
    _type_check(func_name, overwrite_a, bool, 'overwrite_a')
    _type_check(func_name, check_finite, bool, 'check_finite')
    _type_check(func_name, lower, bool, 'lower')
    _mstype_check(func_name, a, mstype.TensorType)
    _dtype_check(func_name, a, [mstype.int32, mstype.int64, mstype.float32, mstype.float64])
    _square_check(func_name, a)

    if F.dtype(a) in (mstype.int32, mstype.int64):
        a = F.cast(a, mstype.float64)
    cholesky_net = Cholesky()
    c = cholesky_net(a)
    if not lower:
        c = _nd_transpose(c)
    return c, lower


def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
    """
    Compute the cholesky decomposition of a matrix.

    Returns the cholesky decomposition, :math:`a = l l^*` or
    :math:`a = u^* u` of a Hermitian positive-definite matrix a.

    Note:
        - `cholesky` is not supported on Windows platform yet.
        - Only `float32`, `float64`, `int32`, `int64` are supported Tensor dtypes. If Tensor with dtype `int32` or
          `int64` is passed, it will be cast to :class:`mstype.float64`.

    Args:
        a (Tensor): square Matrix of (M, M) to be decomposed.
        lower (bool, optional): Whether to compute the upper- or lower-triangular cholesky
            factorization. Default: ``False`` .
        overwrite_a (bool, optional): Whether to overwrite data in `a` (may improve performance). Default: ``False`` .
            in mindspore, this arg does not work right now.
        check_finite (bool, optional): Whether to check that the input matrix contains only finite numbers.
            Disabling may give a performance gain, but may result in problems
            (crashes, non-termination) if the inputs do contain infinities or NaNs. Default: ``True`` .
            in mindspore, this arg does not work right now.

    Returns:
        Tensor, upper- or lower-triangular cholesky factor of `a`.

    Raises:
        ValueError: If input a tensor is not a square matrix or it's dims not equal to 2D.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore import Tensor
        >>> from mindspore.scipy.linalg import cholesky
        >>> a = Tensor(onp.array([[1, 2],[2, 5]]).astype(onp.float32))
        >>> L = cholesky(a, lower=True)
        >>> print(L)
        [[1. 0.]
         [2. 1.]]
    """
    func_name = "cholesky"
    _type_check(func_name, overwrite_a, bool, 'overwrite_a')
    _type_check(func_name, check_finite, bool, 'check_finite')
    _type_check(func_name, lower, bool, 'lower')
    _mstype_check(func_name, a, mstype.TensorType)
    _dtype_check(func_name, a, [mstype.int32, mstype.int64, mstype.float32, mstype.float64])
    _square_check(func_name, a)

    if F.dtype(a) in (mstype.int32, mstype.int64):
        a = F.cast(a, mstype.float64)
    cholesky_net = Cholesky()
    c = cholesky_net(a)
    if not lower:
        c = _nd_transpose(c)
    return c


def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    """
    Given the cholesky factorization of a, solve the linear equation

    .. math::
        a x = b

    Note:
        - `cho_solve` is not supported on Windows platform yet.
        - Only `float32`, `float64`, `int32`, `int64` support Tensor dtypes. If Tensor with dtype `int32` or
          `int64` is passed, it will be cast to :class:`mstype.float64`.

    Args:
        c_and_lower ((Tensor, bool)): cholesky factorization of :math:`a`,
            as given by :func:`mindspore.scipy.linalg.cho_factor`.
        b (Tensor): Right-hand side.
        overwrite_b (bool, optional): Whether to overwrite data in :math:`b` (may improve performance).
            Default: ``False``.
        check_finite (bool, optional): Whether to check that the input matrices contain only finite numbers.
            Disabling may give a performance gain, but may result in problems
            (crashes, non-termination) if the inputs do contain infinities or NaNs. Default: ``True``.

    Returns:
        Tensor, the solution to the system a x = b.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import numpy as onp
        >>> import mindspore as ms
        >>> a = ms.Tensor(onp.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]]).astype(onp.float32))
        >>> b = ms.Tensor(onp.array([1, 1, 1, 1]).astype(onp.float32))
        >>> c, low = ms.scipy.linalg.cho_factor(a)
        >>> x = ms.scipy.linalg.cho_solve((c, low), b)
        >>> print(x)
        [-0.01749266  0.11953348  0.01166185  0.15743434]
    """
    func_name = "cho_solve"
    (c, lower) = c_and_lower
    _type_check(func_name, overwrite_b, bool, 'overwrite_b')
    _type_check(func_name, check_finite, bool, 'check_finite')
    _type_check(func_name, lower, bool, 'lower')
    _mstype_check(func_name, c, mstype.TensorType, 'c')
    _mstype_check(func_name, b, mstype.TensorType, 'b')
    _dtype_check(func_name, c, [mstype.int32, mstype.int64, mstype.float32, mstype.float64], 'c')
    _dtype_check(func_name, b, [mstype.int32, mstype.int64, mstype.float32, mstype.float64], 'b')
    _solve_check(func_name, c, b, 'c', 'b')

    if F.dtype(c) in (mstype.int32, mstype.int64):
        c = F.cast(c, mstype.float64)
        b = F.cast(b, mstype.float64)
    # Do not support complex, so trans is chosen from ('T', 'N')
    if lower:
        l_trans = 'N'
        l_t_trans = 'T'
    else:
        l_trans = 'T'
        l_t_trans = 'N'
    b = SolveTriangular(lower=lower, unit_diagonal=False, trans=l_trans)(c, b)
    b = SolveTriangular(lower=lower, unit_diagonal=False, trans=l_t_trans)(c, b)
    return b


def eigh(a, b=None, lower=True, eigvals_only=False, overwrite_a=False,
         overwrite_b=False, turbo=True, eigvals=None, type=1, check_finite=True):  # pylint: disable=W0622
    """
    Solve a standard or generalized eigenvalue problem for a complex Hermitian or real symmetric matrix.

    Find eigenvalues Tensor `w` and optionally eigenvectors Tensor `v` of Tensor `a`,
    where `b` is positive definite such that for every eigenvalue `λ` (i-th entry of w) and
    its eigenvector `vi` (i-th column of `v`) satisfies::

                      a @ vi = λ * b @ vi
        vi.conj().T @ a @ vi = λ
        vi.conj().T @ b @ vi = 1

    In the standard problem, `b` is assumed to be the identity matrix.

    Note:
        - `eigh` is not supported on Windows platform yet.
        - Only `float32`, `float64`, `int32`, `int64` are supported Tensor dtypes. If Tensor with dtype `int32` or
          `int64` is passed, it will be cast to `mstype.float64`.

    Args:
        a (Tensor): A :math:`(M, M)` complex Hermitian or real symmetric matrix whose eigenvalues and
            eigenvectors will be computed.
        b (Tensor, optional): A :math:`(M, M)` complex Hermitian or real symmetric definite positive matrix in.
            If omitted, identity matrix is assumed. Default: ``None``.
        lower (bool, optional): Whether the pertinent Tensor data is taken from the lower or upper
            triangle of `a` and, if applicable, `b`. Default: ``True``.
        eigvals_only (bool, optional): Whether to calculate only eigenvalues and no eigenvectors.
            Default: ``False`` .
        type (int, optional): For the generalized problems, this keyword specifies the problem type
            to be solved for `w` and `v` (only takes 1, 2, 3 as possible inputs)::

                1 =>     a @ v = w @ b @ v
                2 => a @ b @ v = w @ v
                3 => b @ a @ v = w @ v

            This keyword is ignored for standard problems. Default: ``1`` .
        overwrite_a (bool, optional): Whether to overwrite data in `a` (may improve performance). Default: ``False`` .
        overwrite_b (bool, optional): Whether to overwrite data in `b` (may improve performance). Default: ``False`` .
        check_finite (bool, optional): Whether to check that the input matrices contain only finite numbers.
            Disabling may give a performance gain, but may result in problems (crashes, non-termination)
            if the inputs do contain infinities or NaNs. Default: ``True`` .
        turbo (bool, optional): use divide and conquer algorithm (faster but expensive in memory, only
            for generalized eigenvalue problem and if full set of eigenvalues are requested.).
            Has no significant effect if eigenvectors are not requested. Default: ``True`` .
        eigvals (tuple, optional): Indexes of the smallest and largest (in ascending order) eigenvalues
            and corresponding eigenvectors to be returned: :math:`0 <= lo <= hi <= M-1`. If omitted, all eigenvalues
            and eigenvectors are returned. Default: ``None`` .

    Returns:
        - Tensor with shape :math:`(N,)`, the :math:`N (1<=N<=M)` selected eigenvalues, in ascending order,
          each repeated according to its multiplicity.

        - Tensor with shape :math:`(M, N)`, if `eigvals_only == False`.

    Raises:
        RuntimeError: If eigenvalue computation does not converge, an error occurred, or b matrix is not
            definite positive. Note that if input matrices are not symmetric or Hermitian, no error will
            be reported but results will be wrong.
        TypeError: If `a` is not Tensor.
        TypeError: If `lower` is not bool.
        TypeError: If `eigvals_only` is not bool.
        TypeError: If `overwrite_a` is not bool.
        TypeError: If `overwrite_b` is not bool.
        TypeError: If `turbo` is not bool.
        TypeError: If `check_finite` is not bool.
        ValueError: If `a` is not square matrix.
        ValueError: If `b` is not None.
        ValueError: If `eigvals` is not None.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import numpy as onp
        >>> import mindspore.numpy as mnp
        >>> from mindspore import Tensor, dtype
        >>> from mindspore.scipy.linalg import eigh
        >>> a = Tensor([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]], dtype.float64)
        >>> w, v = eigh(a)
        >>> print(onp.allclose(mnp.dot(a, v).asnumpy(), mnp.dot(v, mnp.diag(w)).asnumpy(), 1e-5, 1e-8))
        True
    """
    func_name = 'eigh'
    eigh_type_check = F.partial(_type_check, func_name)
    eigh_value_check = F.partial(_value_check, func_name)

    eigh_type_check(lower, bool, 'lower')
    eigh_type_check(eigvals_only, bool, 'eigvals_only')
    eigh_type_check(overwrite_a, bool, 'overwrite_a')
    eigh_type_check(overwrite_b, bool, 'overwrite_b')
    eigh_type_check(turbo, bool, 'turbo')
    eigh_type_check(type, int, 'type')
    eigh_type_check(check_finite, bool, 'check_finite')
    _mstype_check(func_name, a, mstype.TensorType)
    _dtype_check(func_name, a,
                 [mstype.int32, mstype.int64, mstype.float32, mstype.float64, mstype.complex64, mstype.complex128])
    _square_check(func_name, a)
    eigh_value_check(b, None, 'b', op='is', fmt='todo')
    eigh_value_check(eigvals, None, 'eigvals', op='is', fmt='todo')

    if F.dtype(a) in (mstype.int32, mstype.int64):
        a = F.cast(a, mstype.float64)
    eigh_net = Eigh(not eigvals_only, lower=lower)
    return eigh_net(a)


def lu_pivots_to_permutation(pivots, permutation_size: int):
    """transfer pivots to permutation"""
    batch_dims = pivots.shape[:-1]
    k = pivots.shape[-1]
    per = mnp.arange(0, permutation_size)
    permutation = mnp.broadcast_to(per, batch_dims + (permutation_size,))
    permutation = mnp.array(permutation)
    if permutation_size == 0:
        return permutation
    for i in range(k):
        j = pivots[..., i]
        loc = mnp.ix_(*(mnp.arange(0, b) for b in batch_dims))
        x = permutation[..., i]
        y = permutation[loc + (j,)]
        permutation[loc + (j,)] = x
        permutation[..., i] = y
    return permutation


def lu_factor(a, overwrite_a=False, check_finite=True):
    """
    Compute pivoted LU decomposition of a square matrix,
    and its outputs can be directly used as the inputs of `lu_solve`.
    The decomposition is:

    .. math::
        a = p l u

    where :math:`p` is a permutation matrix, :math:`l` lower triangular with unit diagonal elements,
    and :math:`u` upper triangular.

    Note:
        - `lu_factor` is not supported on Windows platform yet.
        - Only `float32`, `float64`, `int32`, `int64` are supported Tensor dtypes. If Tensor with dtype `int32` or
          `int64` is passed, it will be cast to :class:`mstype.float64`.

    Args:
        a (Tensor): square matrix of :math:`(M, M)` to decompose. Note that if the input tensor is not a `float`,
            then it will be cast to :class:'mstype.float32'.
        overwrite_a (bool, optional): Whether to overwrite data in :math:`a` (may increase performance).
            Default: ``False`` .
        check_finite (bool, optional): Whether to check that the input matrix contains only finite numbers.
            Disabling may give a performance gain, but may result in problems
            (crashes, non-termination) if the inputs do contain infinities or NaNs. Default: ``True`` .

    Returns:
        - Tensor, a square matrix of :math:`(N, N)` containing `U` in its upper triangle, and `L` in its lower triangle.
          The unit diagonal elements of `L` are not stored.

        - Tensor, :math:`(N,)` pivot indices representing the permutation matrix `P`:
          the i-th element value j in the indices indicates that row i of matrix was interchanged with row j.

    Raises:
        ValueError: If :math:`a` is not square.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore import Tensor
        >>> from mindspore.scipy.linalg import lu_factor
        >>> a = Tensor(onp.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]]).astype(onp.float64))
        >>> lu, piv = lu_factor(a)
        >>> print(lu)
        [[ 7.          5.          6.          6.        ]
         [ 0.28571429  3.57142857  6.28571429  5.28571429]
         [ 0.71428571  0.12       -1.04        3.08      ]
         [ 0.71428571 -0.44       -0.46153846  7.46153846]]
        >>> print(piv)
        [2 2 3 3]
    """
    func_name = "lu_factor"
    _type_check(func_name, overwrite_a, bool, 'overwrite_a')
    _type_check(func_name, check_finite, bool, 'check_finite')
    _mstype_check(func_name, a, mstype.TensorType)
    _dtype_check(func_name, a, [mstype.int32, mstype.int64, mstype.float32, mstype.float64])
    _square_check(func_name, a)

    if F.dtype(a) in (mstype.int32, mstype.int64):
        a = F.cast(a, mstype.float64)
    msp_lu = LU()
    m_lu, pivots, _ = msp_lu(a)
    return m_lu, pivots


def lu(a, permute_l=False, overwrite_a=False, check_finite=True):
    """
    Compute pivoted LU decomposition of a general matrix.

    The decomposition is:

    .. math::
        a = p l u

    where :math:`P` is a permutation matrix, :math:`L` lower triangular with unit
    diagonal elements, and :math:`U` upper triangular.

    Note:
        - `lu` is not supported on Windows platform yet.
        - Only `float32`, `float64`, `int32`, `int64` are supported Tensor dtypes. If Tensor with dtype `int32` or
          `int64` is passed, it will be cast to :class:`mstype.float64`.

    Args:
        a (Tensor): a :math:`(M, N)` matrix to decompose. Note that if the input tensor is not a `float`,
            then it will be cast to :class:'mstype.float32'.
        permute_l (bool, optional): Perform the multiplication :math:`P L` (Default: do not permute).
            Default: ``False`` .
        overwrite_a (bool, optional): Whether to overwrite data in :math:`a` (may improve performance).
            Default: ``False`` .
        check_finite (bool, optional):  Whether to check that the input matrix contains
            only finite numbers. Disabling may give a performance gain, but may result
            in problems (crashes, non-termination) if the inputs do contain infinities or NaNs. Default: ``True`` .

    Returns:
        **If permute_l == False**

        - Tensor, :math:`(M, M)` permutation matrix.
        - Tensor, :math:`(M, K)` lower triangular or trapezoidal matrix with unit diagonal. :math:`K = min(M, N)`.
        - Tensor, :math:`(K, N)` upper triangular or trapezoidal matrix.

        **If permute_l == True**

        - Tensor, :math:`(M, K)` permuted L matrix. :math:`K = min(M, N)`.
        - Tensor, :math:`(K, N)` upper triangular or trapezoidal matrix.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore import Tensor
        >>> from mindspore.scipy.linalg import lu
        >>> a = Tensor(onp.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]]).astype(onp.float64))
        >>> p, l, u = lu(a)
        >>> print(p)
        [[0 1 0 0]
         [0 0 0 1]
         [1 0 0 0]
         [0 0 1 0]]
        >>> print(l)
        [[ 1.          0.          0.          0.        ]
         [ 0.2857143   1.          0.          0.        ]
         [ 0.71428573  0.12        1.          0.        ]
         [ 0.71428573 -0.44       -0.46153846  1.        ]]
        >>> print(u)
        [[ 7.          5.          6.          6.        ]
         [ 0.          3.57142854  6.28571415  5.28571415]
         [ 0.          0.         -1.03999996  3.07999992]
         [ 0.         -0.         -0.          7.46153831]]
    """
    func_name = "lu"
    _type_check(func_name, permute_l, bool, 'permute_l')
    _type_check(func_name, overwrite_a, bool, 'overwrite_a')
    _type_check(func_name, check_finite, bool, 'check_finite')
    _mstype_check(func_name, a, mstype.TensorType)
    _dtype_check(func_name, a, [mstype.int32, mstype.int64, mstype.float32, mstype.float64])
    _value_check(func_name, a.ndim, 2, 'a', 'dimension')

    if F.dtype(a) in (mstype.int32, mstype.int64):
        a = F.cast(a, mstype.float64)

    msp_lu = LU()
    m_lu, _, p = msp_lu(a)
    m = a.shape[-2]
    n = a.shape[-1]
    if m > n:
        _raise_value_error("last two dimensions of LU decomposition must be row less or equal to col.")
    k = min(m, n)
    l = mnp.tril(m_lu, -1)[..., :k] + mnp.eye(m, k, dtype=F.dtype(a))
    u = mnp.triu(m_lu)[:k, :]
    if permute_l:
        return mnp.dot(p, l), u
    return p, l, u


def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    """Solve an equation system, a x = b, given the LU factorization of a

    Note:
        - `lu_solve` is not supported on Windows platform yet.
        - Only `float32`, `float64`, `int32`, `int64` are supported Tensor dtypes. If Tensor with dtype `int32` or
          `int64` is passed, it will be cast to :class:`mstype.float64`.

    Args:
        lu_and_piv (Tensor, Tensor): Factorization of the coefficient matrix a, as given by lu_factor
        b (Tensor): Right-hand side
        trans (int, optional): {0, 1, 2}
            Type of system to solve:
            =====  =========
            trans  system
            =====  =========
            0      a x   = b
            1      a^T x = b
            2      a^H x = b
            =====  =========
        overwrite_b (bool, optional): Whether to overwrite data in b (may increase performance)
        check_finite ( bool, optional): Whether to check that the input matrices contain only finite numbers.
            Disabling may give a performance gain, but may result in problems (crashes, non-termination)
            if the inputs do contain infinities or NaNs.

    Returns:
        Tensor, solution to the system

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore import Tensor
        >>> from mindspore.scipy.linalg import lu_factor, lu_solve
        >>> a = Tensor(onp.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]]).astype(onp.float64))
        >>> b = Tensor(onp.array([1, 1, 1, 1]).astype(onp.float64))
        >>> lu, piv = lu_factor(a)
        >>> print(lu_solve((lu, piv), b))
        [ 0.05154639, -0.08247423,  0.08247423,  0.09278351]
    """
    func_name = "lu_solve"
    lu_matrix, pivot = lu_and_piv
    _type_check(func_name, overwrite_b, bool, 'overwrite_b')
    _type_check(func_name, check_finite, bool, 'check_finite')
    _mstype_check(func_name, lu_matrix, mstype.TensorType, 'lu_matrix')
    _mstype_check(func_name, b, mstype.TensorType, 'b')
    _mstype_check(func_name, pivot, mstype.TensorType, 'pivot')
    _dtype_check(func_name, lu_matrix, [mstype.int32, mstype.int64, mstype.float32, mstype.float64], 'lu_matrix')
    _dtype_check(func_name, b, [mstype.int32, mstype.int64, mstype.float32, mstype.float64], 'b')
    _dtype_check(func_name, pivot, [mstype.int32], 'pivot')
    _solve_check(func_name, lu_matrix, b, 'lu_matrix', 'b')
    _value_check(func_name, pivot.ndim, 1, 'pivot', 'dimension')
    _value_check(func_name, lu_matrix.shape, pivot.shape, 'lu_matrix', 'pivot', op='solve', fmt='solve')
    _value_check(func_name, trans, (0, 1, 2), 'trans', 'value')

    if F.dtype(lu_matrix) in (mstype.int32, mstype.int64):
        lu_matrix = F.cast(lu_matrix, mstype.float64)
        b = F.cast(b, mstype.float64)

    permutation = lu_pivots_to_permutation(pivot, pivot.size)
    rhs_vector = lu_matrix.ndim == b.ndim + 1
    x = b[permutation, :]
    if trans == 0:
        x = SolveTriangular(lower=True, unit_diagonal=True, trans='N')(lu_matrix, x)
        x = SolveTriangular(lower=False, unit_diagonal=False, trans='N')(lu_matrix, x)
    else:
        x = SolveTriangular(lower=False, unit_diagonal=False, trans='T')(lu_matrix, x)
        x = SolveTriangular(lower=True, unit_diagonal=True, trans='T')(lu_matrix, x)
    x = mnp.reshape(x, b.shape)
    return x[..., 0] if rhs_vector else x


def _det_2x2(a):
    return (a[..., 0, 0] * a[..., 1, 1] -
            a[..., 0, 1] * a[..., 1, 0])


def _det_3x3(a):
    return (a[..., 0, 0] * a[..., 1, 1] * a[..., 2, 2] +
            a[..., 0, 1] * a[..., 1, 2] * a[..., 2, 0] +
            a[..., 0, 2] * a[..., 1, 0] * a[..., 2, 1] -
            a[..., 0, 2] * a[..., 1, 1] * a[..., 2, 0] -
            a[..., 0, 0] * a[..., 1, 2] * a[..., 2, 1] -
            a[..., 0, 1] * a[..., 1, 0] * a[..., 2, 2])


def det(a, overwrite_a=False, check_finite=True):
    """
    Compute the determinant of a matrix

    The determinant of a square matrix is a value derived arithmetically
    from the coefficients of the matrix.

    The determinant for a 3x3 matrix, for example, is computed as follows::

        a    b    c
        d    e    f = A
        g    h    i

        det(A) = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h

    Note:
        - `det` is not supported on Windows platform yet.
        - Only `float32`, `float64`, `int32`, `int64` are supported Tensor dtypes. If Tensor with dtype `int32` or
          `int64` is passed, it will be cast to :class:`mstype.float64`.

    Args:
        a (Tensor): A square matrix to compute. Note that if the input tensor is not a `float`,
            then it will be cast to :class:`mstype.float32`.
        overwrite_a (bool, optional): Allow overwriting data in a (may enhance performance).
        check_finite (bool, optional): Whether to check that the input matrix contains
            only finite numbers.
            Disabling may give a performance gain, but may result in problems
            (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Raises:
        ValueError: If :math:`a` is not square.

    Returns:
        Tensor, Determinant of `a`.

    Examples:
        >>> import numpy as onp
        >>> from mindspore import Tensor
        >>> from mindspore.scipy.linalg import det
        >>> a = Tensor(onp.array([[0, 2, 3], [4, 5, 6], [7, 8, 9]])).astype(onp.float64)
        >>> print(det(a))
        3.0
    """
    func_name = "det"
    _type_check(func_name, overwrite_a, bool, 'overwrite_a')
    _type_check(func_name, check_finite, bool, 'check_finite')
    _mstype_check(func_name, a, mstype.TensorType)
    _square_check(func_name, a)
    _dtype_check(func_name, a, [mstype.int32, mstype.int64, mstype.float32, mstype.float64])

    if F.dtype(a) in (mstype.int32, mstype.int64):
        a = F.cast(a, mstype.float64)
    # special case
    if a.shape[-2] == 2:
        return _det_2x2(a)
    if a.shape[-2] == 3:
        return _det_3x3(a)

    lu_matrix, pivot = lu_factor(a)
    diag = lu_matrix.diagonal(axis1=-2, axis2=-1)
    pivot_not_equal = (pivot != mnp.arange(a.shape[-1])).astype(mstype.int64)
    pivot_sign = mnp.count_nonzero(pivot_not_equal, axis=-1)
    sign = -2. * (pivot_sign % 2) + 1.
    return sign * P.ReduceProd(keep_dims=False)(diag, -1)

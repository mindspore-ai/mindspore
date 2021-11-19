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
from .. import numpy as mnp
from .. import ops
from .ops import SolveTriangular
from .ops import CholeskySolver
from .ops import Cholesky
from .ops import EighNet
from ..ops import operations as P

__all__ = ['block_diag', 'solve_triangular', 'inv', 'cho_factor', 'cholesky', 'cho_solve', 'eigh']


def block_diag(*arrs):
    """
    Create a block diagonal matrix from provided arrays.

    Given the inputs `A`, `B` and `C`, the output will have these
    Tensor arranged on the diagonal::

        [[A, 0, 0],
         [0, B, 0],
         [0, 0, C]]

    Args:
        A, B, C, ... (Tensor): up to 2-D
            Input Tensors.  A 1-D Tensor or a 2-D Tensor with shape ``(1,n)``.

    Returns:
        D (Tesnor): Tensor with `A`, `B`, `C`, ... on the diagonal. `D` has
            the same dtype as `A`.

    Raises:
        ValueError: If there are tensors with dimensions higher than 2 in all arguments.

    Supported Platforms:
        ``CPU`` ``GPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore.common import Tensor
        >>> from mindspore.scipy.linalg import block_diag
        >>> A = Tensor(onp.array([[1, 0], [0, 1]]))
        >>> B = Tensor(onp.array([[3, 4, 5], [6, 7, 8]]))
        >>> C = Tensor(onp.array([[7]]))
        >>> P = Tensor(onp.zeros((2, ), dtype='int32'))
        >>> block_diag(A, B, C)
        [[1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 3, 4, 5, 0],
         [0, 0, 6, 7, 8, 0],
         [0, 0, 0, 0, 0, 7]]
        >>> block_diag(A, P, B, C)
        [[1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 3, 4, 5, 0],
         [0, 0, 6, 7, 8, 0],
         [0, 0, 0, 0, 0, 7]]
    """
    if not arrs:
        return mnp.zeros((1, 0))
    bad_shapes = [i for i, a in enumerate(arrs) if a.ndim > 2]
    if bad_shapes:
        raise ValueError("Arguments to mindspore.scipy.linalg.block_diag must have at "
                         "most 2 dimensions, got {} at argument {}."
                         .format(arrs[bad_shapes[0]], bad_shapes[0]))
    arrs = [mnp.atleast_2d(a) for a in arrs]
    accum = arrs[0]
    for arr in arrs[1:]:
        _, c = arr.shape
        arr = ops.Pad(((0, 0), (accum.shape[-1], 0)))(arr)
        accum = ops.Pad(((0, 0), (0, c)))(accum)
        accum = mnp.concatenate([accum, arr], axis=0)
    return accum


def solve_triangular(A, b, trans=0, lower=False, unit_diagonal=False,
                     overwrite_b=False, debug=None, check_finite=True):
    """
    Solve the equation `a x = b` for `x`, assuming a is a triangular matrix.

    Args:
        A (Tensor): A triangular matrix of shape :math:`(N, N)`.
        b (Tensor): A tensor of shape :math:`(M,)` or :math:`(M, N)`.
            Right-hand side matrix in :math:`A x = b`.
        lower (bool, optional): Use only data contained in the lower triangle of `a`.
            Default is to use upper triangle.
        trans (0, 1, 2, 'N', 'T', 'C', optional):
            Type of system to solve:

            ========  =========
            trans     system
            ========  =========
            0 or 'N'  a x  = b
            1 or 'T'  a^T x = b
            2 or 'C'  a^H x = b
            ========  =========
        unit_diagonal (bool, optional): If True, diagonal elements of :math:`A` are assumed to be 1 and
            will not be referenced.
        overwrite_b (bool, optional): Allow overwriting data in :math:`b` (may enhance performance)
        check_finite (bool, optional): Whether to check that the input matrices contain only finite numbers.
            Disabling may give a performance gain, but may result in problems
            (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns:
        x (Tensor): A tensor of shape :math:`(M,)` or :math:`(M, N)`,
            which is the solution to the system :math:`A x = b`.
            Shape of :math:`x` matches :math:`b`.

    Raises:
        LinAlgError: If :math:`A` is singular

    Supported Platforms:
        ``CPU`` ``GPU``

    Examples:
        Solve the lower triangular system :math:`A x = b`, where:

                 [3  0  0  0]       [4]
            A =  [2  1  0  0]   b = [2]
                 [1  0  1  0]       [4]
                 [1  1  1  1]       [2]

        >>> import numpy as onp
        >>> from mindspore.common import Tensor
        >>> import mindspore.numpy as mnp
        >>> from mindspore.scipy.linalg import solve_triangular
        >>> A = Tensor(onp.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]], onp.float64))
        >>> b = Tensor(onp.array([4, 2, 4, 2], onp.float64))
        >>> x = solve_triangular(A, b, lower=True, unit_diagonal=False, trans='N')
        >>> x
        Tensor(shape=[4], dtype=Float32, value= [ 1.33333337e+00, -6.66666746e-01,  2.66666651e+00, -1.33333313e+00])
        >>> mnp.dot(A, x)  # Check the result
        Tensor(shape=[4], dtype=Float32, value= [ 4.00000000e+00,  2.00000000e+00,  4.00000000e+00,  2.00000000e+00])
    """
    if isinstance(trans, int):
        trans_table = ['N', 'T', 'C']
        trans = trans_table[trans]
    solve = SolveTriangular(lower, unit_diagonal, trans)
    return solve(A, b)


def inv(a, overwrite_a=False, check_finite=True):
    """
    Compute the inverse of a matrix.

    Args:
        a (Tensor): Tensor
            Square matrix to be inverted.
        overwrite_a (bool, optional): Discard data in `a` (may improve performance).
            Default is False.
        check_finite (bool, optional): Whether to check that the input matrix contains
            only finite numbers.
            Disabling may give a performance gain, but may result in problems
            (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns:
        ainv (Tensor): Inverse of the matrix `a`.

    Raises:
        LinAlgError: If `a` is singular.
        ValueError: If `a` is not square, or not 2D.

    Supported Platforms:
        ``CPU`` ``GPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore.common import Tensor
        >>> import mindspore.numpy as mnp
        >>> from mindspore.scipy.linalg import inv
        >>> a = Tensor(onp.array([[1., 2.], [3., 4.]]))
        >>> inv(a)
        [[-2. ,  1. ],
         [ 1.5, -0.5]]
        >>> mnp.dot(a, inv(a))
        [[ 1.,  0.],
         [ 0.,  1.]]
    """
    matrix_inverse = P.MatrixInverse(adjoint=False)
    return matrix_inverse(a)


def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    """
    Compute the Cholesky decomposition of a matrix, to use in cho_solve

    Returns a matrix containing the Cholesky decomposition,
    ``A = L L*`` or ``A = U* U`` of a Hermitian positive-definite matrix `a`.
    The return value can be directly used as the first parameter to cho_solve.

    .. warning::
        The returned matrix also contains random data in the entries not
        used by the Cholesky decomposition. If you need to zero these
        entries, use the function `cholesky` instead.

    Args:
        a (Tensor): square Matrix of (M, M) to be decomposed
        lower (bool, optional): Whether to compute the upper or lower triangular Cholesky factorization
            (Default: upper-triangular)
        overwrite_a(bool, optional): Whether to overwrite data in a (may improve performance)
        check_finite(bool, optional): Whether to check that the input matrix contains only finite numbers.
            Disabling may give a performance gain, but may result in problems
            (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns:
        c (Tensor): Matrix whose upper or lower triangle contains the Cholesky factor of `a`.
         Other parts of the matrix contain random data.
        lower (bool, optional): Flag indicating whether the factor is in the lower or upper triangle

    Raises:
        LinAlgError: Raised if decomposition fails.

    Supported Platforms:
        ``CPU`` ``GPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore.common import Tensor
        >>> from mindspore.scipy.linalg import cho_factor
        >>> A = Tensor(onp.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]]).astype(onp.float32))
        >>> c, low = cho_factor(A)
        >>> c
        [[ 2.9999998   0.99999994  0.3333333   1.6666665 ]
         [ 0.          2.4494896   1.9051585  -0.27216542]
         [ 0.          0.          2.2933078   0.8559527 ]
         [ 0.          0.          0.          1.5541859 ]]
    """
    cholesky_net = Cholesky(lower=lower, clean=False)
    c = cholesky_net(a)
    return c, lower


def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
    """
    Compute the Cholesky decomposition of a matrix.

    Returns the Cholesky decomposition, :math:`A = L L^*` or
    :math:`A = U^* U` of a Hermitian positive-definite matrix A.

    Args:
        a (Tensor): square Matrix of (M, M) to be decomposed
        lower (bool, optional): Whether to compute the upper- or lower-triangular Cholesky
            factorization.  Default is upper-triangular.
        overwrite_a (bool, optional): Whether to overwrite data in `a` (may improve performance).
        check_finite (bool, optional): Whether to check that the input matrix contains only finite numbers.
            Disabling may give a performance gain, but may result in problems
            (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns:
        c (Tensor): Upper- or lower-triangular Cholesky factor of `a`.

    Raises:
        LinAlgError: if decomposition fails.

    Supported Platforms:
        ``CPU`` ``GPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore.common import Tensor
        >>> from mindspore.scipy.linalg import cholesky
        >>> a = Tensor(onp.array([[1, -2],[2, 5]]).astype(onp.float32))
        >>> L = cholesky(a, lower=True)
        >>> L
        [[1., 0.],
         [2., 1.]]
    """
    cholesky_net = Cholesky(lower=lower, clean=True)
    c = cholesky_net(a)
    return c


def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    """Solve the linear equations Ax = b, given the Cholesky factorization of A.

    Args:
        c_and_lower ((Tensor, bool)): Cholesky factorization of a, as given by cho_factor
        b (Tensor): Right-hand side
        overwrite_b (bool, optional): Whether to overwrite data in b (may improve performance)
        check_finite (bool, optional): Whether to check that the input matrices contain only finite numbers.
            Disabling may give a performance gain, but may result in problems
            (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns:
        x (Tensor):
            The solution to the system A x = b

    Supported Platforms:
        ``CPU`` ``GPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore.common import Tensor
        >>> from mindspore.scipy.linalg import cho_factor, cho_solve
        >>> A = Tensor(onp.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]]).astype(onp.float32))
        >>> b = Tensor(onp.array([1, 1, 1, 1]).astype(onp.float32))
        >>> c, low = cho_factor(A)
        >>> x = cho_solve((c, low), b)
        >>> x
        [-0.01749271,  0.11953353,  0.01166181,  0.1574344 ]
    """
    (c, lower) = c_and_lower
    cholesky_solver_net = CholeskySolver(lower=lower)
    x = cholesky_solver_net(c, b)
    return x


def eigh(a, b=None, lower=True, eigvals_only=False, overwrite_a=False,
         overwrite_b=False, turbo=True, eigvals=None, _type=1,
         check_finite=True):
    """
    Solve a standard or generalized eigenvalue problem for a complex
    Hermitian or real symmetric matrix.

    Find eigenvalues Tensor ``w`` and optionally eigenvectors Tensor ``v`` of
    Tensor ``a``, where ``b`` is positive definite such that for every
    eigenvalue λ (i-th entry of w) and its eigenvector ``vi`` (i-th column of
    ``v``) satisfies::

                      a @ vi = λ * b @ vi
        vi.conj().T @ a @ vi = λ
        vi.conj().T @ b @ vi = 1

    In the standard problem, ``b`` is assumed to be the identity matrix.

    Args:
        a (Tensor): (M, M) Tensor
            A complex Hermitian or real symmetric matrix whose eigenvalues and
            eigenvectors will be computed.
        b (Tensor, optional): (M, M) Tensor
            A complex Hermitian or real symmetric definite positive matrix in.
            If omitted, identity matrix is assumed.
        lower (bool, optional): Whether the pertinent Tensor data is taken from
            the lower or upper triangle of ``a`` and, if applicable, ``b``. (Default: lower)
        eigvals_only (bool, optional): Whether to calculate only eigenvalues
            and no eigenvectors. (Default: both are calculated)
        _type (int, optional): For the generalized problems, this keyword specifies
            the problem type to be solved for ``w`` and ``v`` (only takes 1, 2, 3 as possible
            inputs)::

                1 =>     a @ v = w @ b @ v
                2 => a @ b @ v = w @ v
                3 => b @ a @ v = w @ v

            This keyword is ignored for standard problems.
        overwrite_a (bool, optional): Whether to overwrite data in ``a``
            (may improve performance). Default is False.
        overwrite_b (bool, optional): Whether to overwrite data in ``b``
            (may improve performance). Default is False.
        check_finite (bool, optional): Whether to check that the input matrices
            contain only finite numbers.
            Disabling may give a performance gain, but may result in problems
            (crashes, non-termination) if the inputs do contain infinities or NaNs.
        turbo (bool, optional): use divide and conquer algorithm (faster but
            expensive in memory, only for generalized eigenvalue problem and
            if full set of eigenvalues are requested.). Has no significant
            effect if eigenvectors are not requested.
        eigvals (tuple, optional): Indexes of the smallest and largest (in ascending order)
            eigenvalues and corresponding eigenvectors to be returned: 0 <= lo <= hi <= M-1.
            If omitted, all eigenvalues and eigenvectors are returned.

    Returns:
        w (Tensor): (N,) Tensor, The N (1<=N<=M) selected eigenvalues, in ascending order,
            each repeated according to its multiplicity.
        v (Tensor): (M, N) Tensor, (if ``eigvals_only == False``)

    Raises:
        LinAlgError: If eigenvalue computation does not converge, an error occurred, or
            b matrix is not definite positive. Note that if input matrices are
            not symmetric or Hermitian, no error will be reported but results will
            be wrong.

    Supported Platforms:
        ``CPU`` ``GPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore.common import Tensor
        >>> from mindspore.scipy.linalg import eigh
        >>> A = Tensor(onp.array([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]]))
        >>> w, v = eigh(A)
        >>> onp.allclose(A @ v - v @ onp.diag(w), onp.zeros((4, 4)))
        True
    """
    eigh_net = EighNet(not eigvals_only, lower=True)
    return eigh_net(a)

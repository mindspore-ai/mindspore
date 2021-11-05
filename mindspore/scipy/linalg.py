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

__all__ = ['block_diag', 'solve_triangular']


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

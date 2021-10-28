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

__all__ = ['block_diag']


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

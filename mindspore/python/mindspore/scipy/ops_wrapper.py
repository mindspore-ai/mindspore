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
"""Linear algebra submodule"""
from .. import numpy as mnp
from ..common import dtype as mstype
from .utils import _to_tensor
from .utils_const import _raise_value_error
from ..ops.operations.array_ops import MatrixSetDiagV3


def matrix_set_diag(input_x, diagonal, k=0, alignment="RIGHT_LEFT"):
    """
    Calculate a batched matrix tensor with new batched diagonal values.

    Args:
        input_x (Tensor): a :math:`(..., M, N)` matrix to be set diag.
        diagonal (Tensor): a :math`(..., max_diag_len)`, or `(..., num_diags, max_diag_len)` vector to be placed to
            output's diags.
        k (Tensor): a scalar or 1D list. it's max length is to which indicates the diag's lower index and upper index.
            (k[0], k[1]).
        alignment (str): Some diagonals are shorter than `max_diag_len` and need to be padded.
            `align` is a string specifying how superdiagonals and subdiagonals should be aligned,
             respectively. There are four possible alignments: "RIGHT_LEFT" (default),
             "LEFT_RIGHT", "LEFT_LEFT", and "RIGHT_RIGHT". "RIGHT_LEFT" aligns superdiagonals to
             the right (left-pads the row) and subdiagonals to the left (right-pads the row).

    Returns:
        - Tensor, :math:`(...,M, N)`. a batched matrix with the same shape and values as `input`,
            except for the specified diagonals of the innermost matrices.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore.common import Tensor
        >>> from mindspore.scipy.ops_wrapper import matrix_set_diag
        >>> input_x = Tensor(
        >>>     onp.array([[[7, 7, 7, 7],[7, 7, 7, 7], [7, 7, 7, 7]],
        >>>                 [[7, 7, 7, 7],[7, 7, 7, 7],[7, 7, 7, 7]]])).astype(onp.int)
        >>> diagonal = Tensor(onp.array([[1, 2, 3],[4, 5, 6]])).astype(onp.int)
        >>> output =  matrix_set_diag(input_x, diagonal)
        >>> print(output)
        >>> [[[1 7 7 7]
              [7 2 7 7]
              [7 7 3 7]]

              [[4 7 7 7]
               [7 5 7 7]
               [7 7 6 7]]
    """
    matrix_set_diag_net = MatrixSetDiagV3(alignment)
    k_vec = mnp.zeros((2,), dtype=mstype.int32)
    if isinstance(k, int):
        k_vec += k
    elif isinstance(k, (list, tuple)):
        k_vec = k
    else:
        _raise_value_error("input k to indicate diagonal region is invalid.")
    k_vec = _to_tensor(k_vec, dtype=mstype.int32)
    output = matrix_set_diag_net(input_x, diagonal, k_vec)
    return output

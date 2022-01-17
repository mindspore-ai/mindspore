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
from .ops import MatrixSetDiag
from ..common import dtype as mstype
from .utils_const import _raise_value_error


def matrix_set_diag(input_x, diagonal, k=0, alignment="RIGHT_LEFT"):
    """
    Returns a batched matrix tensor with new batched diagonal values.
    Given `input` and `diagonal`, this operation returns a tensor with the same shape and values as `input`,
    except for the specified diagonals of the innermost matrices. These will be overwritten by the values in `diagonal`.
    `input` has `r+1` dimensions `[I, J, ..., L, M, N]`. When `k` is scalar or `k[0] == k[1]`,
    `diagonal` has `r` dimensions `[I, J, ..., L, max_diag_len]`. Otherwise, it has `r+1` dimensions
    `[I, J, ..., L, num_diags, max_diag_len]`. `num_diags` is the number of diagonals, `num_diags = k[1] - k[0] + 1`.
    `max_diag_len` is the longest diagonal in the range `[k[0], k[1]]`,
    `max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))` The output is a tensor of rank `k+1` with
     dimensions `[I, J, ..., L, M, N]`. If `k` is scalar or `k[0] == k[1]`:
    ```
    output[i, j, ..., l, m, n]
        = diagonal[i, j, ..., l, n-max(k[1], 0)] ; if n - m == k[1]
        input[i, j, ..., l, m, n]              ; otherwise
    ```
    Otherwise,
    ```
    output[i, j, ..., l, m, n]
        = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
        input[i, j, ..., l, m, n]                         ; otherwise
    ```
    where `d = n - m`, `diag_index = k[1] - d`, and `index_in_diag = n - max(d, 0) + offset`.
    `offset` is zero except when the alignment of the diagonal is to the right.
    ```
    offset = max_diag_len - diag_len(d) ; if (`align` in {RIGHT_LEFT, RIGHT_RIGHT}
                                             and `d >= 0`) or
                                           (`align` in {LEFT_RIGHT, RIGHT_RIGHT}
                                             and `d <= 0`)
           0                          ; otherwise
    ```
    where `diag_len(d) = min(cols - max(d, 0), rows + min(d, 0))`.

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
        ``CPU`` ``GPU``

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
    matrix_set_diag_net = MatrixSetDiag(alignment)
    k_vec = mnp.zeros((2,), dtype=mstype.int32)
    if isinstance(k, int):
        k_vec += k
    elif isinstance(k, (list, tuple)):
        k_vec = k
    else:
        _raise_value_error("input k to indicate diagonal region is invalid.")
    output = matrix_set_diag_net(input_x, diagonal, k_vec)
    return output

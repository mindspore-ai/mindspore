# Copyright 2023 Huawei Technologies Co., Ltd
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
# ===========================================================================
"""Linear Sum Assignment"""
import sys
from ..ops import LinearSumAssignment
from ... import Tensor


def linear_sum_assignment(cost_matrix, maximize, dimension_limit=Tensor(sys.maxsize)):
    r"""
    Solve the linear sum assignment problem.

    The assignment problem is represented as follows:

    .. math::
        min\sum_{i}^{} \sum_{j}^{} C_{i,j} X_{i,j}

    where :math:`C` is cost matrix, :math:`X_{i,j} = 1` means column :math:`j` is assigned to row :math:`i` .

    Args:
        cost_matrix (Tensor): 2-D cost matrix. Tensor of shape :math:`(M, N)` .
        maximize (bool): Calculate a maximum weight matching if true, otherwise calculate a minimum weight matching.
        dimension_limit (Tensor, optional): A scalar used to limit the actual size of the 2nd dimension of
            ``cost_matrix``. Default is ``Tensor(sys.maxsize)``, which means no limitation. The type is 0-D int64
            Tensor.

    Returns:
        A tuple of tensors containing 'row_idx' and 'col_idx'.

        - **row_idx** (Tensor) - Row indices of the problem. If `dimension_limit` is given, -1 would be padded at the
          end. The shape is  :math:`(N, )` , where :math:`N` is the minimum value of `cost_matrix` dimension.
        - **col_idx** (Tensor) - Column indices of the problem. If `dimension_limit` is given, -1 would be padded at
          the end. The shape is  :math:`(N, )` , where :math:`N` is the minimum value of `cost_matrix` dimension.

    Raises:
        TypeError: If the data type of `cost_matrix` is not the type in [float16, float32, float64,
                   int8, int16, int32, int64, uint8, uint16, uint32, uint64, bool]
        TypeError: If the type of `maximize` is not bool.
        TypeError: If the data type of `dimension_limit` is not int64.
        ValueError: If the rank of `cost_matrix` is not 2.


    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.scipy.optimize.linear_sum_assignment as lsap
        >>> cost_matrix = Tensor(np.array([[2, 3, 3], [3, 2, 3], [3, 3, 2]])).astype(ms.float64)
        >>> dimension_limit = Tensor(2)
        >>> maximize = False
        >>> a, b = lsap(cost_matrix, maximize, dimension_limit)
        >>> print(a)
        [0 1 -1]
        >>> print(b)
        [0 1 -1]
        >>> a, b = lsap(cost_matrix, maximize)
        >>> print(a)
        [0 1 2]
        >>> print(b)
        [0 1 2]
    """
    solve = LinearSumAssignment()
    return solve(cost_matrix, dimension_limit, maximize)

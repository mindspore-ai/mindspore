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
    """
    Solve the linear sum assignment problem.

    Args:
        cost_matrix (Tensor): 2-D Input Tensor.
            The cost matrix of the bipartite graph.
        maximize (bool): bool.
            Calculates a maximum weight matching if true.
        dimension_limit (Tensor): 0-D Input Tensor.
            A scalar used to limit the actual size of the 2nd dimension. Optimized for
            padding scenes. Default means no dimension limit.

    Returns:
        1-D Output Tensors with 'row_idx' and 'col_idx'. An array of row indices and
        one of corresponding column indices giving the optimal assignment. If specified
        dimension_limit, padding value at the end would be -1.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.scipy.optimize.linear_sum_assignment as lsap
        >>> cost_matrix = Tensor(np.array([[2, 3, 3], [3, 2, 3], [3, 3, 2]])).astype("float64")
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

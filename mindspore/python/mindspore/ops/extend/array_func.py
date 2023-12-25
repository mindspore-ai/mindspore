# Copyright 2024 Huawei Technologies Co., Ltd
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

"""

Array Operators

"""

from mindspore.ops.auto_generate.gen_ops_prim import gather_ext_op

# define Primitive global variables


def gather(input, dim, index):
    r"""
    Gather data from a tensor by indices.

    .. math::
        output[(i_0, i_1, ..., i_{dim}, i_{dim+1}, ..., i_n)] =
        input[(i_0, i_1, ..., index[(i_0, i_1, ..., i_{dim}, i_{dim+1}, ..., i_n)], i_{dim+1}, ..., i_n)]

    Args:
        input (Tensor): The target tensor to gather values.
        dim (int): the axis to index along, must be in range `[-input.rank, input.rank)`.
        index (Tensor): The index tensor, with int32 or int64 data type. An valid `index` should be:

            - `index.rank == input.rank`;
            - `index.shape[axis] <= input.shape[axis]` where axis goes through all dimensions of `input` except `dim`;
            - the value of `index` is in range `[-input.shape[dim], input.shape[dim])`.

    Returns:
        Tensor, has the same type as `input` and the same shape as `index`.

    Raises:
        ValueError: If the shape of `index` is illegal.
        ValueError: If `dim` is not in `[-input.rank, input.rank)`.
        ValueError: If the value of `index` is out of the valid range.
        TypeError: If the type of `index` is illegal.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> index = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
        >>> output = ops.extend.gather(input, 1, index)
        >>> print(output)
        [[-0.1 -0.1]
        [ 0.5  0.5]]
    """
    return gather_ext_op(input, dim, index)


__all__ = ['gather']

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

"""Operators for function."""

from mindspore.ops.primitive import constexpr
from mindspore.ops import operations as P


@constexpr
def get_x_shape(x_shape):
    s = 1
    for i in x_shape:
        s = s * i
    return (s,)


def unique(x):
    """
    Returns the unique elements of input tensor and also return a tensor containing the index of each value of input
    tensor corresponding to the output unique tensor.

    The output contains Tensor `y` and Tensor `idx`, the format is probably similar to (`y`, `idx`).
    The shape of Tensor `y` and Tensor `idx` is different in most cases, because Tensor `y` will be deduplicated,
    and the shape of Tensor `idx` is consistent with the input.

    To get the same shape between `idx` and `y`, please ref to :class:'mindspore.ops.UniqueWithPad' operator.

    .. warning::
        This module is in beta.

    Args:
        x (Tensor): The input tensor.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tuple, containing Tensor objects `(y, idx), `y` is a tensor with the
        same type as `input_x`, and contains the unique elements in `x`, sorted in
        ascending order. `idx` is a tensor containing indices of elements in
        the input corresponding to the output tensor, have the same shape with `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>> from mindspore import ops
        >>> input_x = Tensor(np.array([1, 2, 5, 2]), mindspore.int32)
        >>> output = ops.unique(input_x)
        >>> print(output)
        (Tensor(shape=[3], dtype=Int32, value= [1, 2, 5]), Tensor(shape=[4], dtype=Int32, value= [0, 1, 2, 1]))
        >>> y = output[0]
        >>> print(y)
        [1 2 5]
        >>> idx = output[1]
        >>> print(idx)
        [0 1 2 1]
        >>> # As can be seen from the above, y and idx shape
        >>> # note that for GPU, this operator must be wrapped inside a model, and executed in graph mode.
        >>> class UniqueNet(nn.Cell):
        ...     def __init__(self):
        ...         super(UniqueNet, self).__init__()
        ...
        ...     def construct(self, x):
        ...         output, indices = ops.unique(x)
        ...         return output, indices
        ...
        >>> input_x = Tensor(np.array([1, 2, 5, 2]), mindspore.int32)
        >>> net = UniqueNet()
        >>> output = net(input_x)
        >>> print(output)
        (Tensor(shape=[3], dtype=Int32, value= [1, 2, 5]), Tensor(shape=[4], dtype=Int32, value= [0, 1, 2, 1]))
    """

    unique_op = P.Unique()
    reshape_op = P.Reshape()

    shape_x = x.shape
    length_x = get_x_shape(shape_x)
    x = reshape_op(x, length_x)
    y, idx = unique_op(x)
    idx = reshape_op(idx, shape_x)
    return y, idx

# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""ReflectionPad"""
from mindspore.common import Tensor
import mindspore.ops as ops
from mindspore.ops.primitive import constexpr
from ..cell import Cell

__all__ = ['ReflectionPad1d', 'ReflectionPad2d']


@constexpr
def _check_padding_dimension(dimension, padding):
    r"""
    Validate the input padding and add place holders if needed.
    Note: the input 'padding' in this function is already converted to list of lists to match MirrorPad
    """
    if dimension < len(padding):
        raise ValueError(f"For padding with length {len(padding) * 2}, the dimension of the tensor should be at least "
                         f"{len(padding)}, but got {dimension}")
    # add place holders
    if dimension > len(padding):
        padding = [(0, 0) for _ in range(dimension - len(padding))] + [x for x in padding]
    return padding


def _swap_to_ms_padding_order(padding):
    r"""
    Check whether the input padding is a tuple or a int converted to a tuple.
    Check if the length of padding is divisible by 2.
    Convert the input padding to the format that MirrorPad would understand.
    """
    number_of_paddings = len(padding) // 2
    new_padding = [[0, 0]] * number_of_paddings
    for i in range(number_of_paddings):
        new_padding[i] = [padding[2 * i], padding[2 * i + 1]]
    # reverse the padding list to match the order of paddings for MirrorPad
    new_padding.reverse()
    return new_padding


class _ReflectionPadNd(Cell):
    r"""
    Using a given padding to do reflection pad on the given tensor.
    Work as a parent class, and only accepts tuple as padding input.
    """
    def __init__(self, padding, name="ReflectionPadNd"):
        super(_ReflectionPadNd, self).__init__()
        self.name = name
        # check if padding and its elements are valid
        if not isinstance(padding, tuple):
            raise TypeError(f"For '{self.name}' the input 'padding' must be an integer or tuple, "
                            f"but got {type(padding).__name__}")
        if len(padding) % 2 != 0:
            raise ValueError(f"For '{self.name}' the length of input 'padding' must be divisible by 2, "
                             f"but got padding of length {len(padding)}. ")
        if not all(isinstance(i, int) for i in padding):
            raise TypeError(f"For '{self.name}' every element in 'padding' must be integer, "
                            f"but got {padding}. ")
        if not all(i >= 0 for i in padding):
            raise ValueError(f"For '{self.name}' every element in 'padding' must be >= 0. "
                             f"but got {padding}. ")
        self.padding = _swap_to_ms_padding_order(padding)

    def construct(self, x):
        input_shape = x.shape
        padding = _check_padding_dimension(len(input_shape), self.padding)
        x = ops.MirrorPad(mode='REFLECT')(x, Tensor(padding))
        return x


class ReflectionPad1d(_ReflectionPadNd):
    r"""
    Using a given padding to do reflection pad on the given tensor.

    Args:
        padding (union[int, tuple]): The padding size to pad the last dimension of input tensor.
            If padding is an integer: all directions will be padded with the same size.
            If padding is a tuple: uses (pad_left, pad_right) to pad.

    Inputs:
        - **x** (Tensor) - 2D or 3D, shape: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.

    Outputs:
        Tensor, after padding. Shape: :math:`(C, W_{out})` or :math:`(N, C, W_{out})`,
        where :math:`W_{out} = W_{in} + pad_{left} + pad_{right}`.

    Raises:
        TypeError: If 'padding' is not a tuple or int.
        TypeError: If there is an element in 'padding' that is not int.
        ValueError: If the length of 'padding' is not divisible by 2.
        ValueError: If there is an element in 'padding' that is negative.
        ValueError: If the there is a dimension mismatch between the padding and the tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.nn import ReflectionPad1d
        >>> x = Tensor(np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]]).astype(np.float32))
        >>> # x has shape (1, 2, 4)
        >>> padding = (3, 1)
        >>> # The first and the second dimension of x remain the same.
        >>> # The third dimension of x: W_out = W_in + pad_left + pad_right = 4 + 3 + 1 = 8
        >>> pad1d = ReflectionPad1d(padding)
        >>> out = pad1d(x)
        >>> # The shape of out is (1, 2, 8)
        >>> print(out)
        [[[3. 2. 1. 0. 1. 2. 3. 2.]
          [7. 6. 5. 4. 5. 6. 7. 6.]]]
    """

    def __init__(self, padding):
        if isinstance(padding, int):
            padding = (padding, padding)
        super(ReflectionPad1d, self).__init__(padding, 'ReflectionPad1d')


class ReflectionPad2d(_ReflectionPadNd):
    r"""
    Using a given padding to do reflection pad the given tensor.

    Args:
        padding (union[int, tuple]): The padding size to pad the input tensor.
            If padding is an integer: all directions will be padded with the same size.
            If padding is a tuple: uses (pad_left, pad_right, pad_up, pad_down) to pad.

    Inputs:
        - **x** (Tensor) - 3D or 4D, shape: :math:`(C, H_{in}, W_{out})` or :math:`(N, C, H_{out}, W_{out})`.

    Outputs:
        Tensor, after padding. Shape: :math:`(C, H_{out}, W_{out})` or :math:`(N, C, H_{out}, W_{out})`,
        where :math:`H_{out} = H_{in} + pad_{up} + pad_{down}`,  :math:`W_{out} = W_{in} + pad_{left} + pad_{right}`.

    Raises:
        TypeError: If 'padding' is not a tuple or int.
        TypeError: If there is an element in 'padding' that is not int.
        ValueError: If the length of 'padding' is not divisible by 2.
        ValueError: If there is an element in 'padding' that is negative.
        ValueError: If the there is a dimension mismatch between the padding and the tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Example:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.nn import ReflectionPad2d
        >>> x = Tensor(np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]).astype(np.float32))
        >>> # x has shape (1, 3, 3)
        >>> padding = (1, 1, 2, 0)
        >>> pad2d = ReflectionPad2d(padding)
        >>> # The first dimension of x remains the same.
        >>> # The second dimension of x: H_out = H_in + pad_up + pad_down = 3 + 1 + 1 = 5
        >>> # The third dimension of x: W_out = W_in + pad_left + pad_right = 3 + 2 + 0 = 5
        >>> out = pad2d(x)
        >>> # The shape of out is (1, 5, 5)
        >>> print(out)
        [[[7. 6. 7. 8. 7.]
          [4. 3. 4. 5. 4.]
          [1. 0. 1. 2. 1.]
          [4. 3. 4. 5. 4.]
          [7. 6. 7. 8. 7.]]]
    """

    def __init__(self, padding):
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        super(ReflectionPad2d, self).__init__(padding, 'ReflectionPad2d')

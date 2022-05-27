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
    Using a given padding to do reflection pad on the last dimension of the given tensor.

        Args:
            padding (union[int, tuple]): The padding size to pad the last dimension of input tensor.
            If padding is an integer: all directions will be padded with the same size.
            If padding is a tuple: uses (pad_left, pad_right, pad_up, pad_down) to pad.

        Inputs:
            Tensor, 2D or 3D

        Outputs:
            Tensor, after padding.
            Suppose the tensor has dimension (N, C, W_in), the padded dimension will become (N, C, W_out),
                where W_out = W_in + pad_left + pad_right

        Raises:
            TypeError: If 'padding' is not a tuple or int.
            TypeError: If there is an element in 'padding' that is not int64.
            ValueError: If the length of 'padding' is not divisible by 2.
            ValueError: If there is an element in 'padding' that is negative.
            ValueError: If the there is a dimension mismatch between the padding and the tensor.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

    """

    def __init__(self, padding):
        if isinstance(padding, int):
            padding = (padding, padding)
        super(ReflectionPad1d, self).__init__(padding, 'ReflectionPad1d')


class ReflectionPad2d(_ReflectionPadNd):
    r"""
    Using a given padding to do reflection pad on the last dimension of the given tensor.

        Args:
            padding (union[int, tuple]): The padding size to pad the last dimension of input tensor.
            If padding is an integer: all directions will be padded with the same size.
            If padding is a tuple: uses (pad_left, pad_right, pad_up, pad_down) to pad.

        Inputs:
            Tensor, 3D or 4D

        Output:
            Tensor, after padding.
            Suppose the tensor has dimension (N, C, H_in, W_in), the padded dimension will become (N, C, H_out, W_out),
                where W_out = W_in + pad_left + pad_right, H_out = H_in + pad_up + pad_down

        Raises:
            TypeError: If 'padding' is not a tuple or int.
            TypeError: If there is an element in 'padding' that is not int64.
            ValueError: If the length of 'padding' is not divisible by 2.
            ValueError: If there is an element in 'padding' that is negative.
            ValueError: If the there is a dimension mismatch between the padding and the tensor.

        Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

    """

    def __init__(self, padding):
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        super(ReflectionPad2d, self).__init__(padding, 'ReflectionPad2d')

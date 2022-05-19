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
""" padding """
from mindspore import ops
from mindspore.ops.primitive import constexpr
from ..cell import Cell

__all__ = ['ConstantPad1d', 'ConstantPad2d', 'ConstantPad3d', 'ZeroPad2d']


@constexpr
def _check(input_shape, padding):
    """
    Check relationship between input shape and padding to make sure after negative dimension padding the out is
    positive.
    """
    if len(input_shape) < len(padding):
        msg = 'Dimension of input must more than or equal to len(padding)/2'  # modify
        raise ValueError(msg)
    if len(input_shape) > len(padding):
        if len(padding) == 2 and isinstance(padding[0], int):
            padding = [(0, 0) for i in range(len(input_shape) - 1)] + [padding]
        else:
            padding = [(0, 0) for i in range(len(input_shape) - len(padding))] + [x for x in padding]
    for index, item in enumerate(padding):
        if item[0] < -input_shape[index]:
            msg = 'Dimension out of range, expected no less than -{}, but got {}'.format(input_shape[index],
                                                                                         item[0])
            raise ValueError(msg)
        if item[1] < -input_shape[index]:
            msg = 'Dimension out of range, expected no less than -{}, but got {}'.format(input_shape[index],
                                                                                         item[1])
            raise ValueError(msg)
        if input_shape[index] + item[0] + item[1] <= 0:
            msg = 'The input size {}, plus negative padding {} and {} resulted in a non-positive output size, ' \
                  'which is invalid. Check dimension of your input'.format(input_shape[index], item[0], item[1])
            raise ValueError(msg)
    return padding


@constexpr
def _get_new_padding(padding):
    """get non-negative padding and make negative position."""
    new_padding = [[item[0], item[1]] for item in padding]
    start = [0 for i in range(len(new_padding))]
    end = [0 for i in range(len(new_padding))]
    for index, item in enumerate(new_padding):
        if item[0] < 0:
            start[index] = item[0]
            new_padding[index][0] = 0
        if item[1] < 0:
            end[index] = item[1]
            new_padding[index][1] = 0
    new_padding = tuple(new_padding)
    return new_padding, start, end


@constexpr
def _get_begin_size(shape, begin, end):
    """Calculate begin and size for ops.Slice."""
    size = tuple([shape[i] + begin[i] + end[i] for i in range(len(shape))])
    begin = tuple([int(-i) for i in begin])
    return begin, size


class _ConstantPadNd(Cell):
    r"""
    Using a given value to pads the last n dimensions of input tensor.

    Args:
        padding(tuple, list): The padding size to pad the last n dimensions of input tensor. The padding
            sequence is starting from the last dimension and moving forward. The length of padding must be
            a multiple of 2. len(padding)/2 dimensions of input will be padded.
        value(union[int, float]): Padding value.

         padding (union[list, tuple]): The padding size to pad the last n dimensions of input tensor.
            The padding sequence is starting from the last dimension and moving forward.
            The length of padding must be a multiple of 2. If padding is :math:`(padding_0, padding_1, padding_2,
            padding_3, ..., padding_2m, padding_{2m+1}, ...)`. The input is `x`,
            the size of last dimension of output is :math:`padding\_0 + x.shape[-1] + padding\_1`.
            The size of penultimate dimension of output is :math:`padding\_2 + x.shape[-2] + padding\_3`.
            The size of 3rd to last dimension of output is :math:`padding\_4 + x.shape[-3] + padding\_5`.
            The size of i-td to last dimension of output is :math:`padding\_{2m} + x.shape[-m-1] + padding\_{2m+1}`.
            The remaining dimensions of the output are consistent with those of the input.
        value (union[int, float]): Padding value.

    Returns:
        Tensor, the tensor after padding.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        ValueError: If the length of padding is not a multiple of 2.
        ValueError: If the length of input less than len(padding)/2.
        ValueError: If the output shape after padding is not positive.
    """

    def __init__(self, padding, value):
        """Initialize Pad."""
        super(_ConstantPadNd, self).__init__()
        self.value = value
        self.padding = self._to_ms_padding(padding)

    def _to_ms_padding(self, padding):
        """Transform the padding to the format of ms.nn.Pad."""
        if len(padding) % 2 != 0:
            msg = 'the length of padding must be a multiple of 2.'
            raise ValueError(msg)
        new_padding = []
        for i in range(len(padding) // 2):
            new_padding.append([padding[2 * i], padding[2 * i + 1]])
        new_padding.reverse()
        return new_padding

    def construct(self, x):
        """Construct the pad net."""
        input_shape = x.shape
        input_type = x.dtype
        padding = _check(input_shape, self.padding)
        new_padding, start, end = _get_new_padding(padding)
        mask = ops.Ones()(input_shape, input_type)
        output = ops.Pad(new_padding)(x)
        mask = ops.Pad(new_padding)(mask)
        ones = ops.Ones()(output.shape, output.dtype)
        value = ops.Fill()(output.dtype, output.shape, self.value)
        output = ops.Add()(ops.Mul()(mask, output), ops.Mul()(ops.Sub()(ones, mask), value))
        slice_op = ops.Slice()
        begin, size = _get_begin_size(output.shape, start, end)
        output = slice_op(output, begin, size)
        return output


class ConstantPad1d(_ConstantPadNd):
    r"""
    Using a given constant value to pads the last dimensions of input tensor.

    Args:
        padding (union[int, tuple]): The padding size to pad the last dimension of input tensor.
            If is int, uses the same padding in both boundaries of input's last dimension.
            If a 2-tuple, uses (padding_0, padding_1) to pad. If the input is `x`, the size of last
            dimension of output is :math:`padding\_0 + x.shape[-1] + padding\_1`. The remaining dimensions
            of the output are consistent with those of the input.
        value (union[int, float]): Padding value.

    Returns:
        Tensor, the tensor after padding.

    Raises:
        TypeError: If `padding` is not a tuple or int.
        TypeError: If `value` is not int or float.
        ValueError: If the length of `padding` with tuple type is not equal to 2.
        ValueError: If the output shape after padding is not positive.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.nn import ConstantPad2d
        >>> x = np.ones(shape=(1, 2, 3, 4)).astype(np.float32)
        >>> x = Tensor(x)
        >>> # padding is tuple
        >>> padding = (0, 1)
        >>> value = 0.5
        >>> pad1d = ConstantPad1d(padding, value)
        >>> out = pad1d(x)
        >>> print(out)
        [[[[1.  1.  1.  1.  0.5]
           [1.  1.  1.  1.  0.5]
           [1.  1.  1.  1.  0.5]]
          [[1.  1.  1.  1.  0.5]
           [1.  1.  1.  1.  0.5]
           [1.  1.  1.  1.  0.5]]]]
        >>> print(out.shape)
        (1, 2, 3, 5)
        >>> # padding is int
        >>> padding = 1
        >>> value = 0.5
        >>> pad1d = ConstantPad1d(padding, value)
        >>> out = pad1d(x)
        >>> print(out)
        [[[[0.5 1.  1.  1.  1.  0.5]
           [0.5 1.  1.  1.  1.  0.5]
           [0.5 1.  1.  1.  1.  0.5]]
          [[0.5 1.  1.  1.  1.  0.5]
           [0.5 1.  1.  1.  1.  0.5]
           [0.5 1.  1.  1.  1.  0.5]]]]
        >>> print(out.shape)
        (1, 2, 3, 6)
        >>> # padding is negative
        >>> padding = (-1, 0)
        >>> value = 0.5
        >>> pad1d = ConstantPad1d(padding, value)
        >>> out = pad1d(x)
        >>> print(out)
        [[[[1. 1. 1.]
           [1. 1. 1.]
           [1. 1. 1.]]
          [[1. 1. 1.]
           [1. 1. 1.]
           [1. 1. 1.]]]]
        >>> print(out.shape)
        (1, 2, 3, 3)
    """

    def __init__(self, padding, value):
        if isinstance(padding, int):
            padding = (padding, padding)
        elif isinstance(padding, tuple):
            if len(padding) != 2:
                msg = 'the length of padding with tuple type must be 2.'
                raise ValueError(msg)
        else:
            msg = 'type of padding must be int or float, but got {}'.format(type(padding))
            raise TypeError(msg)

        if not isinstance(value, (int, float)):
            msg = 'type of value must be int or float, but got {}'.format(type(value))
            raise TypeError(msg)
        super(ConstantPad1d, self).__init__(padding, value)


class ConstantPad2d(_ConstantPadNd):
    r"""
    Using a given constant value to pads the last two dimensions of input tensor.

    Args:
        padding (union[int, tuple]): The padding size to pad the last two dimensions of input tensor.
            If is int, uses the same padding in boundaries of input's last two dimensions.
            If is tuple and length of padding is 4 uses (padding_0, padding_1, padding_2, padding_3) to pad.
            If the input is `x`, the size of last dimension of output is :math:`padding\_0 + x.shape[-1] + padding\_1`.
            The size of penultimate dimension of output is :math:`padding\_2 + x.shape[-2] + padding\_3`.
            The remaining dimensions of the output are consistent with those of the input.
        value (union[int, float]): Padding value.

    Returns:
        Tensor, the tensor after padding.

    Raises:
        TypeError: If `padding` is not a tuple or int.
        TypeError: If `value` is not int or float.
        ValueError: If the length of `padding` is more than 4 or not a multiple of 2.
        ValueError: If the output shape after padding is not positive.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.nn import ConstantPad2d
        >>> x = np.ones(shape=(1, 2, 3, 4)).astype(np.float32)
        >>> x = Tensor(x)
        >>> padding = (-1, 1, 0, 1)
        >>> value = 0.5
        >>> pad2d = ConstantPad2d(padding, value)
        >>> out = pad2d(x)
        >>> print(out)
        [[[[1.  1.  1.  0.5]
           [1.  1.  1.  0.5]
           [1.  1.  1.  0.5]
           [0.5 0.5 0.5 0.5]]
          [[1.  1.  1.  0.5]
           [1.  1.  1.  0.5]
           [1.  1.  1.  0.5]
           [0.5 0.5 0.5 0.5]]]]
        >>> print(out.shape)
        (1, 2, 4, 4)
    """

    def __init__(self, padding, value):
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        elif isinstance(padding, tuple):
            if len(padding) // 2 > 2:
                msg = 'the length of padding with tuple type must less than or equal to 4.'
                raise ValueError(msg)
        else:
            msg = 'type of padding must be int or float, but got {}'.format(type(padding))
            raise TypeError(msg)

        if not isinstance(value, (int, float)):
            msg = 'type of value must be int or float, but got {}'.format(type(value))
            raise TypeError(msg)
        super(ConstantPad2d, self).__init__(padding, value)


class ConstantPad3d(_ConstantPadNd):
    r"""
    Using a given constant value to pads the last three dimensions of input tensor.

    Args:
        padding (union[int, tuple]): The padding size to pad the last two dimensions of input tensor.
            If is int, uses the same padding in boundaries of input's last two dimensions.
            If is tuple and length of padding is 6 uses
            (padding_0, padding_1, padding_2, padding_3, padding_4, padding_5) to pad. If the input is `x`,
            the size of last dimension of output is :math:`padding\_0 + x.shape[-1] + padding\_1`.
            The size of penultimate dimension of output is :math:`padding\_2 + x.shape[-2] + padding\_3`.
            The size of 3rd to last dimension of output is :math:`padding\_4 + x.shape[-3] + padding\_5`.
            The remaining dimensions of the output are consistent with those of the input.
        value (union[int, float]): Padding value.

    Returns:
        Tensor, the tensor after padding.

    Raises:
        TypeError: If `padding` is not a tuple or int.
        TypeError: If `value` is not int or float.
        ValueError: If the length of `padding` is more than 6 or not a multiple of 2.
        ValueError: If the output shape after padding is not positive.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.nn import ConstantPad3d
        >>> x = np.ones(shape=(1, 2, 3, 4)).astype(np.float32)
        >>> x = Tensor(x)
        >>> padding = (-1, 1, 0, 1, 1, 0)
        >>> value = 0.5
        >>> pad3d = ConstantPad3d(padding, value)
        >>> out = pad3d(x)
        >>> print(out)
        [[[[0.5 0.5 0.5 0.5]
           [0.5 0.5 0.5 0.5]
           [0.5 0.5 0.5 0.5]
           [0.5 0.5 0.5 0.5]]
          [[1.  1.  1.  0.5]
           [1.  1.  1.  0.5]
           [1.  1.  1.  0.5]
           [0.5 0.5 0.5 0.5]]
          [[1.  1.  1.  0.5]
           [1.  1.  1.  0.5]
           [1.  1.  1.  0.5]
           [0.5 0.5 0.5 0.5]]]]
        >>> print(out.shape)
        (1, 3, 4, 4)
    """

    def __init__(self, padding, value):
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding, padding, padding)
        elif isinstance(padding, tuple):
            if len(padding) // 2 > 3:
                msg = 'the length of padding with tuple type must less than or equal to 6.'
                raise ValueError(msg)
        else:
            msg = 'type of padding must be int or float, but got {}'.format(type(padding))
            raise ValueError(msg)

        if not isinstance(value, (int, float)):
            msg = 'type of value must be int or float, but got {}'.format(type(value))
            raise ValueError(msg)
        super(ConstantPad3d, self).__init__(padding, value)


class ZeroPad2d(ConstantPad2d):
    r"""
    Pads the last two dimensions of input tensor with zero.

    Args:
        padding (union[int, tuple]): The padding size to pad the last two dimensions of input tensor.
            If is int, uses the same padding in boundaries of input's last two dimensions.
            If is tuple and length of padding is 4 uses (padding_0, padding_1, padding_2, padding_3) to pad.
            If the input is `x`, the size of last dimension of output is :math:`padding\_0 + x.shape[-1] + padding\_1`.
            The size of penultimate dimension of output is :math:`padding\_2 + x.shape[-2] + padding\_3`.
            The remaining dimensions of the output are consistent with those of the input.

    Returns:
        Tensor, the tensor after padding.

    Raises:
        TypeError: If `padding` is not a tuple or int.
        ValueError: If the length of `padding` is more than 4 or not a multiple of 2.
        ValueError: If the output shape after padding is not positive.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.nn import ZeroPad2d
        >>> x = np.ones(shape=(1, 2, 3, 4)).astype(np.float32)
        >>> x = Tensor(x)
        >>> padding = (-1, 1, 0, 1)
        >>> pad = ZeroPad2d(padding)
        >>> out = pad(x)
        >>> print(out)
        [[[[1. 1. 1. 0.]
           [1. 1. 1. 0.]
           [1. 1. 1. 0.]
           [0. 0. 0. 0.]]
          [[1. 1. 1. 0.]
           [1. 1. 1. 0.]
           [1. 1. 1. 0.]
           [0. 0. 0. 0.]]]]
        >>> print(out.shape)
        (1, 2, 4, 4)
    """

    def __init__(self, padding):
        value = 0
        super(ZeroPad2d, self).__init__(padding, value)

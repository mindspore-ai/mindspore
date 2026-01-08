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
"""activation layer for mint"""
from __future__ import absolute_import

from mindspore import mint
from mindspore.nn.cell import Cell
from mindspore import _checkparam as validator


class ConstantPadNd_(Cell):
    """
    Base class for N-dimensional constant padding.
    """
    def __init__(self, padding, value=None, padding_length=None):
        super(ConstantPadNd_, self).__init__()
        self.padding = padding
        self.value = value

        if isinstance(self.padding, int):
            validator.check_positive_int(self.padding, "padding", self.cls_name)
            self.padding = (self.padding,) * padding_length
        elif isinstance(self.padding, (tuple, list)):
            if len(padding) != padding_length:
                msg = f"For '{self.cls_name}', the length of parameter 'padding' with tuple " \
                      f"type must equal to {padding_length}, but got {len(padding)}."
                raise ValueError(msg)
            validator.check_non_negative_int_sequence(self.padding, "padding", self.cls_name)
        else:
            msg = f"For '{self.cls_name}', 'padding' must be positive integer or tuple/list of {padding_length}" \
                  f" positive integers, but got {padding}."
            raise ValueError(msg)

    def construct(self, input):
        return mint.nn.functional.pad(input, self.padding, mode='constant', value=self.value)


class ConstantPad1d(ConstantPadNd_):
    r"""
    Pad the last dimension of `input` tensor using `padding` and `value`.

    For more information, please refer to :func:`mindspore.mint.nn.functional.pad`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        padding (Union[int, tuple, list]): Specifies padding size.
        value (Union[int, float]): Specifies padding value.

    Inputs:
        - **input** (Tensor) - shape is :math:`(N, *)`, where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, the tensor after padding.

    Raises:
        TypeError: If `padding` is not an integer of a list or tuple of 2 integers.
        TypeError: If `input` is not Tensor.
        TypeError: If `value` is not int or float.
        ValueError: If `padding` contains negative value.
        ValueError: If `padding` is a tuple or list, and the length does not match the tensor dimension.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> x = np.ones(shape=(1, 2, 3, 4)).astype(np.float32)
        >>> x = ms.Tensor(x)
        >>> # padding is tuple
        >>> padding = (0, 1)
        >>> value = 0.5
        >>> pad1d = ms.mint.nn.ConstantPad1d(padding, value)
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
        >>> pad1d = ms.mint.nn.ConstantPad1d(padding, value)
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
    """

    def __init__(self, padding, value):
        super(ConstantPad1d, self).__init__(padding, value, padding_length=2)


class ConstantPad2d(ConstantPadNd_):
    """
    Pad the last 2 dimensions of `input` tensor using `padding` and `value`.

    For more information, please refer to :func:`mindspore.mint.nn.functional.pad`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        padding (Union[int, tuple, list]): Specifies padding size.
        value (Union[int, float]): Specifies padding value.

    Inputs:
        - **input** (Tensor) - shape is :math:`(N, *)`, where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, the tensor after padding.

    Raises:
        TypeError: If `padding` is not an integer of a list or tuple of 4 integers.
        TypeError: If `input` is not Tensor.
        TypeError: If `value` is not int or float.
        ValueError: If `padding` contains negative value.
        ValueError: If `padding` is a tuple or list, and the length does not match the tensor dimension.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> x = np.ones(shape=(1, 2, 3, 4)).astype(np.float32)
        >>> x = ms.Tensor(x)
        >>> padding = (1, 1, 0, 1)
        >>> value = 0.5
        >>> pad2d = ms.mint.nn.ConstantPad2d(padding, value)
        >>> out = pad2d(x)
        >>> print(out)
        [[[[0.5  1.  1.  1.  1.  0.5]
           [0.5  1.  1.  1.  1.  0.5]
           [0.5  1.  1.  1.  1.  0.5]
           [0.5  0.5 0.5 0.5 0.5 0.5]]
          [[0.5  1.  1.  1.  1.  0.5]
           [0.5  1.  1.  1.  1.  0.5]
           [0.5  1.  1.  1.  1.  0.5]
           [0.5  0.5 0.5 0.5 0.5 0.5]]]]
        >>> print(out.shape)
        (1, 2, 4, 6)
    """

    def __init__(self, padding, value):
        super(ConstantPad2d, self).__init__(padding, value, padding_length=4)


class ConstantPad3d(ConstantPadNd_):
    """
    Pad the last 3 dimension of `input` tensor using `padding` and `value`.

    For more information, please refer to :func:`mindspore.mint.nn.functional.pad`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        padding (Union[int, tuple, list]): Specifies padding size.
        value (Union[int, float]): Specifies padding value.

    Inputs:
        - **input** (Tensor) - shape is :math:`(N, *)`, where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, the tensor after padding.

    Raises:
        TypeError: If `padding` is not an integer of a list or tuple of 6 integers.
        TypeError: If `input` is not Tensor.
        TypeError: If `value` is not int or float.
        ValueError: If `padding` contains negative value.
        ValueError: If `padding` is a tuple or list, and the length does not match the tensor dimension.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> x = np.ones(shape=(1, 2, 3, 4)).astype(np.float32)
        >>> x = ms.Tensor(x)
        >>> padding = (1, 1, 0, 1, 1, 0)
        >>> value = 0.5
        >>> pad3d = ms.mint.nn.ConstantPad3d(padding, value)
        >>> out = pad3d(x)
        >>> print(out)
        [[[[0.5 0.5 0.5 0.5 0.5 0.5]
           [0.5 0.5 0.5 0.5 0.5 0.5]
           [0.5 0.5 0.5 0.5 0.5 0.5]
           [0.5 0.5 0.5 0.5 0.5 0.5]]
          [[0.5 1.  1.  1.  1.  0.5]
           [0.5 1.  1.  1.  1.  0.5]
           [0.5 1.  1.  1.  1.  0.5]
           [0.5 0.5 0.5 0.5 0.5 0.5]]
          [[0.5 1.  1.  1.  1.  0.5]
           [0.5 1.  1.  1.  1.  0.5]
           [0.5 1.  1.  1.  1.  0.5]
           [0.5 0.5 0.5 0.5 0.5 0.5]]]]
        >>> print(out.shape)
        (1, 3, 4, 6)
    """

    def __init__(self, padding, value):
        super(ConstantPad3d, self).__init__(padding, value, padding_length=6)


class ZeroPadNd_(ConstantPadNd_):
    """
    Base class for N-dimensional zero padding.
    """
    def __init__(self, padding, padding_length):
        super(ZeroPadNd_, self).__init__(padding, value=0, padding_length=padding_length)


class ZeroPad1d(ZeroPadNd_):
    """
    Pad the last dimension of `input` tensor with 0 using `padding`.

    For more information, please refer to :func:`mindspore.mint.nn.functional.pad`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        padding (Union[int, tuple, list]): Specifies padding size.

    Inputs:
        - **input** (Tensor) - shape is :math:`(N, *)`, where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, the tensor after padding.

    Raises:
        ValueError: If `padding` contains negative value.
        ValueError: If `padding` is a tuple or list, and the length does not match the tensor dimension.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> x = mindspore.mint.ones((1, 2, 3, 4))
        >>> # padding is tuple
        >>> padding = (0, 1)
        >>> pad1d = mindspore.mint.nn.ZeroPad1d(padding)
        >>> out = pad1d(x)
        >>> out
        Tensor(shape=[1, 2, 3, 5], dtype=Float32, value=
        [[[[ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00]],
          [[ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00],
           [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00]]]])
        >>> # padding is int
        >>> padding = 1
        >>> pad1d = mindspore.mint.nn.ZeroPad1d(padding)
        >>> out = pad1d(x)
        >>> out
        Tensor(shape=[1, 2, 3, 6], dtype=Float32, value=
        [[[[ 0.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00],
           [ 0.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00],
           [ 0.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00]],
          [[ 0.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00],
           [ 0.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00],
           [ 0.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  0.00000000e+00]]]])
    """

    def __init__(self, padding):
        super(ZeroPad1d, self).__init__(padding, padding_length=2)


class ZeroPad2d(ZeroPadNd_):
    """
    Pad the last 2 dimension of `input` tensor with 0 using `padding`.

    For more information, please refer to :func:`mindspore.mint.nn.functional.pad`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        padding (Union[int, tuple, list]): Specifies padding size.

    Inputs:
        - **input** (Tensor) - shape is :math:`(N, *)`, where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, the tensor after padding.

    Raises:
        TypeError: If `padding` is not an integer of a list or tuple of 4 integers.
        TypeError: If `input` is not Tensor.
        ValueError: If `padding` contains negative value.
        ValueError: If `padding` is a tuple or list, and the length does not match the tensor dimension.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> x = np.ones(shape=(1, 2, 3, 4)).astype(np.float32)
        >>> x = ms.Tensor(x)
        >>> padding = (1, 1, 0, 1)
        >>> pad = ms.mint.nn.ZeroPad2d(padding)
        >>> out = pad(x)
        >>> print(out)
        [[[[0. 1. 1. 1. 1. 0.]
           [0. 1. 1. 1. 1. 0.]
           [0. 1. 1. 1. 1. 0.]
           [0. 0. 0. 0. 0. 0.]]
          [[0. 1. 1. 1. 1. 0.]
           [0. 1. 1. 1. 1. 0.]
           [0. 1. 1. 1. 1. 0.]
           [0. 0. 0. 0. 0. 0.]]]]
        >>> print(out.shape)
        (1, 2, 4, 6)
    """

    def __init__(self, padding):
        super(ZeroPad2d, self).__init__(padding, padding_length=4)


class ZeroPad3d(ZeroPadNd_):
    """
    Pad the last 3 dimension of `input` tensor with 0 using `padding`.

    For more information, please refer to :func:`mindspore.mint.nn.functional.pad`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        padding (Union[int, tuple, list]): Specifies padding size.

    Inputs:
        - **input** (Tensor) - shape is :math:`(N, *)`, where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, the tensor after padding.

    Raises:
        TypeError: If `padding` is not an integer of a list or tuple of 6 integers.
        TypeError: If `input` is not Tensor.
        ValueError: If `padding` contains negative value.
        ValueError: If `padding` is a tuple or list, and the length does not match the tensor dimension.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> x = np.ones(shape=(1, 2, 3, 4)).astype(np.float32)
        >>> x = ms.Tensor(x)
        >>> padding = (1, 1, 0, 1, 1, 0)
        >>> pad3d = ms.mint.nn.ZeroPad3d(padding)
        >>> out = pad3d(x)
        >>> print(out)
        [[[[0. 0. 0. 0. 0. 0.]
           [0. 0. 0. 0. 0. 0.]
           [0. 0. 0. 0. 0. 0.]
           [0. 0. 0. 0. 0. 0.]]
          [[0. 1.  1.  1.  1.  0.]
           [0. 1.  1.  1.  1.  0.]
           [0. 1.  1.  1.  1.  0.]
           [0. 0. 0. 0. 0. 0.]]
          [[0. 1.  1.  1.  1.  0.]
           [0. 1.  1.  1.  1.  0.]
           [0. 1.  1.  1.  1.  0.]
           [0. 0. 0. 0. 0. 0.]]]]
        >>> print(out.shape)
        (1, 3, 4, 6)
    """

    def __init__(self, padding):
        super(ZeroPad3d, self).__init__(padding, padding_length=6)


class ReflectionPadNd_(Cell):
    """
    Base class for N-dimensional reflection padding.
    """
    def __init__(self, padding, padding_length=None):
        super(ReflectionPadNd_, self).__init__()
        self.padding = padding

        if isinstance(self.padding, int):
            validator.check_positive_int(self.padding, "padding", self.cls_name)
            self.padding = (self.padding,) * padding_length
        elif isinstance(self.padding, (tuple, list)):
            if len(padding) != padding_length:
                msg = f"For '{self.cls_name}', the length of parameter 'padding' with tuple type must " \
                      f"equal to {padding_length}, but got {len(padding)}."
                raise ValueError(msg)
            validator.check_non_negative_int_sequence(self.padding, "padding", self.cls_name)
        else:
            msg = f"For '{self.cls_name}', 'padding' must be positive integer or tuple/list of {padding_length}" \
                  f" positive integers, but got {padding}."
            raise ValueError(msg)

    def construct(self, input):
        return mint.nn.functional.pad(input, self.padding, mode='reflect')


class ReflectionPad1d(ReflectionPadNd_):
    """
    Pad the last dimension of `input` tensor using the reflection of the input boundary.

    For more information, please refer to :func:`mindspore.mint.nn.functional.pad`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        padding (Union[int, tuple, list]): Specifies padding size.

    Inputs:
        - **input** (Tensor) - 2D or 3D input Tensor with shape: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.

    Outputs:
        Tensor, the tensor after padding.

    Raises:
        TypeError: If `padding` is not an integer of a list or tuple of 2 integers.
        TypeError: If `input` is not Tensor.
        ValueError: If `padding` contains negative value.
        ValueError: If `padding` is a tuple or list, and the length does not match the tensor dimension.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> x = ms.Tensor(np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]]).astype(np.float32))
        >>> # x has shape (1, 2, 4)
        >>> padding = (3, 1)
        >>> # The first and the second dimension of x remain the same.
        >>> # The third dimension of x: W_out = W_in + pad_left + pad_right = 4 + 3 + 1 = 8
        >>> pad1d = ms.mint.nn.ReflectionPad1d(padding)
        >>> out = pad1d(x)
        >>> # The shape of out is (1, 2, 8)
        >>> print(out)
        [[[3. 2. 1. 0. 1. 2. 3. 2.]
          [7. 6. 5. 4. 5. 6. 7. 6.]]]
    """

    def __init__(self, padding):
        super(ReflectionPad1d, self).__init__(padding, padding_length=2)


class ReflectionPad2d(ReflectionPadNd_):
    """
    Pad the last 2 dimension of `input` tensor using the reflection of the input boundary.

    For more information, please refer to :func:`mindspore.mint.nn.functional.pad`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        padding (Union[int, tuple, list]): Specifies padding size.

    Inputs:
        - **input** (Tensor) - 3D or 4D input Tensor with shape: :math:`(C, H_{in}, W_{in})`
          or :math:`(N, C, H_{in}, W_{in})`.

    Outputs:
        Tensor, the tensor after padding.

    Raises:
        TypeError: If `padding` is not an integer of a list or tuple of 4 integers.
        TypeError: If `input` is not Tensor.
        ValueError: If `padding` contains negative value.
        ValueError: If `padding` is a tuple or list, and the length does not match the tensor dimension.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> x = ms.Tensor(np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]).astype(np.float32))
        >>> # x has shape (1, 3, 3)
        >>> padding = (1, 1, 2, 0)
        >>> pad2d = ms.mint.nn.ReflectionPad2d(padding)
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
        super(ReflectionPad2d, self).__init__(padding, padding_length=4)


class ReflectionPad3d(ReflectionPadNd_):
    """
    Pad the last 3 dimension of `input` tensor using the reflection of the input boundary.

    For more information, please refer to :func:`mindspore.mint.nn.functional.pad`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        padding (Union[int, tuple, list]): Specifies padding size.

    Inputs:
        - **input** (Tensor) - 4D or 5D input Tensor with shape: :math:`(N, D_{in}, H_{in}, W_{in})`
          or :math:`(N, C, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, the tensor after padding.

    Raises:
        TypeError: If `padding` is not an integer of a list or tuple of 6 integers.
        TypeError: If `input` is not Tensor.
        ValueError: If `padding` contains negative value.
        ValueError: If `padding` is a tuple or list, and the length does not match the tensor dimension.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> arr = np.arange(8).astype(np.float32).reshape((1, 2, 2, 2))
        >>> x = ms.Tensor(arr)
        >>> # x has shape (1, 2, 2, 2)
        >>> padding = (1, 1, 1, 0, 0, 1)
        >>> pad3d = ms.mint.nn.ReflectionPad3d(padding)
        >>> out = pad3d(x)
        >>> # The first dimension of x remains the same.
        >>> # The second dimension of x: D_out = D_in + pad_front + pad_back = 2 + 0 + 1 = 3
        >>> # The third dimension of x: H_out = H_in + pad_up + pad_down = 2 + 1 + 0 = 3
        >>> # The last dimension of x: W_out = W_in + pad_left + pad_right = 2 + 1 + 1 = 4
        >>> # The shape of out is (1, 3, 3, 4)
        >>> print(out)
        [[[[3. 2. 3. 2.]
           [1. 0. 1. 0.]
           [3. 2. 3. 2.]]
          [[7. 6. 7. 6.]
           [5. 4. 5. 4.]
           [7. 6. 7. 6.]]
          [[3. 2. 3. 2.]
           [1. 0. 1. 0.]
           [3. 2. 3. 2.]]]]
    """

    def __init__(self, padding):
        super(ReflectionPad3d, self).__init__(padding, padding_length=6)


class ReplicationPadNd_(Cell):
    """
    Base class for N-dimensional replication padding.
    """
    def __init__(self, padding, padding_length=None):
        super(ReplicationPadNd_, self).__init__()
        self.padding = padding

        if isinstance(self.padding, int):
            validator.check_positive_int(self.padding, "padding", self.cls_name)
            self.padding = (self.padding,) * padding_length
        elif isinstance(self.padding, (tuple, list)):
            if len(padding) != padding_length:
                msg = f"For '{self.cls_name}', the length of parameter 'padding' with tuple type must " \
                      f"equal to {padding_length}, but got {len(padding)}."
                raise ValueError(msg)
            validator.check_non_negative_int_sequence(self.padding, "padding", self.cls_name)
        else:
            msg = f"For '{self.cls_name}', 'padding' must be positive integer or tuple/list of {padding_length} " \
                  f"positive integers, but got {padding}."
            raise ValueError(msg)

    def construct(self, input):
        return mint.nn.functional.pad(input, self.padding, mode='replicate')


class ReplicationPad1d(ReplicationPadNd_):
    """
    Pad the last dimension of `input` tensor using the replication of the input boundary.

    For more information, please refer to :func:`mindspore.mint.nn.functional.pad`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        padding (Union[int, tuple, list]): Specifies padding size.

    Inputs:
        - **input** (Tensor) - 2D or 3D input Tensor with shape: :math:`(C, W_{in})` or :math:`(N, C, W_{in})`.

    Outputs:
        The tensor after padding.

    Raises:
        TypeError: If `padding` is not an integer of a list or tuple of 2 integers.
        TypeError: If `input` is not Tensor.
        ValueError: If `padding` is a tuple or list, and the length does not match the tensor dimension.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> pad1d = ms.mint.nn.ReplicationPad1d(2)
        >>> input = ms.Tensor(np.arange(0, 8).reshape(1, 2, 4), ms.float32)
        >>> print(input)
        [[[0. 1. 2. 3.]
          [4. 5. 6. 7.]]]
        >>> out = pad1d(input)
        >>> print(out)
        [[[0. 0. 0. 1. 2. 3. 3. 3.]
          [4. 4. 4. 5. 6. 7. 7. 7.]]]
        >>> pad1d = ms.mint.nn.ReplicationPad1d((3, 1))
        >>> out = pad1d(input)
        >>> print(out)
        [[[0. 0. 0. 0. 1. 2. 3. 3.]
          [4. 4. 4. 4. 5. 6. 7. 7.]]]
    """

    def __init__(self, padding):
        super(ReplicationPad1d, self).__init__(padding, padding_length=2)


class ReplicationPad2d(ReplicationPadNd_):
    """
    Pad the last 2 dimension of `input` tensor using the replication of the input boundary.

    For more information, please refer to :func:`mindspore.mint.nn.functional.pad`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        padding (Union[int, tuple, list]): Specifies padding size.

    Inputs:
        - **input** (Tensor) - 3D or 4D input Tensor with shape: :math:`(C, H_{in}, W_{in})`
          or :math:`(N, C, H_{in}, W_{in})`.

    Outputs:
        Tensor, the tensor after padding.

    Raises:
        TypeError: If `padding` is not an integer of a list or tuple of 4 integers.
        TypeError: If `input` is not Tensor.
        ValueError: If `padding` is a tuple or list, and the length does not match the tensor dimension.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> pad2d = ms.mint.nn.ReplicationPad2d(2)
        >>> input = ms.Tensor(np.arange(0, 9).reshape(1, 1, 3, 3), ms.float32)
        >>> print(input)
        [[[[0. 1. 2.]
           [3. 4. 5.]
           [6. 7. 8.]]]]
        >>> out = pad2d(input)
        >>> print(out)
        [[[[0. 0. 0. 1. 2. 2. 2.]
           [0. 0. 0. 1. 2. 2. 2.]
           [0. 0. 0. 1. 2. 2. 2.]
           [3. 3. 3. 4. 5. 5. 5.]
           [6. 6. 6. 7. 8. 8. 8.]
           [6. 6. 6. 7. 8. 8. 8.]
           [6. 6. 6. 7. 8. 8. 8.]]]]
        >>> pad2d = ms.mint.nn.ReplicationPad2d((1, 1, 2, 0))
        >>> out = pad2d(input)
        >>> print(out)
        [[[[0. 0. 1. 2. 2.]
           [0. 0. 1. 2. 2.]
           [0. 0. 1. 2. 2.]
           [3. 3. 4. 5. 5.]
           [6. 6. 7. 8. 8.]]]]
    """

    def __init__(self, padding):
        super(ReplicationPad2d, self).__init__(padding, padding_length=4)


class ReplicationPad3d(ReplicationPadNd_):
    """
    Pad the last 3 dimension of `input` tensor using the replication of the input boundary.

    For more information, please refer to :func:`mindspore.mint.nn.functional.pad`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        padding (Union[int, tuple, list]): Specifies padding size.

    Inputs:
        - **input** (Tensor) - 4D or 5D input Tensor with shape: :math:`(N, D_{in}, H_{in}, W_{in})` or
          :math:`(N, C, D_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor, the tensor after padding.

    Raises:
        TypeError: If `padding` is not an integer of a list or tuple of 6 integers.
        TypeError: If `input` is not Tensor.
        ValueError: If `padding` is a tuple or list, and the length does not match the tensor dimension.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> pad3d = ms.mint.nn.ReplicationPad3d(1)
        >>> input = ms.Tensor(np.arange(0, 9).reshape(1, 1, 1, 3, 3), ms.float32)
        >>> out = pad3d(input)
        >>> print(out)
        [[[[[0. 0. 1. 2. 2.]
            [0. 0. 1. 2. 2.]
            [3. 3. 4. 5. 5.]
            [6. 6. 7. 8. 8.]
            [6. 6. 7. 8. 8.]]
           [[0. 0. 1. 2. 2.]
            [0. 0. 1. 2. 2.]
            [3. 3. 4. 5. 5.]
            [6. 6. 7. 8. 8.]
            [6. 6. 7. 8. 8.]]
           [[0. 0. 1. 2. 2.]
            [0. 0. 1. 2. 2.]
            [3. 3. 4. 5. 5.]
            [6. 6. 7. 8. 8.]
            [6. 6. 7. 8. 8.]]]]]
    """

    def __init__(self, padding):
        super(ReplicationPad3d, self).__init__(padding, padding_length=6)


__all__ = [
    'ConstantPad1d',
    'ConstantPad2d',
    'ConstantPad3d',
    'ZeroPad1d',
    'ZeroPad2d',
    'ZeroPad3d',
    'ReflectionPad1d',
    'ReflectionPad2d',
    'ReflectionPad3d',
    'ReplicationPad1d',
    'ReplicationPad2d',
    'ReplicationPad3d',
]

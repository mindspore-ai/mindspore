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
# ============================================================================
"""Hypercomplex Pooling"""
from typing import Tuple
from abc import abstractmethod
import mindspore.context as context
import mindspore.nn as nn
from mindspore._checkparam import Rel, Validator as validator
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.operations.nn_ops import AdaptiveAvgPool3D, AdaptiveMaxPool2D
from mindspore.hypercomplex.utils import get_x_and_y, to_2channel, \
                                _size_1_t, _size_2_t, _size_3_t


class _PoolNd(nn.Cell):
    r"""
    Base class for pooling layers for the second-order hypercomplex numbers.

    Includes data validation and initialization of hyperparameters, which are shared by all specific
    implementations of pooling.


    Note:
        pad_mode for training only supports "same" and "valid".

    Args:
        kernel_size (Union[int, Tuple[int]]): The size of kernel window used to take the average value.
            The data type of kernel_size must be int and the value represents all the spatial dimensions
            at once, or a tuple of the corresponding amount of int numbers that represent the spatial
            dimensions separately.
        stride (Union[int, Tuple[int]]): The distance of kernel moving, an int number that represents
            the step size of movement for all the spatial dimensions at once, or a tuple of the
            corresponding amount of int numbers that represent the step size of movement for spatial
            dimensions separately.
        pad_mode (str): The value for pad mode, is "same" or "valid", not case sensitive.

            - same: Adopts the way of completion. The height and width of the output will be the same as
              the input. The total number of padding will be calculated in horizontal and vertical
              directions and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possible largest height and width of output
              will be returned without padding. Extra pixels will be discarded.
        data_format (str): The optional value for data format, is 'NHWC' or 'NCHW'. Note that 'NCHW'
            format is supported only with GPU target device as of now. Default: 'NCHW'.

    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C, *, ..., *)`, with float16 or float32 data type, or
            :math:`(N, C, *, ..., *)` with complex64 data type. The count of spatial dimensions denoted by '*'
            depends on a specific subclass.

    Outputs:
        Tensor of the same data type as `inp` and of shape :math:`(2, N, C, *, ..., *)`, with float16 or float32 data
        type, or :math:`(N, C, *, ..., *)`, with complex64 data type. The count of spatial dimensions denoted by '*'
        is equal to one of the input tensor 'inp', but the sizes of those dimensions can be different.

    Raises:
        TypeError: If `kernel_size` or `stride` is not an int.
        TypeError: If dtype of `inp` is not float16, float32 or complex64.
        ValueError: If `pad_mode` is neither 'same' nor 'valid' (case insensitive).
        ValueError: If `kernel_size` or `stride` is less than 1.
        ValueError: If `data_format` is neither 'NCHW' nor 'NHWC', or it is 'NCHW' and the target
            device is not GPU.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 pad_mode: str,
                 data_format: str = "NCHW") -> None:
        """Initialize _PoolNd."""
        super(_PoolNd, self).__init__()
        validator.check_value_type('pad_mode', pad_mode, [str], self.cls_name)
        self.pad_mode = validator.check_string(pad_mode.upper(),
                                               ['VALID', 'SAME'],
                                               'pad_mode',
                                               self.cls_name)
        self.format = validator.check_string(data_format,
                                             ['NCHW', 'NHWC'],
                                             'format',
                                             self.cls_name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError(f"For '{self.cls_name}, the 'NHWC' format only support in GPU target, but got device "
                             f"target {context.get_context('device_target')}.")

        def _check_int_or_tuple(arg_name, arg_value):
            validator.check_value_type(arg_name, arg_value, [int, tuple], self.cls_name)
            error_msg = f'For \'{self.cls_name}\' the {arg_name} should be an positive int number or ' \
                        f'a tuple of two positive int numbers, but got {arg_value}'
            if isinstance(arg_value, int):
                if arg_value <= 0:
                    raise ValueError(error_msg)
            elif len(arg_value) == 2:
                for item in arg_value:
                    if isinstance(item, int) and item > 0:
                        continue
                    raise ValueError(error_msg)
            else:
                raise ValueError(error_msg)
            return arg_value

        self.kernel_size = _check_int_or_tuple('kernel_size', kernel_size)
        self.stride = _check_int_or_tuple('stride', stride)

    def construct(self, u: Tensor) -> Tensor:
        x, y = get_x_and_y(u)
        x, y = self._construct(x, y)
        out = to_2channel(x, y, u.dtype)
        return out

    def extend_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, pad_mode={pad_mode}'.format(**self.__dict__)

    @abstractmethod
    def _construct(self,
                   x: Tensor,
                   y: Tensor) -> Tuple[Tensor, Tensor]:
        pass


class MaxPool2d(_PoolNd):
    r"""
    2D max pooling operation for temporal hypercomplex data of the second order..

    Applies a 2D max pooling over an input Tensor which can be regarded as a composition of 2D planes.

    Typically the input is of shape :math:`(2, N, C, H_{in}, W_{in})`, MaxPool2d outputs
    regional maximum in the :math:`(H_{in}, W_{in})`- dimension. Given kernel size
    :math:`ks = (h_{ker}, w_{ker})` and stride :math:`s = (s_0, s_1)`, the operation is as follows:

    .. math::
        \text{out}(k, N_i, C_j, h, w) = \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{inp}(k, N_i, C_j, s_0 \times h + m, s_1 \times w + n),

    where :math:`\text{inp}` is a hypercomplex input tensor.

    Note:
        pad_mode for training only supports "same" and "valid".

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the max value,
            is an int number that represents height and width are both kernel_size,
            or a tuple of two int numbers that represent height and width respectively.
            Default: 1.
        stride (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the height and width of movement are both stride, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: 1.
        pad_mode (str): The optional value for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. The height and width of the output will be the same as
              the input. The total number of padding will be calculated in horizontal and vertical
              directions and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possible largest height and width of output
              will be returned without padding. Extra pixels will be discarded.
        data_format (str): The optional value for data format, is 'NHWC' or 'NCHW'. Note that 'NCHW'
            format is supported only with GPU target device as of now. Default: 'NCHW'.

    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C, H_{in}, W_{in})`, with float16 or float32 data type, or
        :math:`(N, C, H_{in}, W_{in})`, with complex64 data type.

    Outputs:
        Tensor of the same data type as `inp` and of shape :math:`(2, N, C, H_{out}, W_{out})`,
        with float16 or float32 data type, or :math:`(N, C, H_{out}, W_{out})`, with complex64 data type.

    Raises:
        TypeError: If `kernel_size` or `stride` is not an int.
        TypeError: If dtype of `inp` is not float16, float32 or complex64.
        ValueError: If `pad_mode` is neither 'same' nor 'valid' (case insensitive).
        ValueError: If `kernel_size` or `stride` is less than 1.
        ValueError: If `data_format` is neither 'NCHW' nor 'NHWC', or it is 'NCHW' and the target
            device is not GPU.
        ValueError: If length of shape of `inp` is not equal to 5

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.hypercomplex.hypercomplex.hc_pool import MaxPool2d
        >>> from mindspore import Tensor
        >>> u = Tensor(np.random.random((2, 8, 64, 32, 32)).astype(np.float32))
        >>> maxp = MaxPool2d(kernel_size=4, stride=4)
        >>> y = maxp(u)
        >>> print(y.shape)
        (2, 8, 64, 8, 8)
    """

    def __init__(self,
                 kernel_size: _size_2_t = 1,
                 stride: _size_2_t = 1,
                 pad_mode: str = "valid",
                 data_format: str = "NCHW") -> None:
        super(MaxPool2d, self).__init__(kernel_size, stride, pad_mode, data_format)
        self.max_pool = P.MaxPool(kernel_size=self.kernel_size,
                                  strides=self.stride,
                                  pad_mode=self.pad_mode,
                                  data_format=self.format)

    def _construct(self,
                   x: Tensor,
                   y: Tensor) -> Tuple[Tensor, Tensor]:
        out_x = self.max_pool(x)
        out_y = self.max_pool(y)
        return out_x, out_y


class MaxPool1d(_PoolNd):
    r"""
    1D max pooling operation for temporal hypercomplex data of the second order.

    Applies a 1D max pooling over an input Tensor which can be regarded as a composition of 1D planes.

    Typically the input is of shape :math:`(2, N, C, L_{in})`, MaxPool1d outputs
    regional maximum in the :math:`(L_{in})`-dimension. Given kernel size
    :math:`ks = (l_{ker})` and stride :math:`s = (s_0)`, the operation is as follows:

    .. math::
        \text{out}(k, N_i, C_j, l) = \max_{n=0, \ldots, l_{ker}-1}
        \text{inp}(k, N_i, C_j, s_0 \times l + n),

    where :math:`\text{inp}` is a hypercomplex input tensor.

    Note:
        pad_mode for training only supports "same" and "valid".

    Args:
        kernel_size (int): The size of kernel used to take the max value, Default: 1.
        stride (int): The distance of kernel moving, an int number that represents
            the width of movement is stride, Default: 1.
        pad_mode (str): The optional value for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. The total number of padding will be calculated in horizontal
              and vertical directions and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possible largest height and width of output
              will be returned without padding. Extra pixels will be discarded.

    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C, L_{in})`, with float16 or float32 data type, or
        :math:`(N, C, L_{in})`, with complex64 data type.

    Outputs:
        Tensor of the same data type as `inp` and of shape :math:`(2, N, C, L_{out})`, with float16 or float32
        data type, or :math:`(N, C, L_{out})`, with complex64 data type.

    Raises:
        TypeError: If `kernel_size` or `stride` is not an int.
        TypeError: If dtype of `inp` is not float16, float32 or complex64.
        ValueError: If `pad_mode` is neither 'same' nor 'valid' with not case sensitive.
        ValueError: If `kernel_size` or `strides` is less than 1.
        ValueError: If length of shape of `inp` is not equal to 4.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.hypercomplex.hypercomplex.hc_pool import MaxPool1d
        >>> from mindspore import Tensor
        >>> u = Tensor(np.random.random((2, 8, 64, 32)).astype(np.float32))
        >>> maxp = MaxPool1d(kernel_size=4, stride=4)
        >>> y = maxp(u)
        >>> print(y.shape)
        (2, 8, 64, 8)
    """

    def __init__(self,
                 kernel_size: _size_1_t = 1,
                 stride: _size_1_t = 1,
                 pad_mode: str = "valid") -> None:
        """Initialize MaxPool1d."""
        super(MaxPool1d, self).__init__(kernel_size, stride, pad_mode)
        validator.check_value_type('kernel_size', kernel_size, [int], self.cls_name)
        validator.check_value_type('stride', stride, [int], self.cls_name)
        validator.check_value_type('pad_mode', pad_mode, [str], self.cls_name)
        self.pad_mode = validator.check_string(pad_mode.upper(),
                                               ['VALID', 'SAME'],
                                               'pad_mode',
                                               self.cls_name)
        validator.check_int(kernel_size, 1, Rel.GE, "kernel_size", self.cls_name)
        validator.check_int(stride, 1, Rel.GE, "stride", self.cls_name)
        self.kernel_size = (1, kernel_size)
        self.stride = (1, stride)
        self.max_pool = P.MaxPool(kernel_size=self.kernel_size,
                                  strides=self.stride,
                                  pad_mode=self.pad_mode)
        self.shape = F.shape
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.expand = P.ExpandDims()
        self.squeeze = P.Squeeze(2)

    def _shape_check(self, in_shape: tuple):
        msg_prefix = f"For '{self.cls_name}', the" if self.cls_name else "The"
        if len(in_shape) != 3:
            raise ValueError(f"{msg_prefix} input must has 3 dim, but got {len(in_shape)}")
        return None

    def _construct(self,
                   x: Tensor,
                   y: Tensor) -> Tuple[Tensor, Tensor]:
        self._shape_check(self.shape(x))
        self._shape_check(self.shape(y))
        x = self.expand(x, 2)
        y = self.expand(y, 2)
        out_x = self.max_pool(x)
        out_y = self.max_pool(y)
        out_x = self.squeeze(out_x)
        out_y = self.squeeze(out_y)
        return out_x, out_y


class AvgPool2d(_PoolNd):
    r"""
    2D average pooling for temporal hypercomplex data of the second order.

    Applies a 2D average pooling over an input Tensor which can be regarded as a composition of 2D input planes.

    Typically the input is of shape :math:`(2, N, C, H_{in}, W_{in})`, AvgPool2d outputs
    regional average in the :math:`(H_{in}, W_{in})`-dimension. Given kernel size
    :math:`ks = (h_{ker}, w_{ker})` and stride :math:`s = (s_0, s_1)`, the operation is as follows:

    .. math::
        \text{out}(k, N_i, C_j, h, w) = \frac{1}{h_{ker} * w_{ker}} \sum_{m=0}^{h_{ker}-1} \sum_{n=0}^{w_{ker}-1}
        \text{inp}(k, N_i, C_j, s_0 \times h + m, s_1 \times w + n),

    where :math:`\text{inp}` is a hypercomplex input tensor.

    Note:
        pad_mode for training only supports "same" and "valid".

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the average value.
            The data type of kernel_size must be int and the value represents the height and width,
            or a tuple of two int numbers that represent height and width respectively.
            Default: 1.
        stride (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: 1.
        pad_mode (str): The optional value for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. The height and width of the output will be the same as
              the input. The total number of padding will be calculated in horizontal and vertical
              directions and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possible largest height and width of output
              will be returned without padding. Extra pixels will be discarded.
        data_format (str): The optional value for data format, is 'NHWC' or 'NCHW'. Note that 'NCHW'
            format is supported only with GPU target device as of now. Default: 'NCHW'.


    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C, H_{in}, W_{in})`, with float16 or float32 data type, or
        :math:`(N, C, H_{in}, W_{in})`, with complex64 data type.

    Outputs:
        Tensor of the same data type as `inp` and of shape :math:`(2, N, C, H_{out}, W_{out})`, with float16
        or float32 data type, or :math:`(N, C, H_{out}, W_{out})`, with complex64 data type.

    Raises:
        TypeError: If `kernel_size` or `stride` is not an int.
        TypeError: If dtype of `inp` is not float16, float32 or complex64.
        ValueError: If `pad_mode` is neither 'same' nor 'valid' (case insensitive).
        ValueError: If `kernel_size` or `stride` is less than 1.
        ValueError: If `data_format` is neither 'NCHW' nor 'NHWC', or it is 'NCHW' and the target
            device is not GPU.
        ValueError: If length of shape of `inp` is not equal to 5

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.hypercomplex.hypercomplex.hc_pool import AvgPool2d
        >>> from mindspore import Tensor
        >>> u = Tensor(np.random.random((2, 8, 64, 32, 32)).astype(np.float32))
        >>> avg = AvgPool2d(kernel_size=4, stride=4)
        >>> y = avg(u)
        >>> print(y.shape)
        (2, 8, 64, 8, 8)
    """

    def __init__(self,
                 kernel_size: _size_2_t = 1,
                 stride: _size_2_t = 1,
                 pad_mode: str = "valid",
                 data_format: str = "NCHW") -> None:
        super(AvgPool2d, self).__init__(kernel_size,
                                        stride,
                                        pad_mode,
                                        data_format)
        self.avg_pool = P.AvgPool(kernel_size=self.kernel_size,
                                  strides=self.stride,
                                  pad_mode=self.pad_mode,
                                  data_format=self.format)

    def _construct(self,
                   x: Tensor,
                   y: Tensor) -> Tuple[Tensor, Tensor]:
        out_x = self.avg_pool(x)
        out_y = self.avg_pool(y)
        return out_x, out_y


class AvgPool1d(_PoolNd):
    r"""
    1D average pooling for temporal hypercomplex data of the second order.

    Applies a 1D average pooling over an input Tensor which can be regarded as a composition of 1D input planes.

    Typically the input is of shape :math:`(2, N, C, L_{in})`, AvgPool1d outputs
    regional average in the :math:`(L_{in})`-dimension. Given kernel size
    :math:`ks = l_{ker}` and stride :math:`s = s_0`, the operation is as follows:

    .. math::
        \text{out}(k, N_i, C_j, l) = \frac{1}{l_{ker}} \sum_{n=0}^{l_{ker}-1}
        \text{inp}(k, N_i, C_j, s_0 \times l + n)

    where :math:`\text{inp}` is a hypercomplex input tensor.

    Note:
        pad_mode for training only supports "same" and "valid".

    Args:
        kernel_size (int): The size of kernel window used to take the average value, Default: 1.
        stride (int): The distance of kernel moving, an int number that represents
            the width of movement is strides, Default: 1.
        pad_mode (str): The optional value for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. The height and width of the output will be the same as
              the input. The total number of padding will be calculated in horizontal and vertical
              directions and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possible largest height and width of output
              will be returned without padding. Extra pixels will be discarded.


    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C, L_{in})`, with float16 or float32 data type, or
        :math:`(N, C, L_{in})`, with complex64 data type.

    Outputs:
        Tensor of the same data type as `inp` and of shape :math:`(2, N, C, L_{out})`, with float16 or float32
        data type, or :math:`(N, C, L_{out})`, with complex64 data type.

    Raises:
        TypeError: If `kernel_size` or `stride` is not an int.
        TypeError: If dtype of `inp` is not float16, float32 or complex64.
        ValueError: If `pad_mode` is neither 'same' nor 'valid' with not case sensitive.
        ValueError: If `kernel_size` or `strides` is less than 1.
        ValueError: If length of shape of `inp` is not equal to 4.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.hypercomplex.hypercomplex.hc_pool import AvgPool1d
        >>> from mindspore import Tensor
        >>> u = Tensor(np.random.random((2, 8, 64, 32)).astype(np.float32))
        >>> avg = AvgPool1d(kernel_size=4, stride=4)
        >>> y = avg(u)
        >>> print(y.shape)
        (2, 8, 64, 8)
    """

    def __init__(self,
                 kernel_size: _size_1_t = 1,
                 stride: _size_1_t = 1,
                 pad_mode: str = "valid") -> None:
        """Initialize AvgPool1d."""
        validator.check_value_type('kernel_size', kernel_size, [int], self.cls_name)
        validator.check_value_type('stride', stride, [int], self.cls_name)
        validator.check_value_type('pad_mode', pad_mode, [str], self.cls_name)
        self.pad_mode = validator.check_string(pad_mode.upper(),
                                               ['VALID', 'SAME'],
                                               'pad_mode',
                                               self.cls_name)
        validator.check_int(kernel_size, 1, Rel.GE, "kernel_size", self.cls_name)
        validator.check_int(stride, 1, Rel.GE, "stride", self.cls_name)
        super(AvgPool1d, self).__init__(kernel_size, stride, pad_mode)
        self.kernel_size = (1, kernel_size)
        self.stride = (1, stride)
        self.avg_pool = P.AvgPool(kernel_size=self.kernel_size,
                                  strides=self.stride,
                                  pad_mode=self.pad_mode)
        self.shape = F.shape
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.slice = P.Slice()
        self.expand = P.ExpandDims()
        self.squeeze = P.Squeeze(2)

    def _shape_check(self, in_shape: tuple):
        msg_prefix = f"For '{self.cls_name}', the" if self.cls_name else "The"
        if len(in_shape) != 3:
            raise ValueError(f"{msg_prefix} input must has 3 dim, but got {len(in_shape)}")
        return None

    def _construct(self,
                   x: Tensor,
                   y: Tensor) -> Tuple[Tensor, Tensor]:
        x = F.depend(x, self._shape_check(self.shape(x)))
        y = F.depend(y, self._shape_check(self.shape(y)))
        batch, channel, width = self.shape(x)
        if width == self.kernel_size[1]:
            x = self.reduce_mean(x, 2)
            y = self.reduce_mean(y, 2)
        elif width - self.kernel_size[1] < self.stride[1]:
            x = self.slice(x, (0, 0, 0), (batch, channel, self.kernel_size[1]))
            y = self.slice(y, (0, 0, 0), (batch, channel, self.kernel_size[1]))
            x = self.reduce_mean(x, 2)
            y = self.reduce_mean(y, 2)
        else:
            x = self.expand(x, 2)
            y = self.expand(y, 2)
            x = self.avg_pool(x)
            y = self.avg_pool(y)
            x = self.squeeze(x)
            y = self.squeeze(y)
        return x, y


class _AdaptivePoolNd(nn.Cell):
    r"""
    Base class for adaptive pooling layers for the second-order temporal hypercomplex data.

    Includes data validation and initialization of hyperparameters, which are shared by all specific
    implementations of adaptive pooling.

    Note:
        The size of every spatial dimension of `inp` must be divisible by the corresponding value of `output_size`.

    Args:
        output_size (Union[int, tuple]): The target output size. `ouput_size` can be a tuple of length being equal to
            the  count of spatial dimensions of the input tensor, or a single integer which then represents the desired
            output size for all of the spatial dimensions at once, or None.
            If it is None, it means the output size is the same as the input size.

    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C, *, ..., *)`, with float16 or float32 data type, or
            :math:`(N, C, *, ..., *)`, with complex64 data type. The count of spatial dimensions denoted by '*'
            depends on a specific subclass.

    Outputs:
        Tensor of the same data type as `inp`, and of shape :math:`(2, N, C, *, ..., *)`, with float16 or float32 data
        type, or :math:`(N, C, *, ..., *)`, with complex64 data type. The number of spatial dimensions denoted by '*'
        is the same as in `inp`.

    Raises:
        TypeError: If dtype of `inp` is not float16, float32 or complex64.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self, output_size: Tuple[int, ...]) -> None:
        super(_AdaptivePoolNd, self).__init__()
        self.output_size = output_size

    def construct(self, u: Tensor) -> Tensor:
        x, y = get_x_and_y(u)
        out_x, out_y = self._construct(x, y)
        out = to_2channel(out_x, out_y, u.dtype)

        return out

    @abstractmethod
    def _construct(self,
                   x: Tensor,
                   y: Tensor) -> Tuple[Tensor, Tensor]:
        pass

    def _adaptive_shape_check(self, in_shape):
        """Check shape."""
        msg_prefix = "For {}, the".format(self.cls_name)
        if len(in_shape) != 3:
            raise ValueError("{} input must has 3 dim, but got {}.".format(msg_prefix, len(in_shape)))
        if in_shape[2] < self.output_size:
            raise ValueError("{} input's last dimension must be greater or equal to "
                             "output size {}, but got {}.".format(msg_prefix, self.output_size, in_shape[2]))
        if in_shape[2] % self.output_size != 0:
            raise ValueError("{} input's last dimension must be divisible by "
                             "output size {}, but got {}.".format(msg_prefix, self.output_size, in_shape[2]))
        return None

    def _adaptive_dtype_check(self, x_dtype):
        """Check dtype."""
        if x_dtype not in [mstype.float16, mstype.float32]:
            raise TypeError("For {}, the x_dtype must be float16 or float32, "
                            "but got {}.".format(self.cls_name, x_dtype))
        return None


class AdaptiveAvgPool1d(_AdaptivePoolNd):
    r"""
    1D adaptive average pooling for temporal hypercomplex data of the second order.

    Applies a 1D adaptive average pooling over an input Tensor which can be regarded as
    a composition of 1D input planes.

    Typically, the input is of shape :math:`(2, N, C, L_{in})`,
    AdaptiveAvgPool1d outputs regional average in the :math:`L_{in}`-dimension.
    The output is of shape :math:`(2, N, C, L_{out})`,
    where :math:`L_{out}` is defined by `output_size`.

    Note:
        :math:`L_{in}` must be divisible by `output_size`.

    Args:
        output_size (int): the target output size :math:`L_{out}`.

    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C, L_{in})`, with float16 or float32 data type, or
        :math:`(N, C, L_{in})`, with complex64 data type.

    Outputs:
        Tensor of the same data type as `inp` and of shape :math:`(2, N, C, L_{out})`, with float16 or float32
        data type, or :math:`(N, C, L_{out})`, with complex64 data type.

    Raises:
        TypeError: If `output_size` is not an int.
        TypeError: If dtype of `inp` is not float16, float32 or complex64.
        ValueError: If `output_size` is less than 1.
        ValueError: If length of shape of `inp` is not equal to 4.
        ValueError: If the last dimension of `inp` is smaller than `output_size`.
        ValueError: If the last dimension of `inp` is not divisible by `output_size`.


    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.hypercomplex.hypercomplex.hc_pool import AdaptiveAvgPool1d
        >>> from mindspore import Tensor
        >>> u = Tensor(np.random.random((2, 8, 64, 32)).astype(np.float32))
        >>> avg = AdaptiveAvgPool1d(output_size=16)
        >>> y = avg(u)
        >>> print(y.shape)
        (2, 8, 64, 16)
    """

    def __init__(self, output_size: int) -> None:
        """Initialize AdaptiveAvgPool1d."""
        super(AdaptiveAvgPool1d, self).__init__(output_size)
        validator.check_value_type('output_size', output_size, [int], self.cls_name)
        validator.check_int(output_size, 1, Rel.GE, "output_size", self.cls_name)
        self.shape = F.shape
        self.expand = P.ExpandDims()
        self.squeeze = P.Squeeze(2)
        self.dtype = P.DType()

    def _construct(self,
                   x: Tensor,
                   y: Tensor) -> Tuple[Tensor, Tensor]:
        self._adaptive_shape_check(self.shape(x))
        self._adaptive_shape_check(self.shape(y))
        self._adaptive_dtype_check(self.dtype(x))
        self._adaptive_dtype_check(self.dtype(y))

        _, _, width = self.shape(x)
        stride = width // self.output_size
        kernel_size = width - (self.output_size - 1) * stride

        stride = (1, width // self.output_size)
        kernel_size = (1, kernel_size)

        x = self.expand(x, 2)
        y = self.expand(y, 2)

        avg_pool = P.AvgPool(kernel_size=kernel_size, strides=stride)

        out_x = avg_pool(x)
        out_y = avg_pool(y)
        out_x = self.squeeze(out_x)
        out_y = self.squeeze(out_y)

        return out_x, out_y


class AdaptiveAvgPool2d(_AdaptivePoolNd):
    r"""
    2D adaptive average pooling for temporal hypercomplex data of the second order.

    This operator applies a 2D adaptive average pooling to an input signal composed of multiple input planes.
    That is, for any input size, the size of the specified output is H x W.
    The number of output features is equal to the number of input features.

    The input and output data format can be "NCHW" and "CHW". N is the batch size, C is the number of channels,
    H is the feature height, and W is the feature width.

    .. math::
        \begin{align}
        h_{start} &= floor(i * H_{in} / H_{out})\\
        h_{end} &= ceil((i + 1) * H_{in} / H_{out})\\
        w_{start} &= floor(j * W_{in} / W_{out})\\
        w_{end} &= ceil((j + 1) * W_{in} / W_{out})\\
        out(i,j) &= \frac{\sum inp[h_{start}:h_{end}, w_{start}:w_{end}]}{(h_{end}- h_{start})
        * (w_{end}- w_{start})}
        \end{align}

    Args:
        output_size (Union[int, tuple]): The target output size is H x W.
            `ouput_size` can be a tuple consisted of int type H and W, or a single H for H x H, or None.
            If it is None, it means the output size is the same as the input size.

    Inputs:
        - **inp** (Tensor) - The input of AdaptiveAvgPool2d, which is a 4D or 5D tensor of shape
          :math:`(2, N, C, H_{in}, W_{in})` or :math:`(2, C, H_{in}, W_{in})`, with float16 or float32 data type,
          or :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`, with complex64 data type.

    Outputs:
        Tensor of the same data type as `inp` and of shape :math:`(2, N, C, H_{out}, W_{out})`, with float16
        or float32 data type, or :math:`(N, C, H_{out}, W_{out})`, with complex64 data type.

    Raises:
        ValueError: If `output_size` is a tuple and the length of `output_size` is not 2.
        TypeError: If `inp` is not a Tensor.
        TypeError: If dtype of `inp` is not float16, float32 or complex64.
        ValueError: If the dimension of `inp` is less than or equal to the dimension of `output_size`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.hypercomplex.hypercomplex.hc_pool import AdaptiveAvgPool2d
        >>> from mindspore import Tensor
        >>> u = Tensor(np.random.random((2, 8, 64, 32, 32)).astype(np.float32))
        >>> avg = AdaptiveAvgPool2d(output_size=16)
        >>> y = avg(u)
        >>> print(y.shape)
        (2, 8, 64, 16, 16)
    """

    def __init__(self, output_size: _size_2_t) -> None:
        """Initialize AdaptiveAvgPool2d."""
        super(AdaptiveAvgPool2d, self).__init__(output_size)
        self.adaptive_avgpool2d = P.AdaptiveAvgPool2D(output_size)

    def _construct(self,
                   x: Tensor,
                   y: Tensor) -> Tuple[Tensor, Tensor]:
        out_x = self.adaptive_avgpool2d(x)
        out_y = self.adaptive_avgpool2d(y)

        return out_x, out_y


class AdaptiveAvgPool3d(_AdaptivePoolNd):
    r"""
    3D adaptive average pooling for temporal hypercomplex data of the second order.

    This operator applies a 3D adaptive average pooling to an input signal composed of multiple input planes.
    That is, for any input size, the size of the specified output is :math:`(2, N, C, D, H, W)` or
    :math:`(2, C, D, H, W)`.
    The number of output features is equal to the number of input planes.

    Suppose the last 3 dimension size of x is :math:`(inD, inH, inW)`, then the last 3 dimension size of output is
    :math:`(outD, outH, outW)`.

    .. math::
        \begin{array}{ll} \\
            \forall \quad od \in [0,outD-1], oh \in [0,outH-1], ow \in [0,outW-1]\\
            output[od,oh,ow] = \\
            \qquad mean(x[istartD:iendD+1,istartH:iendH+1,istartW:iendW+1])\\
            where,\\
            \qquad istartD= \left\lceil \frac{od * inD}{outD} \right\rceil \\
            \qquad iendD=\left\lfloor \frac{(od+1)* inD}{outD} \right\rfloor \\
            \qquad istartH=\left\lceil \frac{oh * inH}{outH} \right\rceil \\
            \qquad iendH=\left\lfloor \frac{(oh+1) * inH}{outH} \right\rfloor \\
            \qquad istartW=\left\lceil \frac{ow * inW}{outW} \right\rceil \\
            \qquad iendW=\left\lfloor \frac{(ow+1) * inW}{outW} \right\rfloor
        \end{array}

    Args:
        output_size (Union[int, tuple]): The target output size. `output_size` can be a tuple :math:`(D, H, W)`,
            or an int D for :math:`(D, D, D)`. :math:`(D)`, :math:`(H)` and :math:`(W)` can be int or None
            which means the output size is the same as that of the input.

    Inputs:
        - **inp** (Tensor) - The input of AdaptiveAvgPool3d, which is a 6D Tensor
          :math:`(2, N, C, D_{in}, H_{in}, W_{in})` or a 5D Tensor :math:`(2, C, D_{in}, H_{in}, W_{in})`,
          with float16 or float32 data type, or 5D Tensor :math:`(N, C, D_{in}, H_{in}, W_{in})` or a 4D Tensor
          :math:`(C, D_{in}, H_{in}, W_{in})`, with complex64 data type.

    Outputs:
        Tensor of the same data type as `inp` and of shape :math:`(2, N, C, D_{out}, H_{out}, W_{out})`, with float16
        or float32 data type, or :math:`(N, C, D_{out}, H_{out}, W_{out})``, with complex64 data type.

    Raises:
        TypeError: If `inp` is not a Tensor.
        TypeError: If dtype of `inp` is not float16, float32 or complex64.
        ValueError: If the dimension of `inp` is not 5D or 6D.
        ValueError: If `output_size` value is not positive.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.hypercomplex.hypercomplex.hc_pool import AdaptiveAvgPool3d
        >>> from mindspore import Tensor
        >>> u = Tensor(np.random.random((2, 8, 64, 32, 48, 96)).astype(np.float32))
        >>> avg = AdaptiveAvgPool3d(output_size=(16, 24, 32))
        >>> y = avg(u)
        >>> print(y.shape)
        (2, 8, 64, 16, 24, 32)
    """

    def __init__(self, output_size: _size_3_t):
        """Initialize AdaptiveAvgPool3d."""
        super(AdaptiveAvgPool3d, self).__init__(output_size)
        self.adaptive_avg_pool3d = AdaptiveAvgPool3D(output_size)

    def _construct(self,
                   x: Tensor,
                   y: Tensor) -> Tuple[Tensor, Tensor]:
        out_x = self.adaptive_avg_pool3d(x)
        out_y = self.adaptive_avg_pool3d(y)

        return out_x, out_y


class AdaptiveMaxPool1d(_AdaptivePoolNd):
    r"""
    1D adaptive maximum pooling for temporal hypercomplex data of the second order.

    Applies a 1D adaptive maximum pooling over an input Tensor which can be regarded as
    a composition of 1D input planes.

    Typically, the input is of shape :math:`(2, N, C, L_{in})`,
    AdaptiveMaxPool1d outputs regional maximum in the :math:`L_{in}`-dimension. The output is of
    shape :math:`(N, C, L_{out})`, where :math:`L_{out}` is defined by `output_size`.

    Note:
        :math:`L_{in}` must be divisible by `output_size`.

    Args:
        output_size (int): the target output size :math:`L_{out}`.

    Inputs:
        - **inp** (Tensor) - Tensor of shape :math:`(2, N, C, L_{in})`, with float16 or float32 data type, or
        :math:`(N, C, L_{in})`, with complex64 data type.

    Outputs:
        Tensor of the same data type as `inp` and of shape :math:`(2, N, C, L_{out})`, with float16 or float32
        data type, or :math:`(N, C, L_{out})`, with complex64 data type.

    Raises:
        TypeError: If dtype of `inp` is not float16, float32 or complex64.
        TypeError: If `output_size` is not an int.
        ValueError: If `output_size` is less than 1.
        ValueError: If the last dimension of `inp` is smaller than `output_size`.
        ValueError: If the last dimension of `inp` is not divisible by `output_size`.
        ValueError: If length of shape of `inp` is not equal to 4.


    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.hypercomplex.hypercomplex.hc_pool import AdaptiveMaxPool1d
        >>> from mindspore import Tensor
        >>> u = Tensor(np.random.random((2, 8, 64, 32)).astype(np.float32))
        >>> maxp = AdaptiveMaxPool1d(output_size=16)
        >>> y = maxp(u)
        >>> print(y.shape)
        (2, 8, 64, 16)
    """

    def __init__(self, output_size: int) -> None:
        """Initialize AdaptiveMaxPool1d."""
        super(AdaptiveMaxPool1d, self).__init__(output_size)
        validator.check_value_type('output_size', output_size, [int], self.cls_name)
        validator.check_int(output_size, 1, Rel.GE, "output_size", self.cls_name)
        self.shape = F.shape
        self.expand = P.ExpandDims()
        self.squeeze = P.Squeeze(2)
        self.dtype = P.DType()

    def _construct(self,
                   x: Tensor,
                   y: Tensor) -> Tuple[Tensor, Tensor]:
        self._adaptive_shape_check(self.shape(x))
        self._adaptive_shape_check(self.shape(y))
        self._adaptive_dtype_check(self.dtype(x))
        self._adaptive_dtype_check(self.dtype(y))

        _, _, width = self.shape(x)
        stride = width // self.output_size
        kernel_size = width - (self.output_size - 1) * stride

        stride = (1, width // self.output_size)
        kernel_size = (1, kernel_size)

        x = self.expand(x, 2)
        y = self.expand(y, 2)

        max_pool = P.MaxPool(kernel_size=kernel_size, strides=stride)

        out_x = max_pool(x)
        out_y = max_pool(y)
        out_x = self.squeeze(out_x)
        out_y = self.squeeze(out_y)

        return out_x, out_y


class AdaptiveMaxPool2d(_AdaptivePoolNd):
    r"""
    AdaptiveMaxPool2d operation for temporal hypercomplex data of the second order.

    This operator applies a 2D adaptive max pooling to an input signal composed of multiple input planes.
    That is, for any input size, the size of the specified output is H x W.
    The number of output features is equal to the number of input planes.

    The input and output data format can be "NCHW" and "CHW". N is the batch size, C is the number of channels,
    H is the feature height, and W is the feature width.

    For max adaptive pool2d:

    .. math::

        \begin{align}
        h_{start} &= floor(i * H_{in} / H_{out})\\
        h_{end} &= ceil((i + 1) * H_{in} / H_{out})\\
        w_{start} &= floor(j * W_{in} / W_{out})\\
        w_{end} &= ceil((j + 1) * W_{in} / W_{out})\\
        out(i,j) &= {\max inp[h_{start}:h_{end}, w_{start}:w_{end}]}
        \end{align}

    Note:
        Ascend platform only supports float16 type for inp.

    Args:
        output_size (Union[int, tuple]): The target output size is H x W.
            ouput_size can be a tuple, or a single H for H x H, and H and W can be int or None
            which means the output size is the same as the input.

        return_indices (bool): If `return_indices` is True, the indices of max value would be output.
            Default: False.

    Inputs:
        - **inp** (Tensor) - The input of AdaptiveMaxPool2d, which is a 5D tensor of shape
          (2, N, C, H_{in}, W_{in}) or a 4D tensor of shape (2, C, H_{in}, W_{in}), with float16 or float32 data type,
          or a 4D tensor of shape (N, C, H_{in}, W_{in}) or a 3D tensor of shape (C, H_{in}, W_{in}),
          with complex64 data type.

    Outputs:
        Tensor of the same data type as `inp` and of shape :math:`(2, N, C, H_{out}, W_{out})` or
        :math:`(2, C, H_{out}, W_{out})`, with float16 or float32 data type, or :math:`(N, C, H_{out}, W_{out})` or
        :math:`(C, H_{out}, W_{out})`, with complex64 data type.

        Shape of the output is `inp_shape[:len(inp_shape) - len(out_shape)] + out_shape`.

    Raises:
        TypeError: If `output_size` is not int or tuple.
        TypeError: If `inp` is not a tensor.
        TypeError: If `return_indices` is not a bool.
        TypeError: If dtype of `inp` is not float16, float32 or complex64.
        ValueError: If `output_size` is a tuple and the length of `output_size` is not 2.
        ValueError: If the dimension of `inp` is not 4D or 5D

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.hypercomplex.hypercomplex.hc_pool import AdaptiveMaxPool2d
        >>> from mindspore import Tensor
        >>> u = Tensor(np.random.random((2, 8, 64, 32, 32)).astype(np.float32))
        >>> maxp = AdaptiveMaxPool2d(output_size=16)
        >>> y = maxp(u)
        >>> print(y.shape)
        (2, 8, 64, 16, 16)
    """

    def __init__(self, output_size: _size_2_t) -> None:
        """Initialize AdaptiveAvgPool2d."""
        super(AdaptiveMaxPool2d, self).__init__(output_size)
        self.adaptive_maxpool2d = AdaptiveMaxPool2D(output_size)

    def _construct(self,
                   x: Tensor,
                   y: Tensor) -> Tuple[Tensor, Tensor]:
        out_x = self.adaptive_maxpool2d(x)
        out_y = self.adaptive_maxpool2d(y)

        return out_x, out_y

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
"""pooling"""
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore._checkparam import Rel, Validator as validator
from mindspore.ops.primitive import constexpr
import mindspore.context as context
from ..cell import Cell

__all__ = ['AvgPool2d', 'MaxPool2d', 'AvgPool1d', 'MaxPool1d']

class _PoolNd(Cell):
    """N-D  AvgPool"""

    def __init__(self, kernel_size, stride, pad_mode, data_format="NCHW"):
        super(_PoolNd, self).__init__()
        self.pad_mode = validator.check_string(pad_mode.upper(), ['VALID', 'SAME'], 'pad_mode', self.cls_name)
        self.format = validator.check_string(data_format, ['NCHW', 'NHWC'], 'format', self.cls_name)
        if context.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError("NHWC format only support in GPU target.")
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

    def construct(self, *inputs):
        pass

    def extend_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, pad_mode={pad_mode}'.format(**self.__dict__)
@constexpr
def _shape_check(in_shape):
    if len(in_shape) != 3:
        raise ValueError("The input must has 3 dim")

class MaxPool2d(_PoolNd):
    r"""
    2D max pooling operation for temporal data.

    Applies a 2D max pooling over an input Tensor which can be regarded as a composition of 2D planes.

    Typically the input is of shape :math:`(N_{in}, C_{in}, H_{in}, W_{in})`, MaxPool2d outputs
    regional maximum in the :math:`(H_{in}, W_{in})`-dimension. Given kernel size
    :math:`ks = (h_{ker}, w_{ker})` and stride :math:`s = (s_0, s_1)`, the operation is as follows.

    .. math::
        \text{output}(N_i, C_j, h, w) = \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times h + m, s_1 \times w + n)

    Note:
        pad_mode for training only supports "same" and "valid".

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the max value,
            is an int number that represents height and width are both kernel_size,
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
        data_format (str): The optional value for data format, is 'NHWC' or 'NCHW'.
            Default: 'NCHW'.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `kernel_size` or `strides` is neither int nor tuple.
        ValueError: If `pad_mode` is neither 'valid' nor 'same' with not case sensitive.
        ValueError: If `data_format` is neither 'NCHW' nor 'NHWC'.
        ValueError: If `kernel_size` or `strides` is less than 1.
        ValueError: If length of shape of `input` is not equal to 4.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> pool = nn.MaxPool2d(kernel_size=3, stride=1)
        >>> x = Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), mindspore.float32)
        >>> output = pool(x)
        >>> print(output.shape)
        (1, 2, 2, 2)
    """

    def __init__(self, kernel_size=1, stride=1, pad_mode="valid", data_format="NCHW"):
        super(MaxPool2d, self).__init__(kernel_size, stride, pad_mode, data_format)
        self.max_pool = P.MaxPool(kernel_size=self.kernel_size,
                                  strides=self.stride,
                                  pad_mode=self.pad_mode,
                                  data_format=self.format)

    def construct(self, x):
        out = self.max_pool(x)
        return out


class MaxPool1d(_PoolNd):
    r"""
    1D max pooling operation for temporal data.

    Applies a 1D max pooling over an input Tensor which can be regarded as a composition of 1D planes.

    Typically the input is of shape :math:`(N_{in}, C_{in}, L_{in})`, MaxPool1d outputs
    regional maximum in the :math:`(L_{in})`-dimension. Given kernel size
    :math:`ks = (l_{ker})` and stride :math:`s = (s_0)`, the operation is as follows.

    .. math::
        \text{output}(N_i, C_j, l) = \max_{n=0, \ldots, l_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times l + n)

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
        - **input** (Tensor) - Tensor of shape :math:`(N, C, L_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C, L_{out}))`.

    Raises:
        TypeError: If `kernel_size` or `strides` is not an int.
        ValueError: If `pad_mode` is neither 'valid' nor 'same' with not case sensitive.
        ValueError: If `data_format` is neither 'NCHW' nor 'NHWC'.
        ValueError: If `kernel_size` or `strides` is less than 1.
        ValueError: If length of shape of `input` is not equal to 4.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> max_pool = nn.MaxPool1d(kernel_size=3, stride=1)
        >>> x = Tensor(np.random.randint(0, 10, [1, 2, 4]), mindspore.float32)
        >>> output = max_pool(x)
        >>> result = output.shape
        >>> print(result)
        (1, 2, 2)
    """

    def __init__(self, kernel_size=1, stride=1, pad_mode="valid"):
        super(MaxPool1d, self).__init__(kernel_size, stride, pad_mode)
        validator.check_value_type('kernel_size', kernel_size, [int], self.cls_name)
        validator.check_value_type('stride', stride, [int], self.cls_name)
        self.pad_mode = validator.check_string(pad_mode.upper(), ['VALID', 'SAME'], 'pad_mode', self.cls_name)
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

    def construct(self, x):
        _shape_check(self.shape(x))
        x = self.expand(x, 2)
        output = self.max_pool(x)
        output = self.squeeze(output)
        return output


class AvgPool2d(_PoolNd):
    r"""
    2D average pooling for temporal data.

    Applies a 2D average pooling over an input Tensor which can be regarded as a composition of 2D input planes.

    Typically the input is of shape :math:`(N_{in}, C_{in}, H_{in}, W_{in})`, AvgPool2d outputs
    regional average in the :math:`(H_{in}, W_{in})`-dimension. Given kernel size
    :math:`ks = (h_{ker}, w_{ker})` and stride :math:`s = (s_0, s_1)`, the operation is as follows.

    .. math::
        \text{output}(N_i, C_j, h, w) = \frac{1}{h_{ker} * w_{ker}} \sum_{m=0}^{h_{ker}-1} \sum_{n=0}^{w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times h + m, s_1 \times w + n)

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
        data_format (str): The optional value for data format, is 'NHWC' or 'NCHW'.
            Default: 'NCHW'.


    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `kernel_size` or `strides` is neither int nor tuple.
        ValueError: If `pad_mode` is neither 'valid' nor 'same' with not case sensitive.
        ValueError: If `data_format` is neither 'NCHW' nor 'NHWC'.
        ValueError: If `kernel_size` or `strides` is less than 1.
        ValueError: If length of shape of `input` is not equal to 4.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> pool = nn.AvgPool2d(kernel_size=3, stride=1)
        >>> x = Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), mindspore.float32)
        >>> output = pool(x)
        >>> print(output.shape)
        (1, 2, 2, 2)
    """

    def __init__(self,
                 kernel_size=1,
                 stride=1,
                 pad_mode="valid",
                 data_format="NCHW"):
        super(AvgPool2d, self).__init__(kernel_size, stride, pad_mode, data_format)
        self.avg_pool = P.AvgPool(kernel_size=self.kernel_size,
                                  strides=self.stride,
                                  pad_mode=self.pad_mode,
                                  data_format=self.format)

    def construct(self, x):
        return self.avg_pool(x)


class AvgPool1d(_PoolNd):
    r"""
    1D average pooling for temporal data.

    Applies a 1D average pooling over an input Tensor which can be regarded as a composition of 1D input planes.

    Typically the input is of shape :math:`(N_{in}, C_{in}, L_{in})`, AvgPool1d outputs
    regional average in the :math:`(L_{in})`-dimension. Given kernel size
    :math:`ks = l_{ker}` and stride :math:`s = s_0`, the operation is as follows.

    .. math::
        \text{output}(N_i, C_j, l) = \frac{1}{l_{ker}} \sum_{n=0}^{l_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times l + n)

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
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, L_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, L_{out})`.

    Raises:
        TypeError: If `kernel_size` or `stride` is not an int.
        ValueError: If `pad_mode` is neither 'same' nor 'valid' with not case sensitive.
        ValueError: If `kernel_size` or `strides` is less than 1.
        ValueError: If length of shape of `input` is not equal to 3.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> pool = nn.AvgPool1d(kernel_size=6, stride=1)
        >>> x = Tensor(np.random.randint(0, 10, [1, 3, 6]), mindspore.float32)
        >>> output = pool(x)
        >>> result = output.shape
        >>> print(result)
        (1, 3, 1)
    """

    def __init__(self,
                 kernel_size=1,
                 stride=1,
                 pad_mode="valid"):
        validator.check_value_type('kernel_size', kernel_size, [int], self.cls_name)
        validator.check_value_type('stride', stride, [int], self.cls_name)
        self.pad_mode = validator.check_string(pad_mode.upper(), ['VALID', 'SAME'], 'pad_mode', self.cls_name)
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

    def construct(self, x):
        x = F.depend(x, _shape_check(self.shape(x)))
        batch, channel, width = self.shape(x)
        if width == self.kernel_size[1]:
            x = self.reduce_mean(x, 2)
        elif width - self.kernel_size[1] < self.stride[1]:
            x = self.slice(x, (0, 0, 0), (batch, channel, self.kernel_size[1]))
            x = self.reduce_mean(x, 2)
        else:
            x = self.expand(x, 2)
            x = self.avg_pool(x)
            x = self.squeeze(x)
        return x

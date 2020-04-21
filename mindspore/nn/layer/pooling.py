# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore._checkparam import Validator as validator
from ... import context
from ..cell import Cell


class _PoolNd(Cell):
    """N-D  AvgPool"""

    def __init__(self, kernel_size, stride, pad_mode):
        super(_PoolNd, self).__init__()
        self.pad_mode = validator.check_string('pad_mode', pad_mode.upper(), ['VALID', 'SAME'], self.cls_name)

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


class MaxPool2d(_PoolNd):
    r"""
    Max pooling operation for temporal data.

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
        pad_mode (str): The optional values for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. Output height and width will be the same as
              the input. Total number of padding will be calculated for horizontal and vertical
              direction and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possibly largest height and width of output
              will be return without padding. Extra pixels will be discarded.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> pool = nn.MaxPool2d(kernel_size=3, stride=1)
        >>> x = Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), mindspore.float32)
        [[[[1. 5. 5. 1.]
           [0. 3. 4. 8.]
           [4. 2. 7. 6.]
           [4. 9. 0. 1.]]
          [[3. 6. 2. 6.]
           [4. 4. 7. 8.]
           [0. 0. 4. 0.]
           [1. 8. 7. 0.]]]]
        >>> output = pool(x)
        >>> output.shape()
        (1, 2, 2, 2)
        >>> output
        [[[[7. 8.]
           [9. 9.]]
          [[7. 8.]
           [8. 8.]]]]
    """

    def __init__(self, kernel_size=1, stride=1, pad_mode="valid"):
        super(MaxPool2d, self).__init__(kernel_size, stride, pad_mode)
        self.max_pool = P.MaxPool(ksize=self.kernel_size,
                                  strides=self.stride,
                                  padding=self.pad_mode)
        self.max_pool_with_arg_max = P.MaxPoolWithArgmax(ksize=self.kernel_size,
                                                         strides=self.stride,
                                                         padding=self.pad_mode)
        self.is_tbe = context.get_context("device_target") == "Ascend"

    def construct(self, x):
        if self.is_tbe and self.training:
            out = self.max_pool_with_arg_max(x)[0]
        else:
            out = self.max_pool(x)
        return out


class AvgPool2d(_PoolNd):
    r"""
    Average pooling for temporal data.

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
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the average value,
            is an int number that represents height and width are both kernel_size,
            or a tuple of two int numbers that represent height and width respectively.
            Default: 1.
        stride (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: 1.
        pad_mode (str): The optional values for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. Output height and width will be the same as
              the input. Total number of padding will be calculated for horizontal and vertical
              direction and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possibly largest height and width of output
              will be return without padding. Extra pixels will be discarded.


    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> pool = nn.AvgPool2d(kernel_size=3, strides=1)
        >>> x = Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), mindspore.float32)
        [[[[5. 5. 9. 9.]
            [8. 4. 3. 0.]
            [2. 7. 1. 2.]
            [1. 8. 3. 3.]]
           [[6. 8. 2. 4.]
            [3. 0. 2. 1.]
            [0. 8. 9. 7.]
            [2. 1. 4. 9.]]]]
        >>> output = pool(x)
        >>> output.shape()
        (1, 2, 2, 2)
        >>> output
        [[[[4.888889  4.4444447]
           [4.111111  3.4444444]]
          [[4.2222223 4.5555553]
           [3.2222223 4.5555553]]]]
    """

    def __init__(self,
                 kernel_size=1,
                 stride=1,
                 pad_mode="valid"):
        super(AvgPool2d, self).__init__(kernel_size, stride, pad_mode)
        self.avg_pool = P.AvgPool(ksize=self.kernel_size,
                                  strides=self.stride,
                                  padding=self.pad_mode)

    def construct(self, x):
        return self.avg_pool(x)


class AvgPool1d(_PoolNd):
    r"""
    Average pooling for temporal data.

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
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the average value,
            is an int number that represents height and width are both kernel_size,
            or a tuple of two int numbers that represent height and width respectively.
            Default: 1.
        stride (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: 1.
        pad_mode (str): The optional values for pad mode, is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Adopts the way of completion. Output height and width will be the same as
              the input. Total number of padding will be calculated for horizontal and vertical
              direction and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possibly largest height and width of output
              will be return without padding. Extra pixels will be discarded.


    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Examples:
        >>> pool = nn.AvgPool2d(kernel_size=3, strides=1)
        >>> x = Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), mindspore.float32)
        [[[[5. 5. 9. 9.]
            [8. 4. 3. 0.]
            [2. 7. 1. 2.]
            [1. 8. 3. 3.]]
           [[6. 8. 2. 4.]
            [3. 0. 2. 1.]
            [0. 8. 9. 7.]
            [2. 1. 4. 9.]]]]
        >>> output = pool(x)
        >>> output.shape()
        (1, 2, 2, 2)
        >>> output
        [[[[4.888889  4.4444447]
           [4.111111  3.4444444]]
          [[4.2222223 4.5555553]
           [3.2222223 4.5555553]]]]
    """

    def __init__(self,
                 kernel_size=1,
                 stride=1,
                 pad_mode="valid"):
        super(AvgPool1d, self).__init__(kernel_size, stride, pad_mode)
        if not isinstance(kernel_size, int):
            raise ValueError("kernel_size should be 1 int number but got {}".
                             format(kernel_size))
        if not isinstance(stride, int):
            raise ValueError("stride should be 1 int number but got {}".format(stride))
        self.kernel_size = (1, kernel_size)
        self.stride = (1, stride)
        self.avg_pool = P.AvgPool(ksize=self.kernel_size,
                                  strides=self.stride,
                                  padding=self.pad_mode)

    def construct(self, x):
        return self.avg_pool(x)

#Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from __future__ import absolute_import

from mindspore.ops.auto_generate.gen_ops_prim import MaxPoolWithIndices, MaxPoolWithMask
from mindspore.nn.cell import Cell

__all__ = ['MaxPool2d']


class MaxPool2d(Cell):
    r"""
    Applies a 2D max pooling over an input Tensor which can be regarded as a composition of 2D planes.

    Typically the input is of shape :math:`(N_{in}, C_{in}, H_{in}, W_{in})`, MaxPool2d outputs
    regional maximum in the :math:`(H_{in}, W_{in})`-dimension. Given kernel size
    :math:`(h_{ker}, w_{ker})` and stride :math:`(s_0, s_1)`, the operation is as follows.

    .. math::
        \text{output}(N_i, C_j, h, w) = \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times h + m, s_1 \times w + n)

    .. warning::
        Only support on Atlas training series.

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the max value,
            is an int number or a single element tuple that represents height and width are both kernel_size,
            or a tuple of two int numbers that represent height and width respectively.
            Default: ``1`` .
        stride (Union[int, tuple[int], None]): The distance of kernel moving, an int number or a single element tuple
            that represents the height and width of movement are both stride, or a tuple of two int numbers that
            represent height and width of movement respectively.
            Default: ``None`` , which indicates the moving step is `kernel_size` .
        padding (Union(int, tuple[int], list[int])): Specifies the padding value of the pooling operation.
            Default: ``0`` . `padding` can only be an integer or a tuple/list containing one or two integers. If
            `padding` is an integer or a tuple/list containing one integer, it will be padded `padding` times in the
            four directions of the input. If `padding` is a tuple/list containing two integers, it will be padded
            `padding[0]` times in the up-down direction of the input and `padding[1]` times in the left-right direction
            of the input.
        dilation (Union(int, tuple[int])): The spacing between the elements of the kernel in convolution,
            used to increase the receptive field of the pooling operation. If it is a tuple, it must contain one or two
            integers. Default: ``1`` .
        return_indices (bool): If ``True`` , the function will return both the result of max pooling and the indices of
            the max elements. Default: ``False`` .
        ceil_mode (bool): If ``True`` , use ceil to compute the output shape instead of floor. Default: ``False`` .

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        If `return_indices` is ``False`` , return a Tensor `output`, else return a tuple (`output`, `argmax`).

        - **output** (Tensor) - Maxpooling result, with shape :math:`(N_{out}, C_{out}, H_{out}, W_{out})`. It has the
          same data type as `input`.
        - **argmax** (Tensor) - Index corresponding to the maximum value. Data type is int32.

        .. math::
            H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                \times (\text{kernel_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

        .. math::
            W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                \times (\text{kernel_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

    Raises:
        TypeError: If `input` is not a Tensor.
        ValueError: If length of shape of `input` is not equal to 4.
        TypeError: If `kernel_size` , `stride` , `padding` or `dilation` is not int or tuple.
        ValueError: If `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `dilation` is not all 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `padding` is more than half of `kernel_size`.
        TypeError: If `ceil_mode` is not bool.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> pool = ms.nn.extend.MaxPool2d(kernel_size=3, stride=1)
        >>> input = ms.Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), ms.float32)
        >>> output = pool(input)
        >>> print(output.shape)
        (1, 2, 2, 2)
    """

    def __init__(self, kernel_size=1, stride=None, padding=0, dilation=1, return_indices=False,
                 ceil_mode=False):
        """Initialize MaxPool2d."""
        super(MaxPool2d, self).__init__()
        self.return_indices = return_indices
        strides = stride if (stride is not None) else kernel_size
        if return_indices:
            self.max_pool_func_ = MaxPoolWithIndices(kernel_size, strides, padding, dilation, ceil_mode)
        else:
            self.max_pool_func_ = MaxPoolWithMask(kernel_size, strides, padding, dilation, ceil_mode)

    def construct(self, input):
        out, indices = self.max_pool_func_(input)
        if self.return_indices:
            return out, indices
        return out

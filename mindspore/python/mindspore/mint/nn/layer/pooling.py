# Copyright 2020-2025 Huawei Technologies Co., Ltd
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
"""normalization for mint"""
from __future__ import absolute_import
from __future__ import division

from mindspore import mint
from mindspore.nn.cell import Cell


class _AdaptiveAvgPoolNd(Cell):
    """Common base of AdaptiveAvgPoolNd"""

    def __init__(self, output_size) -> None:
        super().__init__()
        self.output_size = output_size

    def extend_repr(self):
        return 'output_size={}'.format(self.output_size)


class AdaptiveAvgPool1d(_AdaptiveAvgPoolNd):
    r"""
    Apply a 1-D adaptive average pooling over an input signal composed of several input planes.

    The output is of size :math:`L_{out}`, for any input size.
    The number of output features is equal to the number of input planes.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        output_size (int): The target output size :math:`L_{out}`.

    Inputs:
        - **input** (Tensor) - The input tensor with shape :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.

    Outputs:
        Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[[2, 1, 2], [2, 3, 5]]], mindspore.float16)
        >>> net = mindspore.mint.nn.AdaptiveAvgPool1d(3)
        >>> output = net(input)
        >>> print(output)
        [[[2. 1. 2.]
          [2. 3. 5.]]]
    """

    def construct(self, input):
        return mint.nn.functional.adaptive_avg_pool1d(input, self.output_size)


class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):
    r"""
    Apply a 2-D adaptive average pooling over an input signal composed of several input planes.

    The output is of size :math:`H \times W`, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size (Union[int, tuple[int]]): The target output size of the image of the form :math:`H \times W`.
            Can be a tuple :math:`(H, W)` or a single :math:`H` for square image :math:`H \times H`.
            :math:`H` and :math:`W` can be either an ``int``, or ``None`` which means the size will
            be the same as that of the input.

    Inputs:
        - **input** (Tensor) - The input tensor with shape :math:`(N, C, H, W)` or :math:`(C, H, W)`.

    Outputs:
        Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[[2, 1, 2], [2, 3, 5]]], mindspore.float16)
        >>> net = mindspore.mint.nn.AdaptiveAvgPool2d((2, 2))
        >>> output = net(input)
        >>> print(output)
        [[[1.5 1.5]
          [2.5 4. ]]]
    """

    def construct(self, input):
        return mint.nn.functional.adaptive_avg_pool2d(input, self.output_size)


class AdaptiveAvgPool3d(Cell):
    r"""
    This operator applies a 3D adaptive average pooling to an input signal composed of multiple input planes.
    That is, for any input size, the size of the specified output is :math:`(D, H, W)`.
    The number of output features is equal to the number of input planes.

    Suppose the last 3 dimension size of input is :math:`(inD, inH, inW)`, then the last 3 dimension size of output is
    :math:`(outD, outH, outW)`.

    .. math::
        \begin{array}{ll} \\
            \forall \quad od \in [0,outD-1], oh \in [0,outH-1], ow \in [0,outW-1]\\
            output[od,oh,ow] = \\
            \qquad mean(input[istartD:iendD+1,istartH:iendH+1,istartW:iendW+1])\\
            where,\\
            \qquad istartD= \left\lceil \frac{od * inD}{outD} \right\rceil \\
            \qquad iendD=\left\lfloor \frac{(od+1)* inD}{outD} \right\rfloor \\
            \qquad istartH=\left\lceil \frac{oh * inH}{outH} \right\rceil \\
            \qquad iendH=\left\lfloor \frac{(oh+1) * inH}{outH} \right\rfloor \\
            \qquad istartW=\left\lceil \frac{ow * inW}{outW} \right\rceil \\
            \qquad iendW=\left\lfloor \frac{(ow+1) * inW}{outW} \right\rfloor
        \end{array}

    .. warning::
        For Ascend, it is only supported on Atlas A2 Training Series Products.

    Args:
        output_size (Union[int, tuple]): The target output size. `output_size` can be a tuple :math:`(D, H, W)`,
            or an int D for :math:`(D, D, D)`. :math:`D`, :math:`H` and :math:`W` can be int or None
            which means the output size is the same as that of the input.

    Inputs:
        - **input** (Tensor) - The input of AdaptiveAvgPool3d, which is a 5D or 4D Tensor.

    Outputs:
        Tensor, with the same type as the `input`.

    Raises:
        TypeError: If `input` is not a Tensor.
        ValueError: If the dimension of `input` is not 4D or 5D.
        ValueError: If `output_size` value is not positive.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import mint
        >>> import numpy as np
        >>> # case 1: output_size=(3, 3, 4)
        >>> output_size=(3, 3, 4)
        >>> input_x_val = np.random.randn(4, 3, 5, 6, 7)
        >>> input_x = ms.Tensor(input_x_val, ms.float32)
        >>> net = mint.nn.AdaptiveAvgPool3d(output_size)
        >>> output = net(input_x)
        >>> print(output.shape)
        (4, 3, 3, 3, 4)
        >>> # case 2: output_size=4
        >>> output_size=5
        >>> input_x_val = np.random.randn(2, 3, 8, 6, 12)
        >>> input_x = ms.Tensor(input_x_val, ms.float32)
        >>> net = mint.nn.AdaptiveAvgPool3d(output_size)
        >>> output = net(input_x)
        >>> print(output.shape)
        (2, 3, 5, 5, 5)
        >>> # case 3: output_size=(None, 4, 5)
        >>> output_size=(None, 4, 5)
        >>> input_x_val = np.random.randn(4, 1, 9, 10, 8)
        >>> input_x = ms.Tensor(input_x_val, ms.float32)
        >>> net = mint.nn.AdaptiveAvgPool3d(output_size)
        >>> output = net(input_x)
        >>> print(output.shape)
        (4, 1, 9, 4, 5)
    """

    def __init__(self, output_size):
        """Initialize AdaptiveAvgPool3d."""
        super().__init__()
        self.output_size = output_size

    def construct(self, input):
        return mint.nn.functional.adaptive_avg_pool3d(input, self.output_size)


class MaxUnpool2d(Cell):
    r"""
    Computes the inverse of `Maxpool2d`.

    `MaxUnpool2d` keeps the maximal value and set all position of non-maximal values to zero.
    Typically the input is of shape :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`,
    and the output is of shape :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`.
    The operation is as follows.

    .. math::
        \begin{array}{ll} \\
        H_{out} = (H_{in} - 1) \times stride[0] - 2 \times padding[0] + kernel\_size[0] \\
        W_{out} = (W_{in} - 1) \times stride[1] - 2 \times padding[1] + kernel\_size[1] \\
        \end{array}

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the maximum value,
            an int number that represents height and width of the kernel,
            or a tuple of two int numbers that represent height and width respectively.
        stride (Union[int, tuple[int]], optional): The distance of kernel moving,
            an int number that represents the height and width of movement are both stride,
            or a tuple of two int numbers that represent height and width of movement respectively.
            Default: ``None`` , which indicates the moving step is `kernel_size` .
        padding (Union[int, tuple[int]], optional): The pad value to be filled. Default: ``0`` .
            If `padding` is an integer, the paddings of height and width are the same, equal to padding.
            If `padding` is a tuple of two integers, the padding of height and width equal to padding[0]
            and padding[1] correspondingly.

    Inputs:
        - **input** (Tensor) - The input Tensor to invert.
          Tensor of shape :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - **indices** (Tensor) - Max values' index represented by the indices.
          Tensor of shape must be same with input 'input'.
          Values of indices must belong to :math:`[0, H_{in} \times W_{in} - 1]`.
          Data type must be in int32 or int64.
        - **output_size** (tuple[int], optional) - The target output size. Default: ``None`` .
          If output_size == (), then the shape of output computed by `kernel_size`, `stride` and `padding`.
          If output_size != (), then output_size must be :math:`(N, C, H, W)` , :math:`(C, H, W)` or :math:`(H, W)`
          and output_size must belong to
          :math:`[(N, C, H_{out} - stride[0], W_{out} - stride[1]), (N, C, H_{out} + stride[0], W_{out} + stride[1])]`.

    Outputs:
        Tensor, with shape :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`,
        with the same data type with `input`.

    Raises:
        TypeError: If data type of `input` or `indices` is not supported.
        TypeError: If `kernel_size`, `stride` or `padding` is neither an int nor a tuple.
        ValueError: If numbers in `stride`, `padding` or `kernel_size` is not positive.
        ValueError: If the shapes of `input` and `indices` are not equal.
        ValueError: If `input` whose length is not 3 or 4.
        ValueError: If `output_size` whose type is not tuple.
        ValueError: If `output_size` is not close to output size computed by attr `kernel_size`, `stride`, `padding`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> input = Tensor(np.array([[[[0, 1], [8, 9]]]]).astype(np.float32))
        >>> indices = Tensor(np.array([[[[0, 1], [2, 3]]]]).astype(np.int64))
        >>> net =  mint.nn.MaxUnpool2d(1, stride=1, padding=0)
        >>> output = net(input, indices)
        >>> print(output.asnumpy())
        [[[[0. 1.]
           [8. 9.]]]]
    """

    def __init__(self, kernel_size, stride=None, padding=0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def construct(self, input, indices, output_size=None):
        return mint.nn.functional.max_unpool2d(input, indices,
                                               self.kernel_size, self.stride,
                                               self.padding, output_size)


class _AdaptiveMaxPoolNd(Cell):
    """Common base of AdaptiveMaxPool1d"""

    def __init__(self, output_size, return_indices=False) -> None:
        super().__init__()
        self.output_size = output_size
        self.return_indices = return_indices

    def extend_repr(self):
        return 'output_size={}, return_indices={}'.format(self.output_size, self.return_indices)


class AdaptiveMaxPool1d(_AdaptiveMaxPoolNd):
    r"""
    Applies a 1D adaptive max pooling over an input signal composed of several input planes.

    The output is of size :math:`L_{out}` , for any input size.
    The number of output features is equal to the number of input planes.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    .. note::
        Atlas training series products do not support backward propagation.

    Args:
        output_size (Union[int, tuple]): the target output size :math:`L_{out}` .
        return_indices (bool, optional): Whether to return the index of the maximum value. Default: ``False`` .

    Inputs:
        - **input** (Tensor) - The input with shape :math:`(N, C, L_{in})` or :math:`(C, L_{in})` .

    Outputs:
        Union(Tensor, tuple(Tensor, Tensor)).

        - If `return_indices` is False, output is a Tensor, with shape :math:`(N, C, L_{out})`. It has the same data
          type as `x`.
        - If `return_indices` is True, output is a Tuple of 2 Tensors, representing the result and where the max
          values are generated.

    Raises:
        TypeError: If `input` is not a tensor.
        TypeError: If dtype of `input` is not float16, float32 or float64.
        TypeError: If `output_size` is not int or tuple.
        TypeError: If `return_indices` is not a bool.
        ValueError: If `output_size` is a tuple and the length of `output_size` is not 1.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> import numpy as np
        >>> input = Tensor(np.array([[[2, 1, 2], [2, 3, 5]]]), mindspore.float16)
        >>> net = mint.nn.AdaptiveMaxPool1d(3)
        >>> output = net(input)
        >>> print(output)
        [[[2. 1. 2.]
          [2. 3. 5.]]]
    """

    def construct(self, input):
        return mint.nn.functional.adaptive_max_pool1d(input, self.output_size, self.return_indices)


__all__ = [
    'AdaptiveAvgPool3d',
    'AdaptiveAvgPool2d',
    'AdaptiveAvgPool1d',
    'AdaptiveMaxPool1d',
    'MaxUnpool2d',
]

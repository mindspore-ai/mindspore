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

"""Defines nn operators with functional form."""

from mindspore.ops import operations as P
from mindspore.ops.operations import nn_ops as NN

fast_gelu_ = P.FastGeLU()
softsign_ = P.Softsign()


def fast_gelu(x):
    r"""
    Fast Gaussian Error Linear Units activation function.

    FastGeLU is defined as follows:

    .. math::
        \text{output} = \frac {x} {1 + \exp(-1.702 * \left| x \right|)} * \exp(0.851 * (x - \left| x \right|)),

    where :math:`x` is the element of the input.

    Args:
        x (Tensor): Input to compute the FastGeLU with data type of float16 or float32.

    Returns:
        Tensor, with the same type and shape as `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> output = ops.fast_gelu(x)
        >>> print(output)
        [[-1.5418735e-01  3.9921875e+00 -9.7473649e-06]
         [ 1.9375000e+00 -1.0052517e-03  8.9824219e+00]]
    """
    return fast_gelu_(x)


def hardshrink(x, lambd=0.5):
    r"""
    Hard Shrink activation function. Calculates the output according to the input elements.

    The formula is defined as follows:

    .. math::
        \text{HardShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        x (Tensor): The input of Hard Shrink with data type of float16 or float32.
        lambd (float): The threshold :math:`\lambda` defined by the Hard Shrink formula. Default: 0.5.

    Returns:
        Tensor, has the same data type and shape as the input.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Raises:
        TypeError: If `lambd` is not a float.
        TypeError: If dtype of `input_x` is neither float16 nor float32.

    Examples:
        >>> x = Tensor(np.array([[ 0.5,  1,  2.0], [0.0533,0.0776,-2.1233]]), mindspore.float32)
        >>> output = ops.hardshrink(x)
        >>> print(output)
        [[ 0.      1.      2.    ]
        [ 0.      0.     -2.1233]]
    """
    hshrink_op = P.HShrink(lambd)
    return hshrink_op(x)


def softsign(x):
    r"""
    Softsign activation function.

    The function is shown as follows:

    .. math::
        \text{SoftSign}(x) = \frac{x}{1 + |x|}

    Args:
        x (Tensor): Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
            additional dimensions, with float16 or float32 data type.

    Outputs:
        Tensor, with the same type and shape as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore.ops import functional as F
        >>> x = Tensor(np.array([0, -1, 2, 30, -30]), mindspore.float32)
        >>> output = F.softsign(x)
        >>> print(output)
        [ 0.        -0.5         0.6666667  0.9677419 -0.9677419]
    """
    return softsign_(x)


def deformable_conv2d(x, weight, offsets, kernel_size, strides, padding, bias=None, dilations=(1, 1, 1, 1), groups=1,
                      data_format="NCHW", deformable_groups=1, modulated=True):
    r"""
    Computes a 2D deformable convolution given 4D `x`, `weight` and `offsets` tensors.

    Args:
        x (Tensor): A 4D tensor of input image. With the format "NCHW" or "NHWC", the data is stored in the order of:
            :math:`(batch, in_channels, in_height, in_width)` when the format is "NCHW".
        weight (Tensor): A 4D tensor of learnable filters. Must have the same type as `x`. The data is stored in the
            order of: :math:`(out_channels, in_channels / groups, filter_height, filter_width)` when the format of
            `x` is "NCHW".
        offsets (Tensor): A 4D tensor of x-y coordinates offset and mask. With the format "NCHW" or "NHWC", when the
            format is "NCHW", the data is stored in the order of:
            :math:`(batch, deformable_groups * filter_height * filter_width * 3, out_height, out_width)`.
        kernel_size (tuple[int]): Required. A tuple of 2 integers. The size of kernel.
        strides (tuple[int]): Required. A tuple of 4 integers. The stride of the sliding window for each dimension of
            input. The dimension order is interpreted according to the data format of `x`. The N and C dimensions must
            be set to 1.
        padding (tuple[int]): Required. A list of 4 integers. The number of pixels to add to each (top, bottom, left,
            right) side of the input.
        bias (Tensor): Optional. An 1D tensor of additive biases to the filter outputs. The data is stored in the
            order of: :math:`(out_channels)`.
        dilations (tuple[int]): Optional. A list of 4 integers. The dilation factor for each dimension of input. The
            dimension order is interpreted according to the data format of `x`. The N and C dimensions must be set
            to 1. Defaults to (1, 1, 1, 1).
        groups (int): Optional. An integer of type int32. The number of blocked connections from input channels
            to output channels. In_channels and out_channels must both be divisible by `groups`. Defaults to 1.
        data_format (str): Optional. The value for data format, is 'NCHW' or 'NHWC'. Defaults to 'NCHW'.
        deformable_groups (int) -  Optional. An integer of type int32. The number of deformable group partitions.
            In_channels must be divisible by `deformable_groups`. Defaults to 1.
        modulated (bool) -  Optional. Specify version of DeformableConv2D, True means v2, False means v1, currently
            only support v2. Defaults to True.

    Returns:
        Tensor, A 4D Tensor of output feature map. With the same type as `x`. With the format "NCHW" or "NHWC", the
            data is stored in the order of: :math:`(batch, out_channels, out_height, out_width)` when the format is
            "NCHW".
        .. math::
            \begin{array}{ll} \\
                \text{out\_height} = {\frac{\text{in\_height} + \text{pad\_top} + \text{pad\_bottom}
                - (\text{dilation\_h} * (\text{filter\_height} - 1) + 1)}{\text{stride\_h}}} + 1 \\
                \text{out\_width} = {\frac{\text{in\_width} + \text{pad\_left} + \text{pad\_right}
                - (\text{dilation\_w} * (\text{filter\_width} - 1) + 1)}{\text{stride\_w}}} + 1 \\
            \end{array}

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Raises:
        TypeError: If `strides`, `padding`, `kernel_size` or `dilations` is not a tuple with integer elements.
        TypeError: If `modulated` is not a bool.
        ValueError: The N or C dimensions of 'strides' or `dilations` is not set to 1.
        ValueError: If `data_format` is neither 'NCHW' nor 'NHWC'.
        ValueError: If `modulated` is not set to True.

    Examples:
        >>> x = Tensor(np.ones((4, 3, 10, 10)), mstype.float32)
        >>> kh, kw = 3, 3
        >>> weight = Tensor(np.ones((5, 3, kh, kw)), mstype.float32)
        >>> offsets = Tensor(np.ones((4, 3 * kh * kw, 8, 8)), mstype.float32)
        >>> output = ops.deformable_conv2d(x, weight, offsets, (kh, kw), (1, 1, 1, 1), (0, 0, 0, 0), data_format="NCHW")
        >>> print(output.shape)
        (4, 5, 8, 8)
    """
    deformable_offsets = NN.DeformableOffsets(strides, padding, kernel_size, dilations, data_format, deformable_groups,
                                              modulated)
    fm_offset = deformable_offsets(x, offsets)

    weight_shape = weight.shape
    out_channel = weight_shape[0]
    if data_format == "NHWC":
        out_channel = weight_shape[3]
    strides_conv = (kernel_size[0], kernel_size[1])
    conv = P.Conv2D(out_channel, kernel_size, 1, "valid", 0, strides_conv, 1, groups, data_format)
    bias_add = P.BiasAdd(data_format)

    output = conv(fm_offset, weight)
    if bias is not None:
        output = bias_add(output, bias)
    return output


def pdist(x, p=2.0):
    r"""
    Computes the p-norm distance between each pair of row vectors in the input. If `x` is a 2D Tensor of
    shape :math:`(N, M)`, then `output` must be a 1D Tensor of shape :math:`(N * (N - 1) / 2,)`. If `x` id a
    Tensor of shape :math:`(*B, N, M)`, then `output` must be a Tensor of shape :math:`(*B, N * (N - 1) / 2)`.

    .. math::
        y[n] = \sqrt[p]{{\mid x_{i} - x_{j} \mid}^p}

    where :math:`x_{i}, x_{j}` are two different row vectors in the input.

    Args:
        x (Tensor) - Input tensor of shape :math:`(*B, N, M)`. *B: batch size, one-dim or multi-dim.
        dtype: float16, float32, float64.
        p (float): p value for the p norm distance to calculate between each vector pair ∈[0,∞]. Default: 2.0.

    Returns:
        Tensor, has the same dtype as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is float16, float32 or float64.
        TypeError: If `p` is not a float.
        ValueError: If `p` is a negative float.
        ValueError: If dimension of `x` is less than 2.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]).astype(np.float32))
        >>> y = ops.pdist(x, p=2.0)
        >>> print(y)
        [1.4142135 2.828427 1.4142135]
    """
    pdist_ = NN.Pdist(p=p)
    return pdist_(x)


__all__ = [
    'deformable_conv2d',
    'fast_gelu',
    'hardshrink',
    'softsign',
    'pdist',
]
__all__.sort()

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
        x (Tensor) - Input tensor of shape :math:`(*B, N, M)`. :math:`*B` is batch size, one-dim or multi-dim.
            dtype: float16, float32 or float64.
        p (float): p value for the p-norm distance to calculate between each vector pair. :math:`p∈[0,∞]`. Default: 2.0.

    Returns:
        Tensor, has the same dtype as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.
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


def cross_entropy(inputs, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    r"""
    The cross entropy loss between input and target.

    The cross entropy support two kind of targets:

    - Class indices (int) in the range :math:`[0, C)` where :math:`C` is the number of classes,
      the loss with reduction=none can be described as:

      .. math::

          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
          \cdot \mathbb{1}\{y_n \not= \text{ignore_index}\}

      where :math:`x` is the inputs, :math:`t` is the target, :math:`w` is the weight,
      N is the batch size, :math:`c` belonging to [0, C-1] is class index, where :math:`C` is the number of classes.

      If reduction is not 'none' (default 'mean'), then

      .. math::

          \ell(x, y) = \begin{cases}
              \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n} \cdot \mathbb{1}\{y_n \not= \text{ignore_index}\}} l_n, &
              \text{if reduction} = \text{`mean';}\\
              \sum_{n=1}^N l_n,  &
              \text{if reduction} = \text{`sum'.}
              \end{cases}

    - Probabilities (float) for each class, useful when labels beyond a single class per minibatch item
      are required, the loss with reduction=none can be described as:

      .. math::

          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - \sum_{c=1}^C w_c \log \frac{\exp(x_{n,c})}{\sum_{i=1}^C \exp(x_{n,i})} y_{n,c}

      where :math:`x` is the inputs, :math:`t` is the target, :math:`w` is the weight,
      N is the batch size, :math:`c` belonging to [0, C-1] is class index, where :math:`C` is the number of classes.

      If reduction is not 'none' (default 'mean'), then

      .. math::

          \ell(x, y) = \begin{cases}
              \frac{\sum_{n=1}^N l_n}{N}, &
              \text{if reduction} = \text{`mean';}\\
              \sum_{n=1}^N l_n,  &
              \text{if reduction} = \text{`sum'.}
              \end{cases}

    Args:
        inputs (Tensor): :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)`.
            `inputs` is expected to be log-probabilities, data type must be float16 or float32.
        target (Tensor): :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` for
            high-dimensional loss.
        weight (Tensor): A rescaling weight applied to the loss of each batch element.
            If not None, the shape is :math:`(C,)`,
            data type must be float16 or float32. Default: None.
        ignore_index (int): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default: -100
        reduction (string):  Apply specific reduction method to the output: 'none', 'mean', or 'sum'.
            Default: 'mean'.
        label_smoothing (float): Label smoothing values, a regularization tool used to prevent the model
            from overfitting when calculating Loss. The value range is [0.0, 1.0]. Default value: 0.0.

    Returns:
        Tensor, the computed loss value.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Example::

        >>> inputs = mindspore.Tensor(np.random.randn(3, 5))
        >>> target = mindspore.Tensor(np.array([1, 0, 4]))
        >>> output = ops.cross_entropy(inputs, target)

    """
    class_dim = 0 if inputs.ndim == 1 else 1
    if inputs.size == target.size:
        return _cross_entropy(inputs, target, class_dim, weight, reduction, label_smoothing)
    return nll_loss(P.LogSoftmax(class_dim)(inputs), target, weight, ignore_index, reduction, label_smoothing)


def _cross_entropy(inputs, target, target_dim, weight=None, reduction='mean', label_smoothing=0.0):
    """cross entropy inner function"""
    class_dim = 0 if inputs.ndim == 1 else 1
    n_classes = inputs.shape[class_dim]
    inputs = P.LogSoftmax(target_dim)(inputs)
    if label_smoothing > 0.0:
        target = target * (1 - label_smoothing) + label_smoothing / n_classes

    if weight is None:
        weight = P.OnesLike()(inputs)

    if reduction == 'mean':
        return -(inputs * target * weight).sum() / (inputs.size / n_classes)
    if reduction == 'sum':
        return -(inputs * target * weight).sum()
    return -(inputs * target * weight).sum(class_dim)


def nll_loss(inputs, target, weight=None, ignore_index=None, reduction='mean', label_smoothing=0.0):
    r"""
    Gets the negative log likelihood loss between inputs and target.

    The nll loss with reduction=none can be described as:

    .. math::

        \ell(x, t)=L=\left\{l_{1}, \ldots, l_{N}\right\}^{\top},
        \quad l_{n}=-w_{t_{n}} x_{n, t_{n}},
        \quad w_{c}=\text { weight }[c] \cdot \mathbb{1}
        \{c \not= \text{ignore_index}\},

    where :math:`x` is the inputs, :math:`t` is the target, :math:`w` is the weight,
    N is the batch size, :math:`c` belonging to [0, C-1] is class index, where :math:`C` is the number of classes.

    If reduction is not 'none' (default 'mean'), then

    .. math::

        \ell(x, t)=\left\{\begin{array}{ll}
        \sum_{n=1}^{N} \frac{1}{\sum_{n=1}^{N} w_{t n}} l_{n}, & \text { if reduction }=\text { 'mean'; } \\
        \sum_{n=1}^{N} l_{n}, & \text { if reduction }=\text { 'sum' }
        \end{array}\right.

    Args:
        inputs (Tensor): :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)`.
            `inputs` is expected to be log-probabilities, data type must be float16 or float32.
        target (Tensor): :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` for
            high-dimensional loss, data type must be int32.
        weight (Tensor): A rescaling weight applied to the loss of each batch element.
            If not None, the shape is :math:`(C,)`.
            The data type must be float16 or float32. Default: None.
        ignore_index (int): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default: -100
        reduction (string):  Apply specific reduction method to the output: 'none', 'mean', or 'sum'.
            Default: 'mean'.
        label_smoothing (float): Label smoothing values, a regularization tool used to prevent the model
            from overfitting when calculating Loss. The value range is [0.0, 1.0]. Default value: 0.0.

    Outputs:
        Tensor, the computed loss value.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Example::

        >>> inputs = mindspore.Tensor(np.random.randn(3, 5))
        >>> target = mindspore.Tensor(np.array([1, 0, 4]))
        >>> output = ops.nll_loss(inputs, target)

    """
    ndim = inputs.ndim
    if ndim == 2:
        ret = _nll_loss(inputs, target, -1, weight, ignore_index, reduction, label_smoothing)
    elif ndim == 4:
        ret = _nll_loss(inputs, target, 1, weight, ignore_index, reduction, label_smoothing)
    else:
        n = inputs.shape[0]
        c = inputs.shape[1]
        out_size = (n,) + inputs.shape[2:]
        inputs = inputs.view(n, c, 1, -1)
        target = target.view(n, 1, -1)
        if reduction != 'none':
            ret = _nll_loss(inputs, target, 1, weight, ignore_index, reduction, label_smoothing)
        else:
            ret = _nll_loss(inputs, target, 1, weight, ignore_index, label_smoothing=label_smoothing)
            ret = ret.view(out_size)
    return ret


def _nll_loss(inputs, target, target_dim=-1, weight=None, ignore_index=None, reduction='none', label_smoothing=0.0):
    """nll loss inner function"""
    if target.ndim == inputs.ndim - 1:
        target = target.expand_dims(target_dim)
    loss = P.Neg()(P.GatherD()(inputs, target_dim, target))
    smooth_loss = P.Neg()(inputs.sum(axis=target_dim, keepdims=True))
    if weight is not None:
        loss_weights = P.Gather()(weight, target, 0)
        loss = loss * loss_weights
    else:
        loss_weights = P.OnesLike()(loss)
    if ignore_index is not None:
        non_pad_mask = P.Equal()(target, ignore_index)
        loss = loss.masked_fill(non_pad_mask, 0.)
        loss_weights = loss_weights.masked_fill(non_pad_mask, 0.)
        smooth_loss = smooth_loss.masked_fill(non_pad_mask, 0.)

    loss = loss.squeeze(target_dim)
    smooth_loss = smooth_loss.squeeze(target_dim)

    if reduction == 'sum':
        loss = loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == 'mean':
        loss = loss.sum() / loss_weights.sum()
        smooth_loss = smooth_loss.mean()

    eps_i = label_smoothing / inputs.shape[target_dim]
    loss = (1. - label_smoothing) * loss + eps_i * smooth_loss

    return loss


def intopk(x1, x2, k):
    r"""
    Determines whether the targets are in the top `k` predictions.

    Args:
        x1 (Tensor): A 2D Tensor defines the predictions of a batch of samples with float16 or float32
          data type.
        x2 (Tensor): A 1D Tensor defines the labels of a batch of samples with int32 data type. The size of `x2`
          must be equal to the first dimension of `x1`. The values of `x2` can not be negative and
          must be equal to or less than index of x1's second dimension.
        k (int): Specifies the number of top elements to be used for computing precision along the last dimension.

    Returns:
        Tensor has 1 dimension of type bool and the same shape with `x2`. For labeling sample `i` in `x2`,
        if the label in the first `k` predictions for sample `i` is in `x1`, then the value is True, otherwise False.

    Raises:
        TypeError: If `k` is not an int.
        TypeError: If `x1` or `x2` is not a Tensor.
        TypeError: If dtype of `x1` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = Tensor(np.array([[1, 8, 5, 2, 7], [4, 9, 1, 3, 5]]), mindspore.float32)
        >>> x2 = Tensor(np.array([1, 3]), mindspore.int32)
        >>> output = ops.intopk(x1, x2, 3)
        >>> print(output)
        [ True  False]
    """
    return P.InTopK(k)(x1, x2)


def grid_sample(input_x, grid, interpolation_mode='bilinear', padding_mode='zeros', align_corners=False):
    """
    Given an `input_x` and a flow-field `grid`, computes the `output` using `input_x` values and pixel locations from
    `grid`. Only spatial (4-D) and volumetric (5-D) `input_x` is supported.

    In the spatial (4-D) case, for `input_x` with shape :math:`(N, C, H_{in}, W_{in})` and `grid` with shape
    :math:`(N, H_{out}, W_{out}, 2)`, the `output` will have shape :math:`(N, C, H_{out}, W_{out})`.

    For each output location `output[n, :, h, w]`, the size-2 vector `grid[n, h, w]` specifies `input_x` pixel
    locations `x` and `y`, which are used to interpolate the output value `output[n, :, h, w]`. In the case of 5D
    inputs, `grid[n, d, h, w]`, specifies the `x`, `y`, `z` pixel locations for interpolating
    `output[n, :, d, h, w]`. And `interpolation_mode` argument specifies "nearest" or "bilinear" or "bicubic"
    (supported in 4D case only) interpolation method to sample the input pixels.

    `grid` specifies the sampling pixel locations normalized by the `input_x` spatial dimensions. Therefore, it should
    have most values in the range of :math:`[-1, 1]`.

    If `grid` has values outside the range of :math:`[-1, 1]`, the corresponding outputs are handled as defined by
    `padding_mode`. If `padding_mode` is set to be "zeros", use :math:`0` for out-of-bound grid locations. If
    `padding_mode` is set to be "border", use border values for out-of-bound grid locations. If `padding_mode` is set
    to be "reflection", use values at locations reflected by the border for out-of-bound grid locations. For location
    far away from the border, it will keep being reflected until becoming in bound.

    Args:
        input_x (Tensor): input with shape of :math:`(N, C, H_{in}, W_{in})`(4-D case) or :math:`(N, C, D_{in},
            H_{in}, W_{in})`(5-D case) and dtype of float32 or float64.
        grid (Tensor): flow-field with shape of :math:`(N, H_{out}, W_{out}, 2)`(4-D case) or :math:`(N, D_{out},
            H_{out}, W_{out}, 3)`(5-D case) and same dtype as `input_x`.
        interpolation_mode (str): An optional string specifying the interpolation method. The optional values are
            "bilinear", "nearest" or "bicubic". Default: "bilinear". Note: `bicubic` supports only 4-D input. When
            `interpolation_mode`="bilinear"` and the input is 5-D, the interpolation mode used internally will actually
            be trilinear. However, when the input is 4-D, the interpolation mode will legistimately be bilinear.
        padding_mode (str): An optional string specifying the pad method. The optional values are "zeros", "border" or
            "reflection". Default: "zeros".
        align_corners (bool): An optional bool. If set to `True`, the extrema (-1 and 1) are considered as referring to
            the center points of the input’s corner pixels. If set to `False`, they are instead considered as referring
            to the corner points of the input’s corner pixels, making the sampling more resolution agnostic. Default:
            `False`.

    Outputs:
        Tensor, dtype is the same as `input_x` and whose shape is: math:`(N, C, H_{out}, W_{out})`(4-D) and
            :math:`(N, C, D_{out}, H_{out}, W_{out})`(5-D).

    Raises:
        TypeError: If `input_x` or `grid` is not a Tensor.
        TypeError: If the dtypes of `input_x` and `grid` are inconsistent.
        TypeError: If the dtype of `input_x` or `grid` is not a valid type.
        TypeError: If `align_corners` is not a boolean value.
        ValueError: If the rank of `input_x` or `grid` is not equal to 4(4-D case) or 5(5-D case).
        ValueError: If the first dimension of `input_x` is not equal to that of `grid`.
        ValueError: If the last dimension of `grid` is not equal to 2(4-D case) or 3(5-D case).
        ValueError: If `interpolation_mode` is not "bilinear", "nearest", "bicubic" or a string value.
        ValueError: If `padding_mode` is not "zeros", "border", "reflection" or a string value.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> input_x = Tensor(np.arange(16).reshape((2, 2, 2, 2)).astype(np.float32))
        >>> grid = Tensor(np.arange(0.2, 1, 0.1).reshape((2, 2, 1, 2)).astype(np.float32))
        >>> output = grid_sample(input_x, grid, interpolation_mode='bilinear', padding_mode='zeros',
                                     align_corners=True)
        >>> print(output)
        [[[[ 1.9      ]
           [ 2.1999998]]
          [[ 5.9      ]
           [ 6.2      ]]]
         [[[10.5      ]
           [10.8      ]]
          [[14.5      ]
           [14.8      ]]]]
    """
    if input_x.ndim == 4:
        return NN.GridSampler2D(interpolation_mode, padding_mode, align_corners)(input_x, grid)
    return NN.GridSampler3D(interpolation_mode, padding_mode, align_corners)(input_x, grid)


__all__ = [
    'deformable_conv2d',
    'fast_gelu',
    'hardshrink',
    'intopk',
    'softsign',
    'pdist',
    'cross_entropy',
    'grid_sample',
    'nll_loss'
]
__all__.sort()

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

"""

NN Operators with better performance

"""
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.auto_generate.gen_ops_prim import Convolution, ConstantPadND, MaxPoolWithIndices, MaxPoolWithMask
from mindspore.ops.auto_generate import leaky_relu_ext
from mindspore import _checkparam as validator


def _check_stride_when_same_mode(stride):
    """ stride must be 1 when pad mode is same """
    if isinstance(stride, int):
        if stride != 1:
            raise ValueError(f"For conv2d, 'padding=same' is not supported for stride convolution, " \
                                f"but got {stride}")
    elif isinstance(stride, tuple):
        validator.check_int(len(stride), 2, validator.EQ, "stride", 'conv2d')
        if not all(s == 1 for s in stride):
            raise ValueError(f"For conv2d, 'padding=same' is not supported for stride convolution, " \
                                f"but got {stride}")
    else:
        raise TypeError(f"For conv2d, the parameter 'stride' must be a int/tuple, but got {type(stride)}")


def _get_pad_info(dilation, weight):
    """ Get pad list by dilation and weight shape """
    need_pad_nd = False
    pad_l = ()
    pad_r = ()
    for i in range(2):
        d = dilation[i]
        weight_size = weight.shape[i + 2]
        pad = d * (weight_size - 1)
        pad_l += (int(pad / 2),)
        pad_r += (int(pad - pad_l[i]),)
        if pad_l[i] != pad_r[i]:
            need_pad_nd = True
    return need_pad_nd, pad_l, pad_r


def _get_pad_nd_info(pad_l, pad_r):
    """ Get pad_nd list if input need to exec pad_nd """
    pad_nd = ()
    new_pad_l = ()
    for i in range(2):
        delta_pad = pad_r[i] - pad_l[i]
        if delta_pad > 0:
            pad_nd = (0, delta_pad,) + pad_nd
            new_pad_l += (pad_l[i],)
        else:
            pad_nd = (delta_pad, 0,) + pad_nd
            new_pad_l += (pad_r[i],)
    return pad_nd, new_pad_l


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    r"""
    Applies a 2D convolution over an input tensor. The input tenor is typically of
    shape :math:`(N, C_{in}, H_{in}, W_{in})`, where :math:`N` is batch size, :math:`C` is
    channel number, :math:`H` is feature height, :math:`W` is feature width.

    The output is calculated based on formula:

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{X}(N_i, k)})

    where :math:`bias` is the output channel bias, :math:`ccor` is
    the `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    , :math:`weight` is the convolution kernel value and :math:`X` represents the input feature map.

    Here are the indices' meanings:

    - :math:`i` corresponds to the batch number, the range is :math:`[0, N-1]`,
      where :math:`N` is the batch size of the input.

    - :math:`j` corresponds to the output channel, the range is :math:`[0, C_{out}-1]`,
      where :math:`C_{out}` is the number of output channels, which is also equal to the number of kernels.

    - :math:`k` corresponds to the input channel, the range is :math:`[0, C_{in}-1]`,
      where :math:`C_{in}` is the number of
      input channels, which is also equal to the number of channels in the convolutional kernels.

    Therefore, in the above formula, :math:`{bias}(C_{out_j})` represents the bias of the :math:`j`-th
    output channel, :math:`{weight}(C_{out_j}, k)` represents the slice of the :math:`j`-th convolutional
    kernel in the :math:`k`-th channel, and :math:`{X}(N_i, k)` represents the slice of the :math:`k`-th input
    channel in the :math:`i`-th batch of the input feature map.

    The shape of the convolutional kernel is given by :math:`(\text{kernel_size[0]}, \text{kernel_size[1]})`,
    where :math:`\text{kernel_size[0]}` and :math:`\text{kernel_size[1]}` are the height and width of the kernel,
    respectively.
    If we consider the input and output channels as well as the `group` parameter, the complete kernel shape
    will be :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size[0]}, \text{kernel_size[1]})`,
    where `group` is the number of groups dividing `x`'s input channel when applying group convolution.

    For more details about convolution layer, please refer to `Gradient Based Learning Applied to Document Recognition
    <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_ and
    `ConvNets <http://cs231n.github.io/convolutional-networks/>`_.

    Note:
        On Ascend platform, only group convolution in depthwise convolution scenarios is supported.
        That is, when `groups>1`, condition :math:`C_{in}` = :math:`C_{out}` = `groups` must be satisfied.

    Args:
        input (Tensor): Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.
        weight (Tensor): Tensor of shape
            :math:`(N, C_{in} / \text{groups}, \text{kernel_size[0]}, \text{kernel_size[1]})`, then the size of kernel
            is :math:`(\text{kernel_size[0]}, \text{kernel_size[1]})`.
        bias (Tensor, optional): Bias Tensor with shape :math:`(C_{out})`.
            When bias is ``None`` , zeros will be used. Default: ``None`` .
        stride (Union(int, tuple[int]), optional): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: ``1`` .
        padding (Union(int, tuple[int], list[int], str), optional): Implicit paddings on both sides of the input `x`.
            Can be a string, one integer or a tuple/list with 2 integers.
            If `padding` is a string, the optional values are ``"same"`` , ``"valid"``.

            - same: Adopts the way of completion. The height and width of the output will be equal to
              the input `x` divided by stride. The padding will be evenly calculated in top and bottom,
              left and right possiblily. Otherwise, the last extra padding will be calculated from the bottom
              and the right side. If this mode is set, `padding` must be 0.

            - valid: Adopts the way of discarding. The possible largest height and width of output will be returned
              without padding. Extra pixels will be discarded. If this mode is set, `padding` must be 0.

            If `padding` is one integer, the paddings of top, bottom, left and right are the same, equal to padding.
            If `padding` is a tuple/list with 2 integers, the padding of top adn bottom is padding[0],
            and the padding of left and right is padding[1]. Default: ``0`` .
        dilation (Union(int, tuple[int]), optional): Gaps between kernel elements.The data type is int or a tuple of
            2 integers. Specifies the dilation rate to use for dilated convolution. If set to be :math:`k > 1`,
            there will be :math:`k - 1` pixels skipped for each sampling location. Its value must
            be greater than or equal to 1 and bounded by the height and width of the input `x`. Default: ``1`` .
        groups (int, optional): Splits `input` into groups. Default: ``1`` .

    Returns:
        Tensor, the value that applied 2D convolution. The shape is :math:`(N, C_{out}, H_{out}, W_{out})`.
        To see how different pad modes affect the output shape, please refer to
        :class:`mindspore.nn.Conv2d` for more details.


    Raises:
        TypeError: If `stride`, `padding` or `dilation` is neither an int nor a tuple.
        TypeError: `groups` is not an int.
        TypeError: If `bias` is not a Tensor.
        ValueError: If  the shape of `bias` is not :math:`(C_{out})` .
        ValueError: If `stride` or `dilation` is less than 1.
        ValueError: If `pad_mode` is not one of 'same', 'valid' or 'pad'.
        ValueError: If `padding` is a tuple/list whose length is not equal to 2.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.ones([10, 32, 32, 32]), mindspore.float32)
        >>> weight = Tensor(np.ones([32, 32, 3, 3]), mindspore.float32)
        >>> output = ops.extend.conv2d(x, weight)
        >>> print(output.shape)
        (10, 32, 30, 30)
    """

    def _convolution_same(input, weight, bias, dilation, groups):
        """ convolution when mode is 'same' """
        if isinstance(dilation, int):
            dilation = (dilation,) * 2
        validator.check_int(len(weight.shape), 4, validator.EQ, "weight.shape", 'conv2d')
        validator.check_int(len(dilation), 2, validator.EQ, "dilation", 'conv2d')

        # Calc padding info
        need_pad_nd, pad_l, pad_r = _get_pad_info(dilation, weight)
        if not need_pad_nd:
            conv = _get_cache_prim(Convolution)(stride, pad_l, dilation, False, (0, 0), groups)
            return conv(input, weight, bias)

        # Calc pad nd info
        pad_nd, pad_l = _get_pad_nd_info(pad_l, pad_r)
        pad_nd_op = _get_cache_prim(ConstantPadND)()
        padded_input = pad_nd_op(input, pad_nd, 0)
        conv = _get_cache_prim(Convolution)(stride, pad_l, dilation, False, (0, 0), groups)
        return conv(padded_input, weight, bias)

    if isinstance(padding, int):
        padding = (padding,) * 2

    if isinstance(padding, (tuple, list)):
        conv = _get_cache_prim(Convolution)(stride, padding, dilation, False, (0, 0), groups)
        return conv(input, weight, bias)
    if isinstance(padding, str):
        if padding == 'valid':
            conv = _get_cache_prim(Convolution)(stride, (0, 0), dilation, False, (0, 0), groups)
            return conv(input, weight, bias)
        if padding == 'same':
            _check_stride_when_same_mode(stride)
            return _convolution_same(input, weight, bias, dilation, groups)
        raise ValueError(f"For conv2d, the parameter 'padding' must be 'same' or 'valid' when " \
                            f"the type of 'padding' is string.")
    raise TypeError(f"For conv2d, the parameter 'padding' must be a tuple/list " \
                    f"or a string, but got {type(padding)}")


def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, *, ceil_mode=False, return_indices=False):
    r"""
    Performs a 2D max pooling on the input Tensor.

    Typically, the input is a Tensor with shape :math:`(N_{in}, C_{in}, H_{in}, W_{in})`, outputs
    regional maximum in the :math:`(H_{in}, W_{in})`-dimension. Given `kernel_size`
    :math:`ks = (h_{ker}, w_{ker})` and `stride` :math:`s = (s_0, s_1)`, the operation is as follows:

    .. math::
        \text{output}(N_i, C_j, h, w) =
        \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times h + m, s_1 \times w + n)

    .. warning::
        Only support on Atlas A2 training series.

    Args:
        input (Tensor): Tensor of shape :math:`(N_{in}, C_{in}, H_{in}, W_{in})` with data type of float32
            in Ascend.
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the maximum value and arg
            value, is an int number that represents height and width of the kernel, or a tuple of
            two int numbers that represent height and width respectively.
        stride (Union[int, tuple[int], None]): The distance of kernel moving, an int number that represents
            the height and width of movement are both stride, or a tuple of two int numbers that
            represent height and width of movement respectively.
            Default: ``None`` , which indicates the moving step is `kernel_size` .
        padding (Union[int, tuple[int]]): An int number that represents the height and width of movement are both
            strides, or a tuple of two int numbers that represent height and width of movement respectively.
            Default: ``0`` .
        dilation (Union[int, tuple[int]]): Control the stride of elements in the kernel. Default: ``1`` .
        ceil_mode (bool): Whether to use ceil instead of floor to calculate output shape. Default: ``False`` .
        return_indices (bool): Whether to output the indices of max value. Default: ``False`` .

    Returns:
        If `return_indices` is ``False`` , return a Tensor `output`, else return a tuple (`output`, `argmax`).

        - **output** (Tensor) - Maxpooling result, with shape :math:`(N_{out}, C_{out}, H_{out}, W_{out})`.
          It has the same data type as `input`.

        .. math::
            H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                \times (\text{kernel_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

        .. math::
            W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                \times (\text{kernel_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

        - **argmax** (Tensor) - Index corresponding to the maximum value. In Ascend, data type is int32.
          It will be return only when `return_indices` is True.

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
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.arange(20 * 16 * 50 * 32).reshape((20, 16, 50, 32)), mindspore.float32)
        >>> output_tensor, argmax = ops.extend.max_pool2d(input, kernel_size=(3, 2), stride=(2, 1),
        ...                                               ceil_mode=False, return_indices=True)
        >>> print(output_tensor.shape)
        (20, 16, 24, 31)
        >>> print(argmax.shape)
        (20, 16, 24, 31)
    """
    strides = stride if (stride is not None) else kernel_size
    if return_indices:
        max_pool_func_ = _get_cache_prim(MaxPoolWithIndices)(kernel_size, strides, padding, dilation, ceil_mode)
        out, indices = max_pool_func_(input)
    else:
        max_pool_func_ = _get_cache_prim(MaxPoolWithMask)(kernel_size, strides, padding, dilation, ceil_mode)
        out, indices = max_pool_func_(input)
    if return_indices:
        return out, indices
    return out


__all__ = ['conv2d', 'max_pool2d', 'leaky_relu_ext']

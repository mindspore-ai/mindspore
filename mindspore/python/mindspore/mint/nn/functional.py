# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
"""mint nn functional."""
from __future__ import absolute_import
from mindspore import ops
from mindspore import _checkparam as validator
from mindspore.ops.function.nn_func import max_pool2d_ext as max_pool2d
from mindspore.ops.functional import (
    conv_transpose2d,
    grid_sample
)
# 1

# 2

# 3

# 4
from mindspore.ops.function.nn_func import interpolate_ext as interpolate
# 5
from mindspore.ops.function.nn_func import pad_ext as pad
# 6
from mindspore.ops.auto_generate import unfold_ext as unfold
# 7
from mindspore.ops.auto_generate import fold_ext as fold
# 8
from mindspore.ops.functional import layer_norm
# 9

# 10

# 11
from mindspore.ops.functional import relu

from mindspore.ops.function.nn_func import relu_

# 12

# 13

# 14
from mindspore.ops.function.nn_func import dropout_ext as dropout
from mindspore.ops.function.nn_func import dropout2d_ext as dropout2d
# 15
from mindspore.ops.functional_overload import conv1d
from mindspore.ops.function.nn_func import conv2d_ext as conv2d
# 16
from mindspore.ops.function.nn_func import log_softmax_ext as log_softmax
# 18
from mindspore.ops.auto_generate import prelu
# 19

# 20

# 21
from mindspore.ops.functional_overload import conv3d
# 22

# 23

# 24

# 25

# 26

# 27

# 28

# 29

# 30

# 31
from mindspore.ops.function.nn_func import softmax_ext as softmax

# 32

# 33

# 34
from mindspore.ops.function.nn_func import batch_norm_ext as batch_norm
# 35

# 36
from mindspore.ops.functional_overload import gelu
# 37

# 38
from mindspore.ops.functional import dense as linear
# 39
from mindspore.ops.functional import group_norm
# 40

# 41

# 42

# 43

# 44
from mindspore.ops.auto_generate import soft_margin_loss
# 45

# 46
from mindspore.ops.auto_generate import silu as silu_func
from mindspore.ops.auto_generate import inplace_silu
# 47

# 48

# 49
from mindspore.ops.functional import sigmoid
from mindspore.ops.functional import inplace_sigmoid as sigmoid_
# 50

# 51

# 52
from mindspore.ops.functional import embedding
# 53

# 54
from mindspore.ops.functional_overload import pixel_shuffle
# 55

# 56

# 57

# 58

# 59

# 60

# 61

# 62

# 63

# 64

# 65

# 66

# 67

# 68

# 69

# 70

# 71

# 72

# 73

# 74

# 75
from mindspore.ops.function.nn_func import adaptive_max_pool2d
# 76

# 77

# 78

# 79

# 80

# 81

# 82

# 83

# 84

# 85

# 86

# 87

# 88

# 89
from mindspore.ops.auto_generate import avg_pool1d_ext as avg_pool1d
# 90
from mindspore.ops.function.nn_func import avg_pool2d_ext as avg_pool2d
# 91
from mindspore.ops.function.nn_func import avg_pool3d_ext as avg_pool3d
# 92
from mindspore.ops.auto_generate import leaky_relu_ext as leaky_relu
# 93
from mindspore.ops.auto_generate import softplus_ext as softplus  # pylint: disable=W0611
# 94
from mindspore.ops.function.math_func import tanh
# 95

# 96

# 97

# 98

# 99
from mindspore.ops.auto_generate import selu_ext as selu  # pylint: disable=W0611
# 100
from mindspore.ops.auto_generate import softshrink  # pylint: disable=W0611
# 152
from mindspore.ops.auto_generate import adaptive_avg_pool3d_ext
# 220
from mindspore.ops.function.nn_func import hardshrink  # pylint: disable=W0611
# 221
from mindspore.ops.function.nn_func import hardsigmoid  # pylint: disable=W0611
# 222
from mindspore.ops.function.nn_func import hardswish  # pylint: disable=W0611
# 267
from mindspore.ops.auto_generate import mish_ext as mish  # pylint: disable=W0611
# 238
from mindspore.ops.auto_generate import l1_loss_ext as l1_loss  # pylint: disable=W0611

#254
from mindspore.ops.auto_generate import max_unpool2d_ext as max_unpool2d

# 256
from mindspore.ops.auto_generate import inplace_threshold as threshold_
from mindspore.ops.auto_generate import threshold as threshold_op
# 257

# 258
from mindspore.ops.function.nn_func import mse_loss_ext as mse_loss
# 259

# 323

# 324
from mindspore.ops.auto_generate import elu_ext
from mindspore.ops.auto_generate import inplace_elu

# 421
from mindspore.ops.auto_generate import flatten_ext as flatten

# 426
from mindspore.ops.function.clip_func import clamp
# 427
from mindspore.ops.function.math_func import norm_ext
# 428
from mindspore.ops.functional import broadcast_to
# 536
from mindspore.ops.function.nn_func import glu_ext as glu
# 537
from mindspore.ops.auto_generate import hardtanh as hardtanh_op
from mindspore.ops.auto_generate import inplace_hardtanh as hardtanh_
# 548
from mindspore.ops.function.nn_func import kl_div_ext as kl_div
# 556
from mindspore.ops.function.nn_func import logsigmoid_ext as logsigmoid

from mindspore.ops.auto_generate import adaptive_avg_pool1d
from mindspore.ops.auto_generate import cosine_embedding_loss
from mindspore.ops.functional import adaptive_avg_pool2d_ext as adaptive_avg_pool2d
from mindspore.ops.function.nn_func import cross_entropy_ext as cross_entropy
from mindspore.ops.function.nn_func import nll_loss_ext as nll_loss

def silu(input, inplace=False):
    r"""
    Computes Sigmoid Linear Unit of input element-wise. The SiLU function is defined as:

    .. math::

        \text{SiLU}(x) = x * \sigma(x),

    where :math:`x` is an element of the input, :math:`\sigma(x)` is Sigmoid function.

    .. math::

        \text{sigma}(x_i) = \frac{1}{1 + \exp(-x_i)},

    SiLU Function Graph:

    .. image:: ../images/SiLU.png
        :align: center

    Args:
        input (Tensor): `input` is :math:`x` in the preceding formula. Input with the data type
            float16 or float32.
        inplace (bool, optional): If it is ``True``, enable the in place update function. Default value: ``False``.

    Returns:
        Tensor, with the same type and shape as the `input`.

    Raises:
        TypeError: If dtype of `input` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> import numpy as np
        >>> input = Tensor(np.array([-1, 2, -3, 2, -1]), mindspore.float16)
        >>> output = mint.nn.functional.silu(input, inplace=False)
        >>> print(output)
        [-0.269  1.762  -0.1423  1.762  -0.269]
    """
    if inplace:
        return inplace_silu(input)
    return silu_func(input)


def elu(input, alpha=1.0, inplace=False):
    r"""
    Exponential Linear Unit activation function

    Applies the exponential linear unit function element-wise. The activation function is defined as:

    .. math::
        ELU_{i} =
        \begin{cases}
        x_i, &\text{if } x_i \geq 0; \cr
        \alpha * (\exp(x_i) - 1), &\text{otherwise.}
        \end{cases}

    where :math:`x_i` represents the element of the input and :math:`\alpha` represents the `alpha` parameter, and
    `alpha` represents the smoothness of the ELU.

    ELU Activation Function Graph:

    .. image:: ../images/ELU.png
        :align: center

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The input of ELU is a Tensor of any dimension.
        alpha (float, optional): The alpha value of ELU, the data type is float. Default: ``1.0``.
        inplace (bool, optional): Whether to use inplace mode, the data type is bool. Default: ``False``.

    Returns:
        Tensor, with the same shape and type as the `input`.

    Raises:
        RuntimeError: If the dtype of `input` is not float16, float32 or bfloat16.
        TypeError: If the dtype of `alpha` is not float.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> import numpy as np
        >>> input = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float32)
        >>> output = mint.nn.functional.elu(input)
        >>> print(output)
        [-0.63212055  -0.86466473  0.  2.  1.]
    """
    if inplace:
        return inplace_elu(input, alpha)
    return elu_ext(input, alpha)


def elu_(input, alpha=1.0):
    r"""
    Exponential Linear Unit activation function

    Applies the exponential linear unit function inplace element-wise. The activation function is defined as:

    .. math::
        ELU_{i} =
        \begin{cases}
        x_i, &\text{if } x_i \geq 0; \cr
        \alpha * (\exp(x_i) - 1), &\text{otherwise.}
        \end{cases}

    where :math:`x_i` represents the element of the input and :math:`\alpha` represents the `alpha` parameter, and
    `alpha` represents the smoothness of the ELU.

    ELU Activation Function Graph:

    .. image:: ../images/ELU.png
        :align: center

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The input of ELU is a Tensor of any dimension.
        alpha (float, optional): The alpha value of ELU, the data type is float and `alpha` should be
            greater than 0. Default: ``1.0``.

    Returns:
        Tensor, with the same shape and type as the `input`.

    Raises:
        RuntimeError: If the dtype of `input` is not float16, float32 or bfloat16.
        TypeError: If the dtype of `alpha` is not float.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> import numpy as np
        >>> input = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float32)
        >>> mint.nn.functional.elu_(input)
        >>> print(input)
        [-0.63212055  -0.86466473  0.  2.  1.]
    """
    return inplace_elu(input, alpha)


def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    r"""
    Applies the hardtanh activation function element-wise. The activation function is defined as:

    .. math::
        \text{hardtanh}(input) = \begin{cases}
            max\_val, & \text{ if } input > max\_val \\
            min\_val, & \text{ if } input < min\_val \\
            input, & \text{ otherwise. }
        \end{cases}

    Linear region range :math:`[min\_val, max\_val]` can be adjusted using `min_val` and `max_val`.

    Hardtanh Activation Function Graph:

    .. image:: ../images/Hardtanh.png
        :align: center

    .. warning::
        This is an experimental optimizer API that is subject to change.

    Args:
        input (Tensor): Input Tensor.
        min_val (Union[bool, int, float], optional): Minimum value of the linear region range. Default: ``-1.0`` .
        max_val (Union[bool, int, float], optional): Maximum value of the linear region range. Default: ``1.0`` .
        inplace (bool, optional): Whether to apply erasing inplace. Default: ``False``.

    Returns:
        Tensor, with the same dtype and shape as `input`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If dtype of `input` is not one of: int8, int16, int32, int64, uint8, float16, float32, bfloat16.
        TypeError: If dtype of `min_val` is neither float nor int.
        TypeError: If dtype of `max_val` is neither float nor int.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> x = Tensor([-1, -2, 0, 2, 1], mindspore.float16)
        >>> output = mint.nn.functional.hardtanh(x, min_val=-1.0, max_val=1.0, inplace=False)
        >>> print(output)
        [-1. -1.  0.  1.  1.]
    """
    if inplace:
        return hardtanh_(input, min_val, max_val)
    return hardtanh_op(input, min_val, max_val)


def relu6(input, inplace=False):
    r"""
    Computes ReLU (Rectified Linear Unit) upper bounded by 6 of input tensors element-wise.

    .. math::

        \text{ReLU6}(input) = \min(\max(0,input), 6)

    It returns :math:`\min(\max(0,input), 6)` element-wise.

    ReLU6 Activation Function Graph:

    .. image:: ../images/ReLU6.png
        :align: center

    Args:
        input (Tensor): input Tensor. Dtype is in int8, int16, int32, int64, uint8, float16, float32, bfloat16.
        inplace (bool, optional): Whether to apply erasing inplace. Default: ``False``.

    Returns:
        Tensor, with the same dtype and shape as the `input`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If dtype of `input` is not one of: int8, int16, int32, int64, uint8, float16, float32, bfloat16.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> result = mint.nn.functional.relu6(x)
        >>> print(result)
        [[0. 4. 0.]
         [2. 0. 6.]]
    """
    if inplace:
        return hardtanh_(input, 0, 6)
    return hardtanh_op(input, 0, 6)


def binary_cross_entropy(input, target, weight=None, reduction='mean'):
    r"""
    Computes the binary cross entropy(Measure the difference information between two probability distributions) between
    predictive value `input` and target value `target`.

    Set `input` as :math:`x`, `target` as :math:`y`, output as :math:`\ell(x, y)`, the
    weight of nth batch of binary cross entropy is :math:`w_n`.
    Let,

    .. math::
        L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]

    In which, :math:`L` indicates the loss of all `batch_size`, :math:`l` indicates the loss of one `batch_size`,
    and :math:`n` indicates one `batch_size` in the :math:`1-N` range. Then,

    .. math::
        \ell(x, y) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    .. warning::
        The value of `input` must range from `0` to `l`.

    .. note::
        Currently, when the platform is Ascend, all gradient calculations are performed on NPU.

    Args:
        input (Tensor): The predictive value whose data type must be float16 or float32.
        target (Tensor): The target value which has the same shape and data type as `input`.
            And the data type is float16 or float32.
        weight (Tensor, optional): A rescaling weight applied to the loss of each batch element.
            Its shape must be able to broadcast to that of `input` and `target`.
            And it must have the same shape and data type as `input`. Default: ``None`` . If set to ``None`` ,
            the loss function
            will not consider any sample weights, and each sample will be treated as having equal importance
            when calculating the loss.
        reduction (str, optional): Apply specific reduction method to the output: ``'none'`` , ``'mean'`` ,
            ``'sum'`` . Default: ``'mean'`` .

            - ``'none'``: no reduction will be applied.
            - ``'mean'``: compute and return the weighted mean of elements in the output.
            - ``'sum'``: the output elements will be summed.

    Returns:
        Tensor or Scalar. Returns Tensor that has the same dtype and shape as `input` if `reduction` is ``'none'``.
        Otherwise, returns a scalar Tensor.

    Raises:
        TypeError: If `input`, `target` or `weight` is not a Tensor.
        TypeError: If dtype of `input`, `target` or `weight` (if given) is neither float16 nor float32.
        ValueError: If `reduction` is not one of ``'none'``, ``'mean'`` or ``'sum'``.
        ValueError: If shape of `target` is not the same as `input` or `weight` (if given).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([0.2, 0.7, 0.1]), mindspore.float32)
        >>> target = Tensor(np.array([0., 1., 0.]), mindspore.float32)
        >>> weight = Tensor(np.array([1, 2, 2]), mindspore.float32)
        >>> output = mint.nn.functional.binary_cross_entropy(input, target, weight)
        >>> print(output)
        0.38240486
    """
    return ops.function.binary_cross_entropy(input, target, weight, reduction)


def binary_cross_entropy_with_logits(input, target, weight=None, reduction='mean', pos_weight=None):
    r"""
    Adds sigmoid activation function to `input` as logits, and uses this logits to compute binary cross entropy
    between the logits and the target.
    Consistent with the function of :func:`mindspore.ops.binary_cross_entropy_with_logits` .

    Sets input `input` as :math:`X`, input `target` as :math:`Y`, input `weight` as :math:`W`, output as :math:`L`.
    Then,

    .. math::

        \begin{array}{ll} \\
            p_{ij} = sigmoid(X_{ij}) = \frac{1}{1 + e^{-X_{ij}}} \\
            L_{ij} = -[Y_{ij}log(p_{ij}) + (1 - Y_{ij})log(1 - p_{ij})]
        \end{array}

    :math:`i` indicates the :math:`i^{th}` sample, :math:`j` indicates the category. Then,

    .. math::
        \ell(x, y) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`\ell` indicates the method of calculating the loss. There are three methods:
    the first method is to provide the loss value directly,
    the second method is to calculate the average value of all losses,
    and the third method is to calculate the sum of all losses.

    This operator will multiply the output by the corresponding weight.
    The tensor :math:`weight` assigns different weights to each piece of data in the batch,
    and the tensor :math:`pos\_weight` adds corresponding weights to the positive examples of each category.

    In addition, it can trade off recall and precision by adding weights to positive examples.
    In the case of multi-label classification the loss can be described as:

    .. math::
        \begin{array}{ll} \\
            p_{ij,c} = sigmoid(X_{ij,c}) = \frac{1}{1 + e^{-X_{ij,c}}} \\
            L_{ij,c} = -[P_{c}Y_{ij,c} * log(p_{ij,c}) + (1 - Y_{ij,c})log(1 - p_{ij,c})]
        \end{array}

    where c is the class number (c>1 for multi-label binary classification, c=1 for single-label binary classification),
    n is the number of the sample in the batch and :math:`P_c` is the weight of the positive answer for the class c.
    :math:`P_c>1` increases the recall, :math:`P_c<1` increases the precision.

    Args:
        input (Tensor): Input `input` with shape :math:`(N, *)` where :math:`*` means, any number
          of additional dimensions. The data type must be float16, float32 or bfloat16(only Atlas A2 series products
          are supported).
        target (Tensor): Ground truth label, has the same shape as `input`.
          The data type must be float16, float32 or bfloat16(only Atlas A2 series products are supported).
        weight (Tensor, optional): A rescaling weight applied to the loss of each batch element. It can be
          broadcast to a tensor with shape of `input`. Data type must be float16, float32 or bfloat16(only
          Atlas A2 series products are supported).
          Default: ``None``, `weight` is a Tensor whose value is ``1``.
        reduction (str, optional): Apply specific reduction method to the output: ``'none'`` , ``'mean'`` ,
            ``'sum'`` . Default: ``'mean'`` .

            - ``'none'``: no reduction will be applied.
            - ``'mean'``: compute and return the weighted mean of elements in the output.
            - ``'sum'``: the output elements will be summed.
        pos_weight (Tensor, optional): A weight of positive examples. Must be a vector with length equal to the
          number of classes. It can be broadcast to a tensor with shape of `input`.
          Data type must be float16, float32 or bfloat16(only Atlas A2 series products are supported).
          Default: ``None``, it equals to `pos_weight` is a Tensor whose value is ``1``.

    Returns:
        Tensor or Scalar, if `reduction` is ``'none'``, it's a tensor with the same shape and type as input `input`.
        Otherwise, the output is a Scalar.

    Raises:
        TypeError: If input `input`, `target`, `weight`, `pos_weight` is not Tensor.
        TypeError: If data type of input `reduction` is not string.
        ValueError: If `weight` or `pos_weight` can not be broadcast to a tensor with shape of `input`.
        ValueError: If `reduction` is not one of ``'none'``, ``'mean'`` or ``'sum'``.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> input = Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]), mindspore.float32)
        >>> target = Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]), mindspore.float32)
        >>> weight = Tensor(np.array([1.0, 1.0, 1.0]), mindspore.float32)
        >>> pos_weight = Tensor(np.array([1.0, 1.0, 1.0]), mindspore.float32)
        >>> output = mint.nn.functional.binary_cross_entropy_with_logits(input, target, weight, 'mean', pos_weight)
        >>> print(output)
        0.3463612
    """
    return ops.function.binary_cross_entropy_with_logits(input, target, weight, pos_weight, reduction)


def one_hot(tensor, num_classes=-1):
    r"""
    Computes a one-hot tensor.

    The locations represented by tensor in `tensor` take value `1`, while all
    other locations take value `0`.

    Args:
        tensor (Tensor): A tensor of indices. Tensor of shape :math:`(X_0, \ldots, X_n)`.
            Data type must be int32 or int64. Dimension cannot be greater than 7.
        num_classes (int, optional): A scalar defining the depth of the one-hot dimension, default: ``-1``.

    Returns:
        Tensor, one-hot tensor.

    Raises:
        TypeError: If `num_classes` is not an int.
        TypeError: If dtype of `tensor` is not int32 or int64.
        ValueError: If `num_classes` is less than -1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> tensor = Tensor(np.array([0, 1, 2]), mindspore.int32)
        >>> num_classes = 3
        >>> output = mint.nn.functional.one_hot(tensor, num_classes)
        >>> print(output)
        [[1 0 0]
         [0 1 0]
         [0 0 1]]
    """
    return ops.function.array_func.one_hot_ext(tensor, num_classes)


def smooth_l1_loss(input, target, reduction='mean', beta=1.0):
    r"""
    Computes smooth L1 loss, a robust L1 loss.

    SmoothL1Loss is a Loss similar to MSELoss but less sensitive to outliers as described in the
    `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_ by Ross Girshick.

    Given two inputs :math:`x,\  y` of length :math:`N`, the SmoothL1Loss can be described
    as follows:

    .. math::
        L_{i} =
        \begin{cases}
        \frac{0.5 (x_i - y_i)^{2}}{\text{beta}}, & \text{if } |x_i - y_i| < \text{beta} \\
        |x_i - y_i| - 0.5 * \text{beta}, & \text{otherwise. }
        \end{cases}

    If `reduction` is not `none`, then:

    .. math::
        L =
        \begin{cases}
            \operatorname{mean}(L_{i}), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L_{i}),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    Here :math:`\text{beta}` controls the point where the loss function changes from quadratic to linear.
    :math:`\text{beta} \geq 0` , its default value is ``1.0`` . :math:`N` is the batch size.

    Note:
        - Arg `input` and `target` comply with the implicit type conversion rules to make the data types consistent.
          If they have different data types, the lower precision data type will be converted to relatively the
          highest precision data type.

    Args:
        input (Tensor): Tensor of shape :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
            Supported dtypes:

            - Ascend: float16, float32, bfloat16.

        target (Tensor): Ground truth data, tensor of shape :math:`(N, *)`, same shape as the `input`.
            Supported dtypes:

            - Ascend: float16, float32, bfloat16.

        reduction (str, optional): Apply specific reduction method to the output: ``'none'`` , ``'mean'`` ,
            ``'sum'`` . Default: ``'mean'`` .

            - ``'none'``: no reduction will be applied.
            - ``'mean'``: compute the mean of elements in the output.
            - ``'sum'``: the output elements will be summed.
        beta (number, optional): A parameter used to control the point where the function will change between
            L1 to L2 loss. The value should be greater than or equal to zero. Default: ``1.0`` .

    Returns:
        Tensor, the data type is the same as `input`.
        If `reduction` is ``'none'``, then output is a tensor with the same shape as `input`.
        Otherwise, the shape of output tensor is :math:`()`.

    Raises:
        TypeError: If `input` or `target` is not a Tensor.
        RuntimeError: If dtype of `input` or `target` is not one of float16, float32, bfloat16.
        ValueError: If shape of `input` is not the same as `target`.
        ValueError: If `reduction` is not one of ``'none'``, ``'mean'``, ``'sum'``.
        TypeError: If `beta` is not a float, int or bool.
        RuntimeError: If `beta` is less than 0.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([2, 2, 3]), mindspore.float32)
        >>> target = Tensor(np.array([2, 2, 2]), mindspore.float32)
        >>> beta = 1.0
        >>> reduction_1 = 'none'
        >>> output = ops.nn.functional.smooth_l1_loss(input, target, reduction_1, beta)
        >>> print(output)
        [0.  0.  0.5]
        >>> reduction_2 = 'mean'
        >>> output = ops.nn.functional.smooth_l1_loss(input, target, reduction_2, beta)
        >>> print(output)
        0.16666667
        >>> reduction_3 = 'sum'
        >>> output = ops.nn.functional.smooth_l1_loss(input, target, reduction_3, beta)
        >>> print(output)
        0.5
    """
    return ops.function.smooth_l1_loss(input, target, beta, reduction)


def normalize(input, p=2.0, dim=1, eps=1e-12):
    r"""
    Perform normalization of inputs over specified dimension

    For a tensor input of sizes :math:`(n_{0},..., n_{dim},..., n_{k})`, each :math:`n_{dim}` -element vector `v`
    along dimension `dim` is transformed as

    .. math::
        v=\frac{v}{\max(\left \| v \right \| _{p},\in )}

    With the default arguments it uses the Euclidean norm over vectors along dimension ``1`` for normalization.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): input tensor of any shape.
        p (float): the exponent value in the norm formulation. default: ``2``.
        dim (int): the dimension to reduce. default: ``1``.
        eps (float): small value to avoid division by zero. default: ``1e-12``.

    Returns:
        Tensor, shape and data type are the same as input.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> tensor = Tensor(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]), mindspore.float32)
        >>> output = mint.nn.functional.normalize(tensor)
        >>> print(output)
        [[0.0000 0.4472 0.8944]
         [0.4243 0.5657 0.7071]
         [0.4915 0.5735 0.6554]]
    """
    denom = broadcast_to(clamp(norm_ext(input, p, dim, keepdim=True), min=eps), input.shape)
    return input / denom


def upsample(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    r"""
    Samples `input` by the given `size` or `scale_factor`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Refer to :func:`mindspore.mint.nn.functional.interpolate` for more details.

    Supported Platforms:
        ``Ascend``
    """
    return interpolate(input, size, scale_factor, mode, align_corners)


def threshold(input, threshold, value, inplace=False):  # pylint: disable=W0621
    r"""
    Compute the Threshold activation function element-wise.

    The Threshold is defined as:

    .. math::
        y =
        \begin{cases}
        x, &\text{ if } x > \text{threshold} \\
        \text{value}, &\text{ otherwise }
        \end{cases}

    Args:
        input (Tensor): The input Tensor.
        threshold (Union[int, float]): The value of the threshold.
        value (Union[int, float]): The value to replace with when element is less than threshold.
        inplace (bool, optional): Whether to apply erasing inplace. Default: ``False``.

    Returns:
        Tensor, the same shape and data type as the input.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `threshold` is not a float or an int.
        TypeError: If `value` is not a float or an int.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> inputs = mindspore.Tensor([0.0, 2, 3], mindspore.float32)
        >>> outputs = mint.nn.functional.threshold(inputs, 1, 100)
        >>> print(outputs)
        [100.   2.   3.]
    """
    if inplace is True:
        return threshold_(input, threshold, value)
    return threshold_op(input, threshold, value)


def adaptive_avg_pool3d(input, output_size):
    r"""
    Performs 3D adaptive average pooling on a multi-plane input signal.
    That is, for any input size, the size of the specified output is :math:`(D, H, W)`.
    The number of output features is equal to the number of input planes.

    Suppose the last 3 dimension size of x is :math:`(D_{in}, H_{in}, W_{in})`, the last 3 dimension size of output is
    :math:`(D_{out}, H_{out}, W_{out})`.

    .. math::
        \begin{array}{ll} \\
            \forall \quad od \in [0, D_{out}-1], oh \in [0, H_{out}-1], ow \in [0, W_{out}-1] \\
            output[od,oh,ow] = \\
            \qquad mean(x[D_{istart}:D_{iend}+1,H_{istart}:H_{iend}+1,W_{istart}:W_{iend}+1]) \\
            where, \\
            \qquad D_{istart}= \left\lceil \frac{od * D_{in}}{D_{out}} \right\rceil \\
            \qquad D_{iend}=\left\lfloor \frac{(od+1)* D_{in}}{D_{out}} \right\rfloor \\
            \qquad H_{istart}=\left\lceil \frac{oh * H_{in}}{H_{out}} \right\rceil \\
            \qquad H_{iend}=\left\lfloor \frac{(oh+1) * H_{in}}{H_{out}} \right\rfloor \\
            \qquad W_{istart}=\left\lceil \frac{ow * W_{in}}{W_{out}} \right\rceil \\
            \qquad W_{iend}=\left\lfloor \frac{(ow+1) * W_{in}}{W_{out}} \right\rfloor
        \end{array}

    .. warning::
        For Ascend, it is only supported on Atlas A2 Training Series Products.

    Args:
        input (Tensor): The input of adaptive_avg_pool3d, which is a 4D or 5D Tensor.
        output_size (Union[int, tuple]): The target output size. `output_size` can be a tuple :math:`(D, H, W)`,
            or an int D for :math:`(D, D, D)`. :math:`D`, :math:`H` and :math:`W` can be int or None
            which means the output size is the same as that of the input.

    Returns:
        Tensor, with the same type as the `input`.

    Raises:
        TypeError: If `input` is not a Tensor.
        ValueError: If the dimension of `input` is not 4D or 5D.
        ValueError: If `output_size` value is not positive.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> # case 1: output_size=(3, 3, 4)
        >>> output_size=(3, 3, 4)
        >>> input_val = np.random.randn(4, 3, 5, 6, 7)
        >>> input = Tensor(input_val, mindspore.float32)
        >>> output = mint.nn.functional.adaptive_avg_pool3d(input, output_size)
        >>> print(output.shape)
        (4, 3, 3, 3, 4)
        >>> # case 2: output_size=4
        >>> output_size=5
        >>> input_val = np.random.randn(2, 3, 8, 6, 12)
        >>> input = Tensor(input_val, mindspore.float32)
        >>> output = mint.nn.functional.adaptive_avg_pool3d(input, output_size)
        >>> print(output.shape)
        (2, 3, 5, 5, 5)
        >>> # case 3: output_size=(None, 4, 5)
        >>> output_size=(None, 4, 5)
        >>> input_val = np.random.randn(4, 1, 9, 10, 8)
        >>> input = Tensor(input_val, mindspore.float32)
        >>> output = mint.nn.functional.adaptive_avg_pool3d(input, output_size)
        >>> print(output.shape)
        (4, 1, 9, 4, 5)
    """
    validator.check_value_type("output_size", output_size, [int, tuple, list], "adaptive_avg_pool3d")
    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)
    output_size = tuple(-1 if val is None else val for val in output_size)
    return adaptive_avg_pool3d_ext(input, output_size)


def adaptive_max_pool1d(input, output_size, return_indices=False):
    r"""
    Performs 1D adaptive max pooling on a multi-plane input signal.
    That is, for any input size, the size of the specified output is L.
    The number of output features is equal to the number of input features.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    .. note::
        Atlas training series products do not support backward propagation.

    Args:
        input (Tensor): The input of adaptive_max_pool1d, which is a 2D or 3D tensor,
            with float16, float32 or float64 data type.
        output_size (int): The target output feature size. `output_size` is an integer.
        return_indices (bool, optional): Whether to return the index of the maximum value. Default: ``False`` .

    Returns:
        Union(Tensor, tuple(Tensor, Tensor)).

        - If `return_indices` is False, output is a Tensor, with shape :math:`(N, C, L_{out})`. It has the same data
          type as `input`.
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
        >>> input = Tensor([[2,3],[3,4]],dtype=mindspore.float16)
        >>> output = mint.nn.functional.adaptive_max_pool1d(input, 3)
        >>> print(output)
        [[2.  3.  3. ]
         [3.  4.  4. ]]
    """
    if return_indices:
        return ops.auto_generate.gen_ops_prim.adaptive_max_pool1d_op(input, output_size)
    return ops.auto_generate.gen_ops_prim.adaptive_max_pool1d_op(input, output_size)[0]


__all__ = [
    'conv_transpose2d',
    'max_pool2d',
    # 1
    'binary_cross_entropy_with_logits',
    # 2

    # 3

    # 4
    "interpolate",
    # 5
    'pad',
    # 6
    'unfold',
    # 7
    'fold',
    # 8
    'layer_norm',
    # 9
    'upsample',
    # 10

    # 11
    'relu',
    'relu_',

    # 12

    # 13

    # 14
    'dropout',
    # 15
    'conv1d',
    'conv2d',
    # 16
    'log_softmax',
    # 17

    # 18
    'prelu',
    # 19
    'binary_cross_entropy',
    # 20
    'cross_entropy',
    # 21
    'conv3d',
    'nll_loss',
    # 22

    # 23

    # 24

    # 25

    # 26

    # 27

    # 28

    # 29

    # 30

    # 31
    'softmax',
    # 32

    # 33

    # 34
    'batch_norm',
    # 35

    # 36
    'gelu',
    # 37

    # 38
    'linear',
    # 39
    'group_norm',
    # 40

    # 41

    # 42

    # 43

    # 44
    'soft_margin_loss',
    # 45

    # 46
    'silu',
    # 47

    # 48

    # 49
    'sigmoid',
    'sigmoid_',
    # 50

    # 51

    # 52
    'embedding',
    # 53

    # 54
    'pixel_shuffle',
    # 55

    # 56

    # 57

    # 58

    # 59

    # 60

    # 61

    # 62

    # 63

    # 64
    'one_hot',
    # 65

    # 66

    # 67

    # 68

    # 69

    # 70

    # 71

    # 72

    # 73

    # 74

    # 75

    # 76

    # 77

    # 78

    # 79

    # 80

    # 81

    # 82

    # 83

    # 84

    # 85

    # 86

    # 87

    # 88
    'avg_pool3d',
    # 89
    'avg_pool1d',
    # 90
    'avg_pool2d',
    # 91
    'grid_sample',
    # 92
    'leaky_relu',
    # 93

    # 94
    'tanh',
    # 95

    # 96

    # 97

    # 98

    # 99

    # 100

    # 152
    'adaptive_avg_pool3d',
    # 254
    'max_unpool2d',
    # 256
    'threshold',
    'threshold_',

    # 288
    'adaptive_max_pool2d',

    # 312
    'normalize',

    # 323

    # 324
    'elu',
    'elu_',
    # 325

    #556
    'logsigmoid',

    # 257
    'adaptive_max_pool1d',
    # 258
    'mse_loss',
    # 259
    'adaptive_avg_pool1d',
    'adaptive_avg_pool2d',

    # 350

    # 393
    'dropout2d',
    # 421
    'flatten',
    # 536
    'glu',
    # 537
    'hardtanh',
    'hardtanh_',
    'relu6',
    # 548
    'kl_div',
    'cosine_embedding_loss',
]

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
"""mint nn functional."""
from __future__ import absolute_import
from mindspore.ops.extend import max_pool2d
from mindspore.ops.functional import (
    conv_transpose2d,
    grid_sample
)
# 1

# 2

# 3

# 4

# 5
from mindspore.ops.function.nn_func import pad_ext as pad
# 6
from mindspore.ops.auto_generate import unfold_ext as unfold
# 7
from mindspore.ops.auto_generate import fold_ext as fold
# 8
from mindspore.ops.functional import layer_norm
# 9
from mindspore.ops.function.nn_func import interpolate_ext as interpolate
# 10

# 11
from mindspore.ops.functional import relu
# 12

# 13

# 14
from mindspore.ops.function.nn_func import dropout_ext as dropout
# 15

# 16

# 17
from mindspore.ops.function.nn_func import binary_cross_entropy
# 18

# 19

# 20

# 21

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
from mindspore.ops.functional import gelu
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

# 45

# 46
from mindspore.ops.functional import silu
# 47

# 48

# 49
from mindspore.ops.functional import sigmoid
# 50

# 51

# 52
from mindspore.ops.functional import embedding
# 53

# 54

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
from mindspore.ops.extend import one_hot as one_hot_ext

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

# 89

# 90
from mindspore.ops.function.nn_func import avg_pool2d_ext as avg_pool2d
# 91

# 92
from mindspore.ops.extend import leaky_relu_ext as leaky_relu
# 93
from mindspore.ops.auto_generate import softplus_ext as softplus  # pylint: disable=W0611
# 94
from mindspore.ops.function.math_func import tanh
# 95

# 96

# 97

# 98

# 99

# 100
from mindspore.ops.function import binary_cross_entropy_with_logits as bce_with_logits
# 220
from mindspore.ops.function.nn_func import hardshrink # pylint: disable=W0611
# 221
from mindspore.ops.function.nn_func import hardsigmoid  # pylint: disable=W0611
# 222
from mindspore.ops.function.nn_func import hardswish  # pylint: disable=W0611
# 238
from mindspore.ops.extend import l1_loss_ext as l1_loss # pylint: disable=W0611
# 323

# 324
from mindspore.ops.auto_generate import elu_ext as elu
# 325



def binary_cross_entropy_with_logits(input, target, weight=None, reduction='mean', pos_weight=None):
    r"""
    Adds sigmoid activation function to `input` as logits, and uses this logits to compute binary cross entropy
    between the logits and the target.
    Consistent with the function of `mindspore.ops.binary_cross_entropy_with_logits` .

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
        >>> output = mint.nn.functional.binary_cross_entropy_with_logits(input, target, weight, pos_weight)
        >>> print(output)
        0.3463612
    """
    return bce_with_logits(input, target, weight, pos_weight, reduction)


def one_hot(tensor, num_classes=-1):
    r"""
    Computes a one-hot tensor.

    The locations represented by tensor in `tensor` take value `1`, while all
    other locations take value `0`.

    Args:
        tensor (Tensor): A tensor of indices. Tensor of shape :math:`(X_0, \ldots, X_n)`.
            Data type must be int32 or int64.
        num_classes (int): A scalar defining the depth of the one-hot dimension, default: ``-1``.

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
    return one_hot_ext(tensor, num_classes)

__all__ = [
    'conv_transpose2d',
    'max_pool2d',
    # 1
    'binary_cross_entropy_with_logits',
    # 2

    # 3

    # 4

    # 5
    'pad',
    # 6
    'unfold',
    # 7
    'fold',
    # 8
    'layer_norm',
    # 9
    'interpolate',
    # 10

    # 11
    'relu',
    # 12

    # 13

    # 14
    'dropout',
    # 15

    # 16

    # 17

    # 18

    # 19
    'binary_cross_entropy',
    # 20

    # 21

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

    # 45

    # 46
    'silu',
    # 47

    # 48

    # 49
    'sigmoid',
    # 50

    # 51

    # 52
    'embedding',
    # 53

    # 54

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

    # 89

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

    # 323

    # 324
    'elu',
    # 325
]

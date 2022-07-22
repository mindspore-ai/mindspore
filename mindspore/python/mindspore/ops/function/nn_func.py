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

from mindspore.ops.primitive import constexpr
from mindspore.ops import operations as P
from mindspore.ops.operations import nn_ops as NN_OPS
from mindspore.ops.operations import image_ops as IMG
import mindspore.common.dtype as mstype
from .math_func import logsumexp
from ...common.tensor import Tensor
from ..._c_expression import Tensor as Tensor_
from .._primitive_cache import _get_cache_prim
from ..._checkparam import Rel
from ..._checkparam import Validator as validator

slice_ = P.Slice()
fast_gelu_ = P.FastGeLU()
softsign_ = P.Softsign()
hardswish_ = P.HSwish()
mish_ = NN_OPS.Mish()
selu_ = NN_OPS.SeLU()


def adaptive_avg_pool2d(input_x, output_size):
    r"""
    2D adaptive average pooling for temporal data.

    This operator applies a 2D adaptive average pooling to an input signal composed of multiple input planes.
    That is, for any input size, the size of the specified output is H x W.
    The number of output features is equal to the number of input features.

    The input and output data format can be "NCHW" and "CHW". N is the batch size, C is the number of channels,
    H is the feature height, and W is the feature width.

    For adaptive average pooling for 2D:

    ..  math::
        \begin{align}
        h_{start} &= floor(i * H_{in} / H_{out})\\
        h_{end} &= ceil((i + 1) * H_{in} / H_{out})\\
        w_{start} &= floor(j * W_{in} / W_{out})\\
        w_{end} &= ceil((j + 1) * W_{in} / W_{out})\\
        Output(i,j) &= \frac{\sum Input[h_{start}:h_{end}, w_{start}:w_{end}]}{(h_{end}- h_{start})
        * (w_{end}- w_{start})}
        \end{align}

    Args:
        input_x (Tensor): The input of adaptive_avg_pool2d, which is a 3D or 4D tensor,
          with float16, float32 or float64 data type.
        output_size (Union[int, tuple]): The target output size is H x W.
            `ouput_size` can be a tuple consisted of int type H and W, or a single H for H x H, or None.
            If it is None, it means the output size is the same as the input size.

    Returns:
        Tensor, with the same type as the `input_x`.

        Shape of the output is `input_x_shape[:len(input_x_shape) - len(out_shape)] + out_shape`.

    .. math::

        out\_shape = \begin{cases}
        input\_x\_shape[-2] + output\_size[1], & \text{if output_size is (None, w);}\\
        output\_size[0] + input\_x\_shape[-1], & \text{if output_size is (h, None);}\\
        input\_x\_shape[-2:], & \text{if output_size is (None, None);}\\
        (h, h), & \text{if output_size is h;}\\
        (h, w), & \text{if output_size is (h, w)}
        \end{cases}

    Raises:
        ValueError: If `output_size` is a tuple and the length of `output_size` is not 2.
        TypeError: If `input_x` is not a Tensor.
        TypeError: If dtype of `input_x` is not float16, float32 or float64.
        ValueError: If the dimension of `input_x` is less than or equal to the dimension of `output_size`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> # case 1: output_size=(None, 2)
        >>> input_x = Tensor(np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        ...                            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        ...                            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]), mindspore.float32)
        >>> output = ops.adaptive_avg_pool2d(input_x, (None, 2))
        >>> print(output)
        [[[1.5 2.5]
          [4.5 5.5]
          [7.5 8.5]]
         [[1.5 2.5]
          [4.5 5.5]
          [7.5 8.5]]
         [[1.5 2.5]
          [4.5 5.5]
          [7.5 8.5]]]
        >>> # case 2: output_size=2
        >>> output = ops.adaptive_avg_pool2d(input_x, 2)
        >>> print(output)
        [[[3. 4.]
          [6. 7.]]
         [[3. 4.]
          [6. 7.]]
         [[3. 4.]
          [6. 7.]]]
        >>> # case 3: output_size=(1, 2)
        >>> output = ops.adaptive_avg_pool2d(input_x, (1, 2))
        >>> print(output)
        [[[4.5 5.5]]
         [[4.5 5.5]]
         [[4.5 5.5]]]
    """
    adaptive_avgpool2d_ = _get_cache_prim(P.AdaptiveAvgPool2D)(output_size)
    return adaptive_avgpool2d_(input_x)


def avg_pool2d(x, kernel_size=1, strides=1, pad_mode='valid', data_format='NCHW'):
    r"""
    Average pooling operation.

    Applies a 2D average pooling over an input Tensor which can be regarded as a composition of 2D input planes.
    Typically the input is of shape :math:`(N_{in}, C_{in}, H_{in}, W_{in})`, outputs regional average in the
    :math:`(H_{in}, W_{in})`-dimension. Given kernel size :math:`(k_{h}, k_{w})` and `strides` , the operation
    is as follows.

    .. math::
        \text{output}(N_i, C_j, h, w) = \frac{1}{k_{h} * k_{w}} \sum_{m=0}^{k_{h}-1} \sum_{n=0}^{k_{w}-1}
        \text{input}(N_i, C_j, strides[0] \times h + m, strides[1] \times w + n)

    .. warning::
        - Global pooling is supported.
        - For Ascend, the height of `kernel_size` and the weight of `kernel_size` are positive integers
          within the range [1, 255]. ksize_h * ksize_w < 256.
        - For Ascend, due to instruction restrictions, the values of 'strides_h' and 'strides_w' are
          positive integers within the range [1, 63].

    Args:
        x (Tensor): Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the average value.
            It is an int number that represents height and width of the kernel, or a tuple
            of two int numbers that represent height and width respectively. Default: 1.
        strides (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the height and width of movement are both strides, or a tuple of two int numbers that
            represent height and width of movement respectively. Default: 1.
        pad_mode (str): The optional value for pad mode, is 'same' or 'valid'.
            Default: 'valid'.

            - same: Adopts the way of completion. The height and width of the output will be the same as
              the input. The total number of padding will be calculated in horizontal and vertical
              directions and evenly distributed to top and bottom, left and right if possible.
              Otherwise, the last extra padding will be done from the bottom and the right side.

            - valid: Adopts the way of discarding. The possible largest height and width of output
              will be returned without padding. Extra pixels will be discarded.
        data_format (str): The format of input and output data. It should be 'NHWC' or 'NCHW'.
            Default: 'NCHW'.

    Returns:
        Tensor, with shape :math:`(N, C_{out}, H_{out}, W_{out})`.

    Raises:
        TypeError: If `kernel_size` or `strides` is neither int nor tuple.
        ValueError: If `kernel_size` or `strides` is less than 1.
        ValueError: If `pad_mode` is neither 'valid' nor 'same' with not case sensitive.
        ValueError: If `data_format` is neither 'NCHW' nor 'NHWC'.
        ValueError: If length of shape of `x` is not equal to 4.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), mindspore.float32)
        >>> output = ops.avg_pool2d(x, kernel_size=2, strides=1, pad_mode='VALID')
        >>> print(output)
        [[[[ 2.5   3.5   4.5]
           [ 6.5   7.5   8.5]]
          [[14.5  15.5  16.5]
           [18.5  19.5  20.5]]
          [[26.5  27.5  28.5]
           [30.5  31.5  32.5]]]]
    """
    _avg_pool = _get_cache_prim(P.AvgPool)(kernel_size, strides, pad_mode, data_format)
    return _avg_pool(x)


def adaptive_max_pool3d(x, output_size, return_indices=False):
    r"""
    Applies a 3D adaptive max pooling over an input signal composed of several input planes.

    The output is of size :math:`(D, H, W)`, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        x (Tensor): Tensor, with shape :math:`(C, D, H, W)` or :math:`(N, C, D, H, W)`, which support int8, int16,
            int32, int64, uint8, uint16, uint32, uint64, float16, float32 or float64 data type.
        output_size (Union[int, tuple]): The target output size. `ouput_size` can be a tuple :math:`(D, H, W)`,
            or an int D for :math:`(D, D, D)`. :math:`(D)`, :math:`(H)` and :math:`(W)` can be int or None
            which means the output size is the same as that of the input.
        return_indices (bool): If `return_indices` is True, the indices of max value would be output,
            else would not be output. Default: False.

    Returns:
        - **y** (Tensor) - Tensor, with the same number of dims and data type as the `x`.
        - **argmax** (Tensor) - Tensor, the indices of max value, which has the same shape as the
          `y` and it's data type is int32. It will output only when `return_indices` is True.

    Raises:
        TypeError: If `x` is not a Tensor.
        ValueError: If the dimensions number of `x` is not 4 or 5.
        TypeError: If dtype of `x` is not int8, int16, int32, int64, uint8, uint16, uint32, uint64,
                   float16, float32 or float64.
        ValueError: If `output_size` is neither an int nor a tuple with shape (3,).

    Supported Platforms:
        ``GPU``

    Examples:
        >>> x = Tensor(np.arange(0,36).reshape((1, 3, 3, 4)).astype(np.float32))
        >>> output_size = (1, 1, 2)
        >>> output = ops.adaptive_max_pool3d(x, output_size, True)
        >>> print(output[0].asnumpy())
        [[[[33. 35.]]]]
        >>> print(output[1].asnumpy())
        [[[[33 35]]]]
    """
    adaptive_max_pool3d_ = _get_cache_prim(NN_OPS.AdaptiveMaxPool3D)()
    output_size_ = Tensor(output_size, dtype=mstype.int32)
    out = adaptive_max_pool3d_(x, output_size_)
    output = out if return_indices else out[0]
    return output


def binary_cross_entropy_with_logits(logits, label, weight, pos_weight, reduction='mean'):
    r"""
    Adds sigmoid activation function to input `logits`, and uses the given logits to compute binary cross entropy
    between the logits and the label.

    Sets input logits as :math:`X`, input label as :math:`Y`, input weight as :math:`W`, output as :math:`L`. Then,

    .. math::

        \begin{array}{ll} \\
            p_{ij} = sigmoid(X_{ij}) = \frac{1}{1 + e^{-X_{ij}}} \\
            L_{ij} = -[Y_{ij} * log(p_{ij}) + (1 - Y_{ij})log(1 - p_{ij})]
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
    The tensor weight assigns different weights to each piece of data in the batch,
    and the tensor pos_weight adds corresponding weights to the positive examples of each category.

    In addition, it can trade off recall and precision by adding weights to positive examples.
    In the case of multi-label classification the loss can be described as:

    .. math::
        \begin{array}{ll} \\
            p_{ij,c} = sigmoid(X_{ij,c}) = \frac{1}{1 + e^{-X_{ij,c}}} \\
            L_{ij,c} = -[P_{c}Y_{ij,c} * log(p_{ij,c}) + (1 - Y_{ij,c})log(1 - p_{ij,c})]
        \end{array}

    where c is the class number (c>1 for multi-label binary classification, c=1 for single-label binary classification),
    n is the number of the sample in the batch and :math:`p_c` is the weight of the positive answer for the class c.
    :math:`p_c>1` increases the recall, :math:`p_c<1` increases the precision.

    Args:
        logits (Tensor): Input logits. Data type must be float16 or float32.
          Tensor of shape :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        label (Tensor): Ground truth label, has the same shape as `logits`.
          Data type must be float16 or float32.
        weight (Tensor): A rescaling weight applied to the loss of each batch element. It can be
          broadcast to a tensor with shape of `logits`. Data type must be float16 or float32.
        pos_weight (Tensor): A weight of positive examples. Must be a vector with length equal to the
          number of classes. It can be broadcast to a tensor with shape of `logits`.
          Data type must be float16 or float32.
        reduction (str): Type of reduction to be applied to loss. The optional values are 'mean', 'sum', and 'none',
             not case sensitive. If 'none', do not perform reduction. Default: 'mean'.
    Returns:
        Tensor or Scalar, if `reduction` is 'none', it's a tensor with the same shape and type as input `logits`.
        Otherwise, the output is a scalar.

    Raises:
        TypeError: If input `logits`, `label`, `weight`, `pos_weight` is not Tensor.
        TypeError: If data type of input `logits`, `label`, `weight`, `pos_weight` is neither float16 nor float32.
        TypeError: If data type of input `reduction` is not string.
        ValueError: If `weight` or `pos_weight` can not be broadcast to a tensor with shape of `logits`.
        ValueError: If `reduction` is not one of 'none', 'mean' or 'sum'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> logits = Tensor(np.array([[-0.8, 1.2, 0.7], [-0.1, -0.4, 0.7]]), mindspore.float32)
        >>> label = Tensor(np.array([[0.3, 0.8, 1.2], [-0.6, 0.1, 2.2]]), mindspore.float32)
        >>> weight = Tensor(np.array([1.0, 1.0, 1.0]), mindspore.float32)
        >>> pos_weight = Tensor(np.array([1.0, 1.0, 1.0]), mindspore.float32)
        >>> output = ops.bce_with_logits_loss(logits, label, weight, pos_weight, reduction)
        >>> print(output)
        0.3463612
    """

    bce_with_logits_loss_op = _get_cache_prim(NN_OPS.BCEWithLogitsLoss)(reduction)
    return bce_with_logits_loss_op(logits, label, weight, pos_weight)


def celu(x, alpha=1.0):
    r"""
    Computes celu (Continuously differentiable exponential linear units) of input tensors element-wise.

    .. math::

        \text{CeLU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))

    It returns :math:`\max(0,x) + \min(0, \alpha * (\exp(x/\alpha) - 1))` element-wise.

    The picture about celu looks like this `celu <https://arxiv.org/abs/1704.07483>`_.

    Args:
        x (Tensor): The input of celu with data type of float16 or float32.
        alpha (float): The :math:`\alpha` value for the Celu formulation. Default: 1.0

    Returns:
        Tensor, has the same data type and shape as the input.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Raises:
        TypeError: If `alpha` is not a float.
        ValueError: If `alpha` has the value of 0.
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Examples:
        >>> x = Tensor(np.array([-2.0, -1.0, 1.0, 2.0]), mindspore.float32)
        >>> output = ops.celu(x, alpha=1.0)
        >>> print(output)
        [-0.86466473 -0.63212055  1.          2.        ]
    """
    celu_op = _get_cache_prim(P.CeLU)(alpha)
    return celu_op(x)


def dropout2d(x, p=0.5):
    r"""
    During training, randomly zeroes some channels of the input tensor with probability `p`
    from a Bernoulli distribution(For a 4-dimensional tensor with a shape of :math:`NCHW`,
    the channel feature map refers to a 2-dimensional feature map with the shape of :math:`HW`).

    For example, the :math:`j\_th` channel of the :math:`i\_th` sample in the batched input is a to-be-processed
    `2D` tensor input[i,j].
    Each channel will be zeroed out independently on every forward call which based on Bernoulli distribution
    probability `p`.
    The parper `Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    <http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf>`_ mentioned this technology，And it is proved that
    it can effectively reduce over fitting and prevent neuronal coadaptation.
    For more details, refer to `Improving neural networks by preventing co-adaptation of feature detectors
    <https://arxiv.org/pdf/1207.0580.pdf>`_ .

    `dropout2d` can improve the independence between channel feature maps.

    Args:
        x (Tensor): A `4D` tensor with shape :math:`(N, C, H, W)`, where `N` is the batch size, `C` is the number
            of channels, `H` is the feature height, and `W` is the feature width. The data type must be int8,
            int16, int32, int64, float16, float32 or float64.
        p (float): The dropping probability of a channel, between 0 and 1, e.g. `p` = 0.8,
            which means dropping out 80% of channels. Default: 0.5.

    Returns:
        Tensor, output, with the same shape and data type as `x`.

        Tensor, mask, with the same shape as `x` and the data type is bool.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not int8, int16, int32, int64, float16, float32 or float64.
        TypeError: If the data type of `p` is not float.
        ValueError: If `p` is out of the range `[0.0, 1.0]`.
        ValueError: If `x` shape is not `4D`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.ones([2, 1, 2, 3]), mindspore.float32)
        >>> output, mask = dropout2d(input_x, 0.5)
        >>> print(output.shape)
        (2, 1, 2, 3)
    """
    dropout_2d_op = NN_OPS.Dropout2D(1.0 - p)
    return dropout_2d_op(x)


def dropout3d(x, p=0.5):
    r"""
    During training, randomly zeroes some channels of the input tensor
    with probability `p` from a Bernoulli distribution(For a 5-dimensional tensor
    with a shape of :math:`NCDHW`, the channel feature map refers to a 3-dimensional
    feature map with a shape of :math:`DHW`).

    For example, the :math:`j\_th` channel of the :math:`i\_th` sample in the batched input is a to-be-processed
    `3D` tensor input[i,j].
    Each channel will be zeroed out independently on every forward call which based on Bernoulli distribution
    probability `p`.

    `dropout3d` can improve the independence between channel feature maps.

    Args:
        x (Tensor): A `5D` tensor with shape :math:`(N, C, D, H, W)`, where `N` is the batch size, `C` is the number
            of channels, `D` is the feature depth, `H` is the feature height, and `W` is the feature width.
            The data type must be int8, int16, int32, int64, float16, float32 or float64.
        p (float): The dropping probability of a channel, between 0 and 1, e.g. `p` = 0.8,
            which means dropping out 80% of channels. Default: 0.5.

    Returns:
        Tensor, output, with the same shape and data type as `x`.

        Tensor, mask, with the same shape as `x` and the data type is bool.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not int8, int16, int32, int64, float16, float32 or float64.
        TypeError: If the data type of `p` is not float.
        ValueError: If `p` is out of the range `[0.0, 1.0]`.
        ValueError: If `x` shape is not 5D.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.ones([2, 1, 2, 1, 2]), mindspore.float32)
        >>> output, mask = dropout3d(input_x, 0.5)
        >>> print(output.shape)
        (2, 1, 2, 1, 2)
    """
    dropout_3d_op = NN_OPS.Dropout3D(1.0 - p)
    return dropout_3d_op(x)


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
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> output = ops.fast_gelu(x)
        >>> print(output)
        [[-1.5418735e-01  3.9921875e+00 -9.7473649e-06]
         [ 1.9375000e+00 -1.0052517e-03  8.9824219e+00]]
    """
    return fast_gelu_(x)


def kl_div(logits, labels, reduction='mean'):
    r"""
    Computes the Kullback-Leibler divergence between the logits and the labels.

    The updating formulas of KLDivLoss algorithm are as follows,

    .. math::
        L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = target_n \cdot (\log target_n - x_n)

    Then,

    .. math::
        \ell(x, target) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{batchmean}(L), & \text{if reduction} = \text{'batchmean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    where :math:`x` represents `logits`.
    :math:`target` represents `labels`.
    :math:`\ell(x, target)` represents `output`.

    Args:
        logits (Tensor): The input Tensor. The data type must be float16, float32 or float64.
        labels (Tensor): The label Tensor which has the same shape and data type as `logits`.
        reduction (str): Specifies the reduction to be applied to the output.
            Its value must be one of 'none', 'mean', 'batchmean' or 'sum'. Default: 'mean'.

    Returns:
        Tensor or Scalar, if `reduction` is 'none', then output is a tensor and has the same shape as `logits`.
        Otherwise it is a scalar.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Raises:
        TypeError: If `reduction` is not a str.
        TypeError: If neither `logits` nor `labels` is a Tensor.
        TypeError: If dtype of `logits` or `labels` is not float32.

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...
        ...     def construct(self, logits, labels):
        ...         result = mindspore.ops.functional.kl_div(logits, labels, 'mean')
        ...         return result
        ...
        >>> net = Net()
        >>> logits = Tensor(np.array([0.2, 0.7, 0.1]), mindspore.float32)
        >>> labels = Tensor(np.array([0., 1., 0.]), mindspore.float32)
        >>> output = net(logits, labels)
        >>> print(output)
        -0.23333333
    """
    if reduction == 'batchmean':
        kl_div_sum = P.KLDivLoss(reduction='sum')(logits, labels)
        batch_size = logits.shape[0]
        return kl_div_sum / batch_size

    return P.KLDivLoss(reduction=reduction)(logits, labels)


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
        Tensor, has the same data type and shape as the input `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If `lambd` is not a float.
        TypeError: If `x` is not a tensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Examples:
        >>> x = Tensor(np.array([[ 0.5,  1,  2.0], [0.0533,0.0776,-2.1233]]), mindspore.float32)
        >>> output = ops.hardshrink(x)
        >>> print(output)
        [[ 0.      1.      2.    ]
        [ 0.      0.     -2.1233]]
    """
    hshrink_op = _get_cache_prim(P.HShrink)(lambd)
    return hshrink_op(x)


def hardswish(x):
    r"""
    Hard swish activation function.

    Applies hswish-type activation element-wise. The input is a Tensor with any valid shape.

    Hard swish is defined as:

    .. math::

        \text{hswish}(x_{i}) = x_{i} * \frac{ReLU6(x_{i} + 3)}{6},

    where :math:`x_i` is an element of the input Tensor.

    Args:
        x (Tensor): The input to compute the Hard Swish with data type of float16 or float32.

    Returns:
        Tensor, has the same data type and shape as the input.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
        >>> output = ops.hardswish(x)
        >>> print(output)
        [-0.3333  -0.3333  0  1.666  0.6665]
    """
    return hardswish_(x)


@constexpr
def _check_interpolate_inputs(input_dims, roi, scales, sizes, coordinate_transformation_mode, mode,
                              prim_name):
    """Check input"""
    msg_prefix = f"For '{prim_name}', the"
    validator.check_value_type("coordinate_transformation_mode", coordinate_transformation_mode, [str], prim_name)
    support_coordinate_mode_list = ["align_corners", "half_pixel", "asymmetric"]
    if coordinate_transformation_mode not in support_coordinate_mode_list:
        raise TypeError(f"{msg_prefix} coordinate_transformation_mode must be in {support_coordinate_mode_list},"
                        " but got {coordinate_transformation_mode}")
    validator.check_value_type("mode", mode, [str], prim_name)
    if mode == "linear":
        validator.check_int(input_dims, 3, Rel.EQ, "input dims", prim_name)
    elif mode == "bilinear":
        validator.check_int(input_dims, 4, Rel.EQ, "input dims", prim_name)
    else:
        raise TypeError(f"{msg_prefix} mode must be 'linear' or 'bilinear', but got {mode}")

    if sizes is None and scales is None:
        raise ValueError(f"{msg_prefix} 'sizes' and 'scale' both none.")
    if sizes is not None and scales is not None:
        raise ValueError(f"{msg_prefix} 'sizes' and 'scale' both not none.")
    if sizes is not None:
        if not isinstance(sizes, tuple):
            raise TypeError(
                f"{msg_prefix} 'sizes' must be tuple or None, but got {type(sizes).__name__}.")
        for item in sizes:
            validator.check_positive_int(item, 'sizes item', prim_name)
            validator.check_value_type("sizes item", item, int, prim_name)
        validator.check_int(len(sizes), input_dims - 2, Rel.EQ, "sizes", prim_name)
        return
    if not isinstance(scales, tuple):
        raise TypeError(
            f"{msg_prefix} 'scales' must be tuple or None, but got {type(scales).__name__}.")
    for item in scales:
        validator.check_positive_float(item, 'scales item', prim_name)
        validator.check_value_type("scales item", item, float, prim_name)
    scales_dims = len(scales)
    validator.check_int(scales_dims, input_dims, Rel.EQ, "scales dims", prim_name)
    validator.check_float(scales[0], 1.0, Rel.EQ, "scales[0]", prim_name)
    validator.check_float(scales[1], 1.0, Rel.EQ, "scales[1]", prim_name)


def _interpolate_output_shape(shape, scales, sizes, mode):
    """calculate output shape"""
    if sizes is not None:
        if mode == "bilinear":
            return sizes
        return Tensor(sizes)
    ret = ()
    for i in range(2, len(shape)):
        ret = ret + (int(scales[i] * shape[i]),)
    if mode == "bilinear":
        return ret
    return Tensor(ret)


def interpolate(x, roi=None, scales=None, sizes=None, coordinate_transformation_mode="align_corners", mode="linear"):
    r"""
    Using the interpolate method specified by `mode` resize the input tensor `x`.

    .. warning::
        - This is an experimental prototype that is subject to change.
        - The `roi` is reserved interface for 'crop_and_resize' coordinate transformation mode,
          which is not support now.
        - The Ascend platforms is currently not supported when `mode` is "linear".
        - The 'half_pixel' coordinate_transformation_mode is currently not supported on CPU device
          when mode is "bilinear".

    Args:
        x (Tensor): a tensor which to resize. `x` is a 3-D tensor when `mode` is "linear". `x` is a 4-D tensor when
            `mode` is "bilinear".
        roi (tuple[float], optional): a tuple of float. Only takes effect when attr coordinate_transformation_mode is
            'crop_and_resize'.
        scales (tuple[float], optional): a tuple of float. Describe the scale along each dimension.
            Its length is the same as that of shape of `x`. The numbers in `scales` must all be positive. Only one of
            `scales` and `sizes` can be specified.
        sizes (tuple[int], optional): a tuple of int, describes the shape of the output tensor. The numbers in `sizes`
            must all be positive. Only one of `scales` and `sizes` can be specified.  If `sizes` is specified, then set
            `scales` to 'None' in this operator's input list. It is 1 int elements :math:`(new\_width,)` when `mode`
            is "linear". It is 2 int elements :math:`(new\_height, new\_width)` when `mode` is "bilinear".
        coordinate_transformation_mode (string): Default is 'align_corners'. Describes how to transform the coordinate
            in the resized tensor to the coordinate in the original tensor. Other optional: 'half_pixel', 'asymmetric'.
            For example, we want to resize the original tensor along axis x. Let's denote `new_i` as the i-th coordinate
            of the resized tensor along axis x, `old_i` as the coordinate of the original tensor along axis x,
            `new_length` as the length of the resized tensor along axis x, `old_length` as the length of the original
            tensor along axis x. We compute the `old_i` via the following formula:

            .. code-block::

                old_i = new_length != 1 ? new_i * (old_length - 1) / (new_length - 1) : 0  # if set to 'align_corners'

                old_i = new_length > 1 ? (new_x + 0.5) * old_length / new_length - 0.5 : 0  # if set to 'half_pixel'

                old_i = new_length != 0 ? new_i * old_length / new_length : 0  # if set to 'asymmetric'

        mode (string): The method used to interpolate: 'linear' | 'bilinear'. Default is 'linear'.

    Returns:
        Resized tensor, with the same data type as input `x`.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If the data type of `x` is not supported.
        TypeError: If `scales` is not a float tuple.
        ValueError: If not all numbers in `scales` are positive.
        TypeError: If `sizes` is not an int tuple.
        ValueError: If not all numbers in `sizes` are positive.
        TypeError: If `coordinate_transformation_mode` is not a string.
        ValueError: If `coordinate_transformation_mode` is not in the support list.
        TypeError: If `mode` is not a string.
        ValueError: If `mode` is not in the support list.

    Examples:
        >>> x = Tensor([[[1, 2, 3], [4, 5, 6]]], mindspore.float32)
        >>> output = ops.interpolate(x, None, None, (6,), "align_corners")
        >>> print(output)
        [[[1. 1.4 1.8 2.2 2.6 3.]
          [4. 4.4 4.8 5.2 5.6 6.]]]
        >>>
        >>> x = Tensor([[[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]]], mindspore.float32)
        >>> output = ops.interpolate(x, None, None, (5, 5), "asymmetric", "bilinear")
        >>> print(output)
        [[[[1. 2. 3. 4. 5.]
           [1. 2. 3. 4. 5.]
           [1. 2. 3. 4. 5.]
           [1. 2. 3. 4. 5.]
           [1. 2. 3. 4. 5.]]]]
    """
    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError("For interpolate, the input x must be tensor")
    input_shape = x.shape
    input_dims = len(input_shape)
    _check_interpolate_inputs(input_dims, roi, scales, sizes, coordinate_transformation_mode, mode,
                              "interpolate")
    output_size = _interpolate_output_shape(input_shape, scales, sizes, mode)

    if mode == "linear":
        resize_linear_inner = _get_cache_prim(IMG.ResizeLinear1D)(
            coordinate_transformation_mode=coordinate_transformation_mode)
        return resize_linear_inner(x, output_size)
    if mode == "bilinear":
        align_corners = False
        half_pixel_centers = False
        if coordinate_transformation_mode == "align_corners":
            align_corners = True
        elif coordinate_transformation_mode == "half_pixel":
            half_pixel_centers = True
        resize_bilinear_inner = _get_cache_prim(IMG.ResizeBilinearV2)(align_corners, half_pixel_centers)
        return resize_bilinear_inner(x, output_size)

    raise TypeError(
        "Input Error: For interpolate,  {} mode is not support now".format(mode))


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


def soft_shrink(x, lambd=0.5):
    r"""
    Applies the SoftShrink function element-wise.

    .. math::
        \text{SoftShrink}(x) =
        \begin{cases}
        x - \lambda, & \text{ if } x > \lambda \\
        x + \lambda, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        x (Tensor): The input of soft shrink with data type of float16 or float32.
        lambd(float): The :math:`\lambda` must be no less than zero. Default: 0.5.

    Outputs:
        Tensor, has the same shape and data type as `x`.

    Raises:
        TypeError: If lambd is not a float.
        TypeError: If input_x is not a Tensor.
        TypeError: If dtype of input_x is neither float16 nor float32.
        ValueError: If lambd is less than 0.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> import numpy as np
        >>> x = Tensor(np.array([[ 0.5297,  0.7871,  1.1754], [ 0.7836,  0.6218, -1.1542]]), mindspore.float32)
        >>> output = ops.soft_shrink(x)
        >>> print(output)
        [[ 0.02979  0.287    0.676  ]
         [ 0.2837   0.1216  -0.6543 ]]
    """
    soft_shrink_op = _get_cache_prim(P.SoftShrink)(lambd)
    return soft_shrink_op(x)


def selu(input_x):
    r"""
    Activation function SeLU (Scaled exponential Linear Unit).

    The activation function is defined as:

    .. math::
        E_{i} =
        scale *
        \begin{cases}
        x_{i}, &\text{if } x_{i} \geq 0; \cr
        \text{alpha} * (\exp(x_i) - 1), &\text{otherwise.}
        \end{cases}

    where :math:`alpha` and :math:`scale` are pre-defined constants(:math:`alpha=1.67326324`
    and :math:`scale=1.05070098`).

    See more details in `Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515>`_.

    Args:
        input_x (Tensor): Tensor of any dimension, the data type is float16 or float32.

    Returns:
        Tensor, with the same type and shape as the `input_x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If dtype of `input_x` is neither float16 nor float32.

    Examples:
        >>> input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> output = ops.selu(input_x)
        >>> print(output)
        [[-1.1113307 4.202804 -1.7575096]
        [ 2.101402 -1.7462534 9.456309 ]]
    """
    return selu_(input_x)


def deformable_conv2d(x, weight, offsets, kernel_size, strides, padding, bias=None, dilations=(1, 1, 1, 1), groups=1,
                      deformable_groups=1, modulated=True):
    r"""
    Given 4D tensor inputs `x`, `weight` and `offsets`, compute a 2D deformable convolution. The deformable convolution
    operation can be expressed as follow:

    Deformable Convolution v1:

    .. math::
        y(p)=\sum_{k=1}^{K}w_{k}\cdot x(p+p_{k}+\Delta{p_{k}})

    Deformable Convolution v2:

    .. math::
        y(p)=\sum_{k=1}^{K}w_{k}\cdot x(p+p_{k}+\Delta{p_{k}})\cdot \Delta{m_{k}}

    Where :math:`\Delta{p_{k}}` and :math:`\Delta{m_{k}}` are the learnable offset and modulation scalar for the k-th
    location. For details, please refer to `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168>`_ and `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`_.

    Args:
        x (Tensor): A 4D tensor of input image. With the format "NCHW",
            the shape is :math:`(N, C_{in}, H_{in}, W_{in})`. Dtype: float16 or float32.
        weight (Tensor): A 4D tensor of learnable filters. Must have the same type as `x`.
            The shape is :math:`(C_{out}, C_{in} / groups, H_{f}, W_{f})`.
        offsets (Tensor): A 4D tensor of x-y coordinates offset and mask. With the format "NCHW",
            the shape is :math:`(batch, 3 * deformable\_groups * H_{f} * W_{f}, H_{out}, W_{out})`. Note the C dimension
            is stored in the order of (offset_x, offset_y, mask). Must have the same type as `x`.
        kernel_size (tuple[int]): A tuple of 2 integers. The size of kernel.
        strides (tuple[int]): A tuple of 4 integers. The stride of the sliding window for each dimension of
            input. The dimension order is interpreted according to the data format of `x`. The N and C dimensions must
            be set to 1.
        padding (tuple[int]): A tuple of 4 integers. The number of pixels to add to each (top, bottom, left,
            right) side of the input.
        bias (Tensor, Optional): An 1D tensor of additive biases to the filter outputs.
            The shape is :math:`(C_{out})`. Defaults to None.
        dilations (tuple[int], Optional): A tuple of 4 integers. The dilation factor for each dimension of input. The
            dimension order is interpreted according to the data format of `x`. The N and C dimensions must be set
            to 1. Defaults to (1, 1, 1, 1).
        groups (int, Optional): An integer of type int32. The number of blocked connections from input channels
            to output channels. In_channels and out_channels must both be divisible by `groups`. Defaults to 1.
        deformable_groups (int, Optional): An integer of type int32. The number of deformable group partitions.
            In_channels must be divisible by `deformable_groups`. Defaults to 1.
        modulated (bool, Optional): Specifies version of DeformableConv2D, True means v2, False means v1, currently
            only supports v2. Defaults to True.

    Returns:
        Tensor, A 4D Tensor of output feature map. With the same type as `x`. With the format "NCHW",
        the shape is :math:`(N, C_{out}, H_{out}, W_{out})`.

        .. math::
            \begin{array}{ll} \\
                H_{out} = \left \lfloor{\frac{H_{in} + padding[0] + padding[1] - (H_{f} - 1) \times
                \text{dilations[2]} - 1 }{\text{stride[0]}} + 1} \right \rfloor \\
                W_{out} = \left \lfloor{\frac{W_{in} + padding[2] + padding[3] - (W_{f} - 1) \times
                \text{dilations[3]} - 1 }{\text{stride[1]}} + 1} \right \rfloor \\
            \end{array}

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If `strides`, `padding`, `kernel_size` or `dilations` is not a tuple with integer elements.
        TypeError: If `modulated` is not a bool.
        ValueError: If the tuple size of `strides`, `padding`, `kernel_size` or `dilations` is not expected.
        ValueError: The N or C dimensions of 'strides' or `dilations` is not set to 1.
        ValueError: If `modulated` is not set to True.

    Note:
        - This is an experimental interface that is subject to change or deletion.
        - For Ascend platform, only AI-CORE kernel is implemented, which has the following limitations:

          - :math:`C_{in}` cannot be divisible by 8 is not supported, e.g. `x` is :math:`(N, 2, H_{in}, W_{in})`.
          - `deformable_groups` must equal to 1.
          - `offsets` value is float which does not contain a decimal part is not supported, e.g. `offsets` is assigned
            with "numpy.ones()".
          - `kernel_size` should meet the requirement::math:`3 * kernel\_size[0] * kernel\_size[1] > 8`.

    Examples:
        >>> x = Tensor(np.ones((4, 3, 10, 10)), mstype.float32)
        >>> kh, kw = 3, 3
        >>> weight = Tensor(np.ones((5, 3, kh, kw)), mstype.float32)
        >>> offsets = Tensor(np.ones((4, 3 * kh * kw, 8, 8)), mstype.float32)
        >>> output = ops.deformable_conv2d(x, weight, offsets, (kh, kw), (1, 1, 1, 1), (0, 0, 0, 0))
        >>> print(output.shape)
        (4, 5, 8, 8)
    """
    deformable_offsets = _get_cache_prim(NN_OPS.DeformableOffsets)(strides, padding, kernel_size, dilations, "NCHW",
                                                                   deformable_groups,
                                                                   modulated)
    fm_offset = deformable_offsets(x, offsets)

    weight_shape = weight.shape
    out_channel = weight_shape[0]
    strides_conv = (kernel_size[0], kernel_size[1])
    conv = _get_cache_prim(P.Conv2D)(out_channel, kernel_size, 1, "valid", 0, strides_conv, 1, groups)
    bias_add = _get_cache_prim(P.BiasAdd)()

    output = conv(fm_offset, weight)
    if bias is not None:
        output = bias_add(output, bias)
    return output


def pdist(x, p=2.0):
    r"""
    Computes the p-norm distance between each pair of row vectors in the input. If `x` is a 2D Tensor of
    shape :math:`(N, M)`, then `output` must be a 1D Tensor of shape :math:`(N * (N - 1) / 2,)`. If `x` is a
    Tensor of shape :math:`(*B, N, M)`, then `output` must be a Tensor of shape :math:`(*B, N * (N - 1) / 2)`.

    .. math::
        y[n] = \sqrt[p]{{\mid x_{i} - x_{j} \mid}^p}

    where :math:`x_{i}, x_{j}` are two different row vectors in the input.

    Args:
        x (Tensor): Input tensor of shape :math:`(*B, N, M)`. :math:`*B` is batch size, one-dim or multi-dim.
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
    pdist_ = _get_cache_prim(NN_OPS.Pdist)(p=p)
    return pdist_(x)


def pad(input_x, paddings):
    r"""
    Pads the input tensor according to the paddings.

    The formula to calculate the shape of the output tensor is as follows,

    .. math::
        \begin{aligned}
            &\text{ input_x_shape} = (N_{1},N_{2},...,N_{n}) \\
            &\begin{aligned}
                \text{output_shape = }(&N_{1}+paddings[0,0]+paddings[0,1], \\
                                 & N_{2}+paddings[1,0]+paddings[1,1], \\
                                 &... , \\
                                 & N_{n}+paddings[n-1,0]+paddings[n-1,1])
            \end{aligned}
        \end{aligned}

    Args:
        input_x (Tensor): Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of additional dimensions.
        paddings (tuple): The shape of parameter `paddings` is (N, 2). N is the rank of input data. All elements of
            paddings are int type. For the input in `D` th dimension, paddings[D, 0] indicates how many sizes to be
            extended(if this value > 0) or clipped(if this value < 0) ahead of the input tensor in the `D` th
            dimension, and paddings[D, 1] indicates how many sizes to be extended(if this value > 0) or
            clipped(if this value < 0) behind the input tensor in the `D` th dimension.

    Returns:
        Tensor, the tensor after padding.

    Raises:
        TypeError: If `paddings` is not a tuple.
        TypeError: If `input_x` is not a Tensor.
        ValueError: If shape of `paddings` is not :math:`(N, 2)`.
        ValueError: If paddings.size is not equal to 2 * len(input_x).
        ValueError: If the calculated output shape contains zero or negative dimension.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> paddings = ((1, 2), (2, 1))
        >>> output = ops.pad(input_x, paddings)
        >>> print(output)
        [[ 0.   0.   0.   0.   0.   0. ]
         [ 0.   0.  -0.1  0.3  3.6  0. ]
         [ 0.   0.   0.4  0.5 -3.2  0. ]
         [ 0.   0.   0.   0.   0.   0. ]
         [ 0.   0.   0.   0.   0.   0. ]]
    """
    if not isinstance(input_x, Tensor):
        raise TypeError(f"For 'pad', the type of 'input_x' must be Tensor, but got {type(input_x)}.")
    if not isinstance(paddings, tuple):
        raise TypeError(f"For 'pad', the type of 'paddings' must be tuple, but got {type(paddings)}.")
    for _, pd in enumerate(paddings):
        if not isinstance(pd, (list, tuple)) or len(pd) != 2 or not isinstance(pd[0], int) or \
                not isinstance(pd[1], int):
            raise TypeError(f"For 'pad', each element in 'paddings' must be a list or tuple of 2 int, but got {pd}.")
    x_shape = input_x.shape
    if len(x_shape) != len(paddings):
        raise ValueError(f"For 'pad', the size of paddings must be 2 * {len(x_shape)}, but got {2 * len(paddings)}")
    pad_all_non_negative = True
    pad_all_non_positive = True
    slice_begin = []
    slice_size = []
    non_negative_padding = []
    for i, pd in enumerate(paddings):
        sz = x_shape[i] + pd[0]
        if sz <= 0:
            raise ValueError(f"For 'pad', input_x_shape[{i}] + paddings[{i}, 0] is {sz}, which is <= 0 and causes "
                             f"the output shape invalid.")
        sz = sz + pd[1]
        if sz <= 0:
            raise ValueError(f"For 'pad', input_x_shape[{i}] + paddings[{i}, 0] + paddings[{i}, 1] is {sz}, which is "
                             f"<= 0 and causes the output shape invalid.")
        slice_size.append(sz)
        if pd[0] < 0:
            slice_begin.append(abs(pd[0]))
        else:
            slice_begin.append(0)
        if pd[0] < 0 or pd[1] < 0:
            pad_all_non_negative = False
        if pd[0] > 0 or pd[1] > 0:
            pad_all_non_positive = False
        non_negative_padding.append((max(0, pd[0]), max(0, pd[1])))
    if pad_all_non_negative:
        _pad = _get_cache_prim(P.Pad)(paddings)
        return _pad(input_x)
    if pad_all_non_positive:
        return slice_(input_x, slice_begin, slice_size)
    _pad = _get_cache_prim(P.Pad)(tuple(non_negative_padding))
    out = _pad(input_x)
    return slice_(out, slice_begin, slice_size)


def _innner_log_softmax(inputs, axis):
    """inner implementation of log_softmax, since the LogSoftmaxGrad op do not support inputs > 2d"""
    return inputs - logsumexp(inputs, axis, True)


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
              \text{if reduction} = \text{'mean',}\\
              \sum_{n=1}^N l_n,  &
              \text{if reduction} = \text{'sum'.}
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
              \text{if reduction} = \text{'mean',}\\
              \sum_{n=1}^N l_n,  &
              \text{if reduction} = \text{'sum'.}
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
        reduction (str):  Apply specific reduction method to the output: 'none', 'mean', or 'sum'.
            Default: 'mean'.
        label_smoothing (float): Label smoothing values, a regularization tool used to prevent the model
            from overfitting when calculating Loss. The value range is [0.0, 1.0]. Default value: 0.0.

    Returns:
        Tensor, the computed loss value.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:

        >>> # Case 1: Indices labels
        >>> inputs = mindspore.Tensor(np.random.randn(3, 5), mindspore.float32)
        >>> target = mindspore.Tensor(np.array([1, 0, 4]), mindspore.int32)
        >>> output = ops.cross_entropy(inputs, target)
        >>> # Case 2: Probability labels
        >>> inputs = mindspore.Tensor(np.random.randn(3, 5), mindspore.float32)
        >>> target = mindspore.Tensor(np.random.randn(3, 5), mindspore.float32)
        >>> output = ops.cross_entropy(inputs, target)
    """
    class_dim = 0 if inputs.ndim == 1 else 1
    if inputs.size == target.size:
        return _cross_entropy(inputs, target, class_dim, weight, reduction, label_smoothing)
    return nll_loss(_innner_log_softmax(inputs, class_dim), target, weight, ignore_index, reduction, label_smoothing)


def _cross_entropy(inputs, target, target_dim, weight=None, reduction='mean', label_smoothing=0.0):
    """cross entropy inner function"""
    _ones_like = _get_cache_prim(P.OnesLike)()

    class_dim = 0 if inputs.ndim == 1 else 1
    n_classes = inputs.shape[class_dim]
    inputs = _innner_log_softmax(inputs, class_dim)
    if label_smoothing > 0.0:
        target = target * (1 - label_smoothing) + label_smoothing / n_classes

    if weight is None:
        weight = _ones_like(inputs)

    if reduction == 'mean':
        return -(inputs * target * weight).sum() / (inputs.size / n_classes)
    if reduction == 'sum':
        return -(inputs * target * weight).sum()
    return -(inputs * target * weight).sum(class_dim)


def nll_loss(inputs, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
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
        \sum_{n=1}^{N} \frac{1}{\sum_{n=1}^{N} w_{t n}} l_{n}, & \text { if reduction }=\text { 'mean', } \\
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
        reduction (str):  Apply specific reduction method to the output: 'none', 'mean', or 'sum'.
            Default: 'mean'.
        label_smoothing (float): Label smoothing values, a regularization tool used to prevent the model
            from overfitting when calculating Loss. The value range is [0.0, 1.0]. Default value: 0.0.

    Outputs:
        Tensor, the computed loss value.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:

        >>> inputs = mindspore.Tensor(np.random.randn(3, 5), mindspore.float32)
        >>> target = mindspore.Tensor(np.array([1, 0, 4]), mindspore.int32)
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
    _neg = _get_cache_prim(P.Neg)()
    _gather_d = _get_cache_prim(P.GatherD)()
    _gather = _get_cache_prim(P.Gather)()
    _ones_like = _get_cache_prim(P.OnesLike)()
    _equal = _get_cache_prim(P.Equal)()

    if target.ndim == inputs.ndim - 1:
        target = target.expand_dims(target_dim)
    loss = _neg(_gather_d(inputs, target_dim, target))
    smooth_loss = _neg(inputs.sum(axis=target_dim, keepdims=True))
    if weight is not None:
        loss_weights = _gather(weight, target, 0)
        loss = loss * loss_weights
    else:
        loss_weights = _ones_like(loss)
    if ignore_index is not None:
        non_pad_mask = _equal(target, ignore_index)
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


def smooth_l1_loss(logits, labels, beta=1.0, reduction='none'):
    r"""
    Computes smooth L1 loss, a robust L1 loss.

    SmoothL1Loss is a Loss similar to MSELoss but less sensitive to outliers as described in the
    `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_ by Ross Girshick.

    Given two input :math:`x,\  y` of length :math:`N`, the unreduced SmoothL1Loss can be described
    as follows:

    .. math::
        L_{i} =
        \begin{cases}
        \frac{0.5 (x_i - y_i)^{2}}{\text{beta}}, & \text{if } |x_i - y_i| < \text{beta} \\
        |x_i - y_i| - 0.5 \text{beta}, & \text{otherwise. }
        \end{cases}

    If `reduction` is not `none`, then:

    .. math::
        L =
        \begin{cases}
            \operatorname{mean}(L_{i}), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L_{i}),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    Here :math:`\text{beta}` controls the point where the loss function changes from quadratic to linear.
    Its default value is 1.0. :math:`N` is the batch size.

    Note:
        For Ascend platform, the 'reduction' is not support set to 'sum' or 'mean' for now.

    Args:
        logits (Tensor): Tensor of shape :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        labels (Tensor): Ground truth data, tensor of shape :math:`(N, *)`, same shape and dtype as the `logits`.
        beta (float): A parameter used to control the point where the function will change from
            quadratic to linear. Default: 1.0.
        reduction (str): Apply specific reduction method to the output: 'none', 'mean' or 'sum'. Default: 'none'.

    Returns:
        Tensor, if `reduction` is 'none', then output is a tensor with the same shape as `logits`.
        Otherwise the shape of output tensor is `(1,)`.

    Raises:
        TypeError: If `beta` is not a float.
        ValueError: If `reduction` is not one of 'none', 'mean', 'sum'.
        TypeError: If dtype of `logits` or `labels` is neither float16 nor float32.
        ValueError: If `beta` is less than or equal to 0.
        ValueError: If shape of `logits` is not the same as `labels`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> loss = ops.SmoothL1Loss()
        >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> labels = Tensor(np.array([1, 2, 2]), mindspore.float32)
        >>> output = loss(logits, labels)
        >>> print(output)
        [0.  0.  0.5]
    """

    return P.SmoothL1Loss(beta, reduction)(logits, labels)


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
    _in_topk = _get_cache_prim(P.InTopK)(k)
    return _in_topk(x1, x2)


def log_softmax(logits, axis=-1):
    r"""
    Log Softmax activation function.

    Applies the Log Softmax function to the input tensor on the specified axis.
    Supposes a slice in the given axis, :math:`x` for each element :math:`x_i`,
    the Log Softmax function is shown as follows:

    .. math::
        \text{output}(x_i) = \log \left(\frac{\exp(x_i)} {\sum_{j = 0}^{N-1}\exp(x_j)}\right),

    where :math:`N` is the length of the Tensor.

    Args:
        logits (Tensor): Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
          additional dimensions, with float16 or float32 data type.
        axis (int): The axis to perform the Log softmax operation. Default: -1.

    Outputs:
        Tensor, with the same type and shape as the logits.

    Raises:
        TypeError: If `axis` is not an int.
        TypeError: If dtype of `logits` is neither float16 nor float32.
        ValueError: If `axis` is not in range [-len(logits.shape), len(logits.shape)).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> logits = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
        >>> output = ops.log_softmax(logits)
        >>> print(output)
        [-4.4519143 -3.4519143 -2.4519143 -1.4519144 -0.4519144]
    """
    _log_softmax = _get_cache_prim(P.LogSoftmax)(axis)
    return _log_softmax(logits)


def lrn(x, depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, norm_region="ACROSS_CHANNELS"):
    r"""
    Local Response Normalization.

    .. math::

        b_{c} = a_{c}\left(k + \frac{\alpha}{n}
        \sum_{c'=\max(0, c-n/2)}^{\min(N-1,c+n/2)}a_{c'}^2\right)^{-\beta}

    where the :math:`a_{c}` indicates the specific value of the pixel corresponding to c in feature map;
    where the :math:`n/2` indicates the `depth_radius`; where the :math:`k` indicates the `bias`;
    where the :math:`\alpha` indicates the `alpha`; where the :math:`\beta` indicates the `beta`.

    Args:
        depth_radius (int): Half-width of the 1-D normalization window with the shape of 0-D. Default: 5.
        bias (float): An offset (usually positive to avoid dividing by 0). Default: 1.0.
        alpha (float): A scale factor, usually positive. Default: 1.0.
        beta (float): An exponent. Default: 0.5.
        norm_region (str): Specifies normalization region. Options: "ACROSS_CHANNELS". Default: "ACROSS_CHANNELS".
        x (Tensor): A 4-D Tensor with float16 or float32 data type.

    Returns:
        Tensor, with the same shape and data type as `x`.

    Raises:
        TypeError: If `depth_radius` is not an int.
        TypeError: If `bias`, `alpha` or `beta` is not a float.
        TypeError: If `norm_region` is not a str.
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[[[0.1], [0.2]],
        ...                       [[0.3], [0.4]]]]), mindspore.float32)
        >>> output = ops.lrn(input_x)
        >>> print(output)
        [[[[0.09534626]
           [0.1825742 ]]
          [[0.2860388 ]
           [0.3651484 ]]]]
    """
    lrn_op = NN_OPS.LRN(depth_radius, bias, alpha, beta, norm_region)
    return lrn_op(x)


def mish(x):
    r"""
    Computes MISH(A Self Regularized Non-Monotonic Neural Activation Function) of input tensors element-wise.

    The function is shown as follows:

    .. math::

        \text{output} = x * \tanh(\log(1 + \exp(\text{x})))

    See more details in `A Self Regularized Non-Monotonic Neural Activation Function
    <https://arxiv.org/abs/1908.08681>`_.

    Args:
        x (Tensor): Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
            additional dimensions, with float16 or float32 data type.

    Returns:
        Tensor, with the same type and shape as the `x`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.

    Examples:
        >>> input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
        >>> output = ops.mish(input_x)
        >>> print(output)
        [[-0.3034014  3.9974129 -0.00026832]
         [ 1.9439590  -0.0033576 9.0000000]]
    """
    return mish_(x)


def max_pool3d(x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    r"""
    Performs a 3D max pooling on the input Tensor.

    Typically the input is a Tensor with shape :math:`(N_{in}, C_{in}, D_{in}, H_{in}, W_{in})`, outputs
    regional maximum in the :math:`(D_{in}, H_{in}, W_{in})`-dimension. Given `kernel_size`
    :math:`ks = (d_{ker}, h_{ker}, w_{ker})` and `stride` :math:`s = (s_0, s_1, s_2)`, the operation is as follows.

    .. math::
        \text{output}(N_i, C_j, d, h, w) =
        \max_{l=0, \ldots, d_{ker}-1} \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times d + l, s_1 \times h + m, s_2 \times w + n)

    Args:
        x (Tensor): Tensor of shape :math:`(N_{in}, C_{in}, D_{in}, H_{in}, W_{in})` with data type of int8,
            int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32 or float64.
        kernel_size (Union[int, tuple[int]]): The size of kernel used to take the maximum value and arg
            value, is an int number that represents depth, height and width of the kernel, or a tuple of
            three int numbers that represent depth, height and width respectively.
        stride (Union[int, tuple[int]]): The distance of kernel moving, an int number that represents
            the depth, height and width of movement are both stride, or a tuple of three int numbers that
            represent depth, height and width of movement respectively. Default: `kernel_size`.
        padding (Union[int, tuple[int]]): An int number that represents the depth, height and width of movement are both
            strides, or a tuple of three int numbers that represent depth, height and width of movement respectively.
            Default: 0.
        dilation (Union[int, tuple[int]]): Control the stride of elements in the kernel. Default: 1.
        ceil_mode (bool): Whether to use ceil instead of floor to calculate output shape. Default: False.
        return_indices (bool): Whether to output the indices of max value. Default: False.

    Returns:
        If `return_indices` is False, return a Tensor `output`, else return a tuple (`output`, `argmax`).

        - **output** (Tensor) - Maxpooling result, with shape :math:`(N_{out}, C_{out}, D_{out}, H_{out}, W_{out})`.
          It has the same data type as `x`.
        - **argmax** (Tensor) - Index corresponding to the maximum value. Data type is int64. It will be return
          only when `return_indices` is True.

    Raises:
        TypeError: If `x` is not a Tensor.
        ValueError: If length of shape of `x` is not equal to 5.
        TypeError: If `kernel_size` , `stride` , `padding` or `dilation` is not int or tuple.
        ValueError: If `kernel_size` or `stride` is less than 1.
        ValueError: If `padding` is less than 0.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> x = Tensor(np.arange(2 * 1 * 2 * 2 * 2).reshape((2, 1, 2, 2, 2)), mindspore.float32)
        >>> output_tensor, argmax = F.max_pool3d(x, kernel_size=2, stride=1, padding=1, return_indices=True)
        >>> print(output_tensor.shape)
        (2, 1, 3, 3, 3)
        >>> print(argmax.shape)
        (2, 1, 3, 3, 3)
    """
    strides = stride if (stride is not None) else kernel_size
    max_pool3d_with_argmax_ = _get_cache_prim(NN_OPS.MaxPool3DWithArgmax)(
        kernel_size, strides, padding, dilation, ceil_mode)
    out, indices = max_pool3d_with_argmax_(x)
    if return_indices:
        return out, indices
    return out


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
        ``CPU`` ``GPU``

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
        _grid_sampler_2d = _get_cache_prim(NN_OPS.GridSampler2D)(interpolation_mode, padding_mode, align_corners)
        return _grid_sampler_2d(input_x, grid)
    _grid_sampler_3d = _get_cache_prim(NN_OPS.GridSampler3D)(interpolation_mode, padding_mode, align_corners)
    return _grid_sampler_3d(input_x, grid)


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction="mean", zero_infinity=False):
    """
    Calculates the CTC (Connectionist Temporal Classification) loss and the gradient.

    The CTC algorithm is proposed in `Connectionist Temporal Classification: Labeling Unsegmented Sequence Data with
    Recurrent Neural Networks <http://www.cs.toronto.edu/~graves/icml_2006.pdf>`_.

    Args:
        log_probs (Tensor): A tensor of shape (T, N, C), where T is input length, N is batch size and C is
            number of classes (including blank).
        targets (Tensor): A tensor of shape (N, S), where S is max target length, means the target sequences.
        input_lengths (Union(Tuple, Tensor)): A tuple or Tensor of shape(N). It means the lengths of the input.
        target_lengths (Union(Tuple, Tensor)): A tuple or Tensor of shape(N). It means the lengths of the target.
        blank (int): The blank label. Default: 0.
        reduction (string): Apply specific reduction method to the output: 'none', 'mean', or 'sum'. Default: 'mean'.
        zero_infinity (bool): Whether to set infinite loss and correlation gradient to zero. Default: False.

    Returns:
        neg_log_likelihood (Tensor), A loss value which is differentiable with respect to each input node.

        log_alpha (Tensor), The probability of possible trace of input to target.

    Raises:
        TypeError: If `zero_infinity` is not a bool, reduction is not string.
        TypeError: If the dtype of `log_probs` or `grad_out` is not float or double.
        TypeError: If the dtype of `targets`, `input_lengths` or `target_lengths` is not int32 or int64.
        RuntimeError: If the rank of `log_probs` is not 3.
        RuntimeError: If the rank of `targets` is not 2.
        RuntimeError: If the shape of `input_lengths` does not match {batch_size|N}.
        RuntimeError: If the shape of `target_lengths` does not match {batch_size|N}.
        RuntimeError: If the types of `targets`, `input_lengths`, `grad_out` or `target_lengths` are different.
        RuntimeError: If the value of `blank` is not in range [0, num_labels|C).
        RuntimeError: If any value of `input_lengths` is larger than (num_labels|C).
        RuntimeError: If any target_lengths[i] is not in range [0, input_length[i]].

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> log_probs = Tensor(np.array([[[0.3, 0.6, 0.6]],
                                         [[0.9, 0.4, 0.2]]]).astype(np.float32))
        >>> targets = Tensor(np.array([[0, 1]]), mstype.int32)
        >>> input_lengths = Tensor(np.array([2]), mstype.int32)
        >>> target_lengths = Tensor(np.array([1]), mstype.int32)
        >>> loss, log_alpha = fun.ctc_loss(log_probs, targets, input_lengths,
                                           target_lengths, 0, 'mean', True)
        >>> print(loss)
        -2.2986124
        >>> print(log_alpha)
        [[[0.3       0.3            -inf      -inf      -inf]
          [1.2       1.8931472 1.2            -inf      -inf]]]
    """
    ctc_loss_op = NN_OPS.CTCLossV2(blank=blank, reduction="none", zero_infinity=zero_infinity)
    loss, log_alpha = ctc_loss_op(log_probs, targets, input_lengths, target_lengths)
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'mean':
        input_type = loss.dtype
        target_length_t = target_lengths.clip(1., None)
        loss = loss.astype("float32")
        loss = loss / target_length_t
        loss = loss.mean()
        loss = loss.astype(input_type)
    return (loss, log_alpha)


def ctc_greedy_decoder(inputs, sequence_length, merge_repeated=True):
    r"""
    Performs greedy decoding on the logits given in inputs.

    Args:
        inputs (Tensor): The input Tensor must be a 3-D tensor whose shape is
            :math:`(max\_time, batch\_size, num\_classes)`. `num_classes` must be `num_labels + 1` classes,
            `num_labels` indicates the number of actual labels. Blank labels are reserved.
            Default blank label is `num_classes - 1`. Data type must be float32 or float64.
        sequence_length (Tensor): A tensor containing sequence lengths with the shape of :math:`(batch\_size, )`.
            The type must be int32. Each value in the tensor must be equal to or less than `max_time`.
        merge_repeated (bool): If true, merge repeated classes in output. Default: True.

    Returns:
        decoded_indices (Tensor), A tensor with shape of :math:`(total\_decoded\_outputs, 2)`.
        Data type is int64.

        decoded_values (Tensor), A tensor with shape of :math:`(total\_decoded\_outputs, )`,
        it stores the decoded classes. Data type is int64.

        decoded_shape (Tensor), A tensor with shape of :math:`(batch\_size, max\_decoded\_legth)`.
        Data type is int64.

        log_probability (Tensor), A tensor with shape of :math:`(batch\_size, 1)`,
        containing sequence log-probability, has the same type as `inputs`.

    Raises:
        TypeError: If `merge_repeated` is not a bool.
        ValueError: If length of shape of `inputs` is not equal to 3.
        ValueError: If length of shape of `sequence_length` is not equal to 1.
        ValueError: If value in the `sequence_length` is larger than `max_time`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> inputs = Tensor(np.array([[[0.6, 0.4, 0.2], [0.8, 0.6, 0.3]],
        ...                           [[0.0, 0.6, 0.0], [0.5, 0.4, 0.5]]]), mindspore.float32)
        >>> sequence_length = Tensor(np.array([2, 2]), mindspore.int32)
        >>> decoded_indices, decoded_values, decoded_shape, log_probability = ctc_greedy_decode(inputs, sequence_length)
        >>> print(decoded_indices)
        [[0 0]
         [0 1]
         [1 0]]
        >>> print(decoded_values)
        [0 1 0]
        >>> print(decoded_shape)
        [2 2]
        >>> print(log_probability)
        [[-1.2]
         [-1.3]]
    """
    _ctc_greedy_decoder = _get_cache_prim(NN.CTCGreedyDecoder)(merge_repeated)
    return _ctc_greedy_decoder(inputs, sequence_length)


__all__ = [
    'adaptive_avg_pool2d',
    'adaptive_max_pool3d',
    'avg_pool2d',
    'binary_cross_entropy_with_logits',
    'max_pool3d',
    'kl_div',
    'celu',
    'deformable_conv2d',
    'dropout2d',
    'dropout3d',
    'fast_gelu',
    'hardshrink',
    'soft_shrink',
    'intopk',
    'interpolate',
    'log_softmax',
    'mish',
    'lrn',
    'hardswish',
    'softsign',
    'selu',
    'pdist',
    'pad',
    'cross_entropy',
    'grid_sample',
    'smooth_l1_loss',
    'nll_loss',
    'ctc_loss',
    'ctc_greedy_decoder'
]
__all__.sort()

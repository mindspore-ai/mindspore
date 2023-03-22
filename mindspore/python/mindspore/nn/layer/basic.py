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

"""basic"""
from __future__ import absolute_import

import math
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import context, log as logger
from mindspore.ops.composite.multitype_ops import _constexpr_utils as const_utils
from mindspore.common.seed import _get_graph_seed
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.primitive import constexpr, Primitive
from mindspore.common.parameter import Parameter
from mindspore._extends import cell_attr_register
from mindspore._checkparam import Rel, Validator
from mindspore.nn.cell import Cell
from mindspore.nn.layer.activation import get_activation

__all__ = ['Dropout', 'Flatten', 'Dense', 'ClipByNorm', 'Norm', 'OneHot', 'Pad', 'Unfold', 'Tril', 'Triu',
           'ResizeBilinear', 'MatrixDiag', 'MatrixDiagPart', 'MatrixSetDiag', 'L1Regularizer', 'Dropout1d',
           'Dropout2d', 'Dropout3d', 'Upsample', 'Roll', 'Identity', 'Unflatten']


class L1Regularizer(Cell):
    r"""
    Applies l1 regularization to weights.

    l1 regularization makes weights sparsity.

    .. math::
        \text{loss}=\lambda * \text{reduce_sum}(\text{abs}(\omega))

    where :math:`\lambda` is `scale` .

    Note:
        scale(regularization factor) should be a number which greater than 0.

    Args:
        scale (int, float): l1 regularization factor which greater than 0.

    Inputs:
        - **weights** (Tensor) - The input of L1Regularizer with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, which dtype is higher precision data type between mindspore.float32 and weights dtype,
        and Tensor shape is ()

    Raises:
        TypeError: If `scale` is neither an int nor float.
        ValueError: If `scale` is not greater than 0.
        ValueError: If `scale` is math.inf or math.nan.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> scale = 0.5
        >>> net = nn.L1Regularizer(scale)
        >>> weights = Tensor(np.array([[1.0, -2.0], [-3.0, 4.0]]).astype(np.float32))
        >>> output = net(weights)
        >>> print(output.asnumpy())
        5.0
    """

    def __init__(self, scale):
        """Initialize L1Regularizer."""
        super(L1Regularizer, self).__init__()
        Validator.check_value_type("scale", scale, [int, float], self.cls_name)
        if scale <= 0:
            raise ValueError(
                f"For '{self.cls_name}', the 'scale' must be greater than 0, but got {scale}.")
        if math.isinf(scale) or math.isnan(scale):
            raise ValueError(
                f"For '{self.cls_name}', the 'scale' can not be INF or NAN, but got {scale}.")
        self.abs = P.Abs()
        self.reduce_sum = P.ReduceSum()
        self.scale = Tensor(scale, dtype=mstype.float32)

    def construct(self, weights):
        const_utils.check_type_valid(
            F.dtype(weights), mstype.number_type, 'weights')
        l1_regularization = self.scale * self.reduce_sum(self.abs(weights))
        return l1_regularization


class Dropout(Cell):
    r"""
    Dropout layer for the input.

    Dropout is a regularization method. The operator randomly sets some neurons output to 0
    according to the probability of discarding the probability of discarding.
    During the reasoning, this layer returns the same Tensor as the `x`.

    This technique is proposed in paper `Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    <http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf>`_ and proved to be effective to reduce
    over-fitting and prevents neurons from co-adaptation. See more details in `Improving neural networks by
    preventing co-adaptation of feature detectors
    <https://arxiv.org/pdf/1207.0580.pdf>`_.

    Note:
        - Each channel will be zeroed out independently on every construct call.
        - Parameter `keep_prob` will be removed in a future version, please use parameter `p` instead.
          Parameter `p` means the probability of the element of the input tensor to be zeroed.

    Args:
        keep_prob (float): Deprecated. The keep rate, greater than 0 and less equal than 1.
            E.g. rate=0.9, dropping out 10% of input neurons. Default: 0.5.
        p (Union[float, int, None]): The dropout rate, greater than or equal to 0 and less than 1.
            E.g. rate=0.9, dropping out 90% of input neurons. Default: None.

    Inputs:
        - **x** (Tensor) - The input of Dropout with data type of float16 or float32.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, output tensor with the same shape as the `x`.

    Raises:
        TypeError: If `keep_prob` is not a float.
        TypeError: If the dtype of `p` is not float or int.
        TypeError: If dtype of `x` is not neither float16 nor float32.
        ValueError: If `keep_prob` is not in range (0, 1].
        ValueError: If `p` is not in range [0, 1).
        ValueError: If length of shape of `x` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.ones([2, 2, 3]), mindspore.float32)
        >>> net = nn.Dropout(p=0.2)
        >>> net.set_train()
        >>> output = net(x)
        >>> print(output.shape)
        (2, 2, 3)
    """

    def __init__(self, keep_prob=0.5, p=None):
        """Initialize Dropout."""
        super(Dropout, self).__init__()
        if p is None:
            logger.warning("For Dropout, this parameter `keep_prob` will be deprecated, please use `p` instead.")
            Validator.check_value_type('keep_prob', keep_prob, [float], self.cls_name)
            if keep_prob <= 0 or keep_prob > 1:
                raise ValueError(f"For '{self.cls_name}', the 'keep_prob' must be a number in range (0, 1], "
                                 f"but got {keep_prob}.")
            seed0, seed1 = _get_graph_seed(0, "dropout")
            self.dropout = P.Dropout(keep_prob, seed0, seed1)
        else:
            Validator.check_value_type('p', p, [float, int], self.cls_name)
            if p < 0 or p >= 1:
                raise ValueError(f"For '{self.cls_name}', the 'p' must be a number in range [0, 1), "
                                 f"but got {p}.")
            seed0, seed1 = _get_graph_seed(0, "dropout")
            self.dropout = P.Dropout(1.0 - p, seed0, seed1)
        self.p = p
        self.keep_prob = keep_prob

    def construct(self, x):
        if not self.training or self.keep_prob == 1 or self.p == 0:
            return x

        out, _ = self.dropout(x)
        return out

    def extend_repr(self):
        if self.p is None:
            logger.warning("For Dropout, this parameter `keep_prob` will be deprecated, please use `p` instead.")
            return f'keep_prob={self.keep_prob}'
        return f'p={self.p}'


class Dropout1d(Cell):
    r"""
    During training, randomly zeroes entire channels of the input tensor with probability `p`
    from a Bernoulli distribution (For a 3-dimensional tensor with a shape of :math:`(N, C, L)`,
    the channel feature map refers to a 1-dimensional feature map with the shape of :math:`L`).

    For example, the :math:`j\_th` channel of the :math:`i\_th` sample in the batched input is a to-be-processed
    `1D` tensor input[i,j].
    Each channel will be zeroed out independently on every forward call with probability `p` using samples
    from a Bernoulli distribution.

    The paper `Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    <http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf>`_ mentioned this technology, And it is proved that
    it can effectively reduce over fitting and prevent neuronal coadaptation.
    For more details, refer to `Improving neural networks by preventing co-adaptation of feature detectors
    <https://arxiv.org/pdf/1207.0580.pdf>`_ .

    `Dropout1d` can improve the independence between channel feature maps.

    Args:
        p (float, optional): The dropping probability of a channel, between 0 and 1, e.g. `p` = 0.8,
            which means an 80% chance of being set to 0. Default: 0.5.

    Inputs:
        - **x** (Tensor) - A tensor with shape :math:`(N, C, L)` or :math:`(C, L)`, where `N` is the batch size,
          `C` is the number of channels, `L` is the feature length. The data type must be int8, int16, int32,
          int64, float16, float32 or float64.

    Outputs:
        Tensor, output, with the same shape and data type as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If the data type of `p` is not float.
        ValueError: If `p` is out of the range `[0.0, 1.0]`.
        ValueError: If `x` shape is not `2D` or `3D`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import nn, Tensor
        >>> op = nn.Dropout1d(p=0.6)
        >>> op.training = True
        >>> a = Tensor(np.ones((3, 3)), ms.float32)
        >>> output = op(a)
    """

    def __init__(self, p=0.5):
        """Initialize Dropout1d."""
        super(Dropout1d, self).__init__()
        Validator.check_value_type('p', p, [float], self.cls_name)
        if p < 0 or p > 1:
            raise ValueError(f"For '{self.cls_name}', the 'p' must be a number in range [0, 1], "
                             f"but got {p}.")
        self.prob = p

    def construct(self, x):
        if not self.training or self.prob == 0:
            return x

        out = F.dropout1d(x, self.prob)
        return out


class Dropout2d(Cell):
    r"""
    During training, randomly zeroes some channels of the input tensor with probability `p`
    from a Bernoulli distribution (For a 4-dimensional tensor with a shape of :math:`NCHW`,
    the channel feature map refers to a 2-dimensional feature map with the shape of :math:`HW`).

    For example, the :math:`j\_th` channel of the :math:`i\_th` sample in the batched input is a to-be-processed
    `2D` tensor input[i,j].
    Each channel will be zeroed out independently on every forward call with probability `p` using samples
    from a Bernoulli distribution.

    `Dropout2d` can improve the independence between channel feature maps.

    Refer to :func:`mindspore.ops.dropout2d` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> dropout = nn.Dropout2d(p=0.5)
        >>> x = Tensor(np.ones([2, 1, 2, 3]), mindspore.float32)
        >>> output = dropout(x)
        >>> print(output.shape)
        (2, 1, 2, 3)
    """

    def __init__(self, p=0.5):
        """Initialize Dropout2d."""
        super(Dropout2d, self).__init__()
        Validator.check_value_type('p', p, [float], self.cls_name)
        if p < 0 or p > 1:
            raise ValueError(f"For '{self.cls_name}', the 'p' must be a number in range [0, 1], "
                             f"but got {p}.")
        self.keep_prob = 1.0 - p
        self.dropout2d = P.Dropout2D(self.keep_prob)

    def construct(self, x):
        if not self.training or self.keep_prob == 1:
            return x

        out, _ = self.dropout2d(x)
        return out

    def extend_repr(self):
        return 'p={}'.format(self.keep_prob)


class Dropout3d(Cell):
    r"""
    During training, randomly zeroes some channels of the input tensor
    with probability `p` from a Bernoulli distribution (For a 5-dimensional tensor with
    a shape of :math:`NCDHW`, the channel feature map refers to a 3-dimensional feature
    map with a shape of :math:`DHW`).

    For example, the :math:`j\_th` channel of the :math:`i\_th` sample in the batched input is a to-be-processed
    `3D` tensor input[i,j].
    Each channel will be zeroed out independently on every forward call which based on Bernoulli distribution
    probability `p`.

    `Dropout3d` can improve the independence between channel feature maps.

    Refer to :func:`mindspore.ops.dropout3d` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> dropout = nn.Dropout3d(p=0.5)
        >>> x = Tensor(np.ones([2, 1, 2, 1, 2]), mindspore.float32)
        >>> output = dropout(x)
        >>> print(output.shape)
        (2, 1, 2, 1, 2)
    """

    def __init__(self, p=0.5):
        """Initialize Dropout3d."""
        super(Dropout3d, self).__init__()
        Validator.check_value_type('p', p, [float], self.cls_name)
        if p < 0 or p > 1:
            raise ValueError(f"For '{self.cls_name}', the 'p' must be a number in range [0, 1], "
                             f"but got {p}.")
        self.keep_prob = 1.0 - p
        self.dropout3d = P.Dropout3D(self.keep_prob)

    def construct(self, x):
        if not self.training or self.keep_prob == 1:
            return x

        out, _ = self.dropout3d(x)
        return out

    def extend_repr(self):
        return 'p={}'.format(self.keep_prob)


class Upsample(Cell):
    r"""
    For details, please refer to :func:`mindspore.ops.interpolate`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]])
        >>> upsample = nn.Upsample(size=(5, 5))
        >>> out = upsample(x)
        >>> print(x.asnumpy())
        [[[[1. 2. 3. 4.]
           [5. 6. 7. 8.]]]]
        >>> print(out.asnumpy())
        [[[[1. 1. 2. 3. 4.]
           [1. 1. 2. 3. 4.]
           [1. 1. 2. 3. 4.]
           [5. 5. 6. 7. 8.]
           [5. 5. 6. 7. 8.]]]]
        >>> print(out.shape)
        (1, 1, 5, 5)
    """

    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None):
        """Initialize Upsample."""
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def construct(self, x):
        out = F.interpolate(x, self.size, self.scale_factor, self.mode,
                            self.align_corners, self.recompute_scale_factor)
        return out


class Flatten(Cell):
    r"""
    Flatten the input Tensor along dimensions from `start_dim` to `end_dim`.

    Args:
        start_dim (int, optional): The first dimension to flatten. Default: 1.
        end_dim (int, optional): The last dimension to flatten. Default: -1.

    Inputs:
        - **x** (Tensor) - The input Tensor to be flattened.

    Outputs:
        Tensor. If no dimensions are flattened, returns the original `x`, otherwise return the flattened Tensor.
        If `x` is a 0-dimensional Tensor, a 1-dimensional Tensor will be returned.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `start_dim` or `end_dim` is not int.
        ValueError: If `start_dim` is greater than `end_dim` after canonicalized.
        ValueError: If `start_dim` or `end_dim` is not in range of [-x.dim, x.dim-1].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[[1.2, 1.2], [2.1, 2.1]], [[2.2, 2.2], [3.2, 3.2]]]), mindspore.float32)
        >>> net = nn.Flatten()
        >>> output = net(x)
        >>> print(output)
        [[1.2 1.2 2.1 2.1]
         [2.2 2.2 3.2 3.2]]
        >>> print(f"before flatten the x shape is {x.shape}")
        before flatten the x shape is  (2, 2, 2)
        >>> print(f"after flatten the output shape is {output.shape}")
        after flatten the output shape is (2, 4)
    """

    def __init__(self, start_dim=1, end_dim=-1):
        """Initialize Flatten."""
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def construct(self, x):
        return F.flatten(x, start_dim=self.start_dim, end_dim=self.end_dim)


class Identity(Cell):
    """
    Returns a Tensor with the same shape and contents as input.

    Inputs:
        - **x** (Tensor) - The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. The data type is Number.

    Outputs:
        Tensor, the shape of tensor and the data type are the same as `input_x`, :math:`(x_1, x_2, ..., x_R)`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3, 4]), mindspore.int64)
        >>> net = nn.Identity()
        >>> output = net(x)
        >>> print(output)
        [1 2 3 4]
    """

    def __init__(self):
        """Initialize Identity."""
        super(Identity, self).__init__()
        self.identity = P.Identity()

    def construct(self, x):
        out = self.identity(x)
        return out


class Dense(Cell):
    r"""
    The dense connected layer.

    Applies dense connected layer for the input. This layer implements the operation as:

    .. math::
        \text{outputs} = \text{activation}(\text{X} * \text{kernel} + \text{bias}),

    where :math:`X` is the input tensors, :math:`\text{activation}` is the activation function passed as the activation
    argument (if passed in), :math:`\text{kernel}` is a weight matrix with the same
    data type as the :math:`X` created by the layer, and :math:`\text{bias}` is a bias vector
    with the same data type as the :math:`X` created by the layer (only if has_bias is True).

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as `x`. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as `x`. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector :math:`\text{bias}`. Default: True.
        activation (Union[str, Cell, Primitive, None]): activate function applied to the output of the fully connected
            layer. Both activation name, e.g. 'relu', and mindspore activation function, e.g. mindspore.ops.ReLU(),
            are supported. Default: None.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(*, in\_channels)`. The `in_channels` in `Args` should be equal
          to :math:`in\_channels` in `Inputs`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Raises:
        TypeError: If `in_channels` or `out_channels` is not an int.
        TypeError: If `has_bias` is not a bool.
        TypeError: If `activation` is not one of str, Cell, Primitive, None.
        ValueError: If length of shape of `weight_init` is not equal to 2 or shape[0] of `weight_init`
                    is not equal to `out_channels` or shape[1] of `weight_init` is not equal to `in_channels`.
        ValueError: If length of shape of `bias_init` is not equal to 1
                    or shape[0] of `bias_init` is not equal to `out_channels`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), mindspore.float32)
        >>> net = nn.Dense(3, 4)
        >>> output = net(x)
        >>> print(output.shape)
        (2, 4)
    """

    @cell_attr_register(attrs=['has_bias', 'activation'])
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None):
        """Initialize Dense."""
        super(Dense, self).__init__()
        self.in_channels = Validator.check_positive_int(
            in_channels, "in_channels", self.cls_name)
        self.out_channels = Validator.check_positive_int(
            out_channels, "out_channels", self.cls_name)
        self.has_bias = Validator.check_bool(
            has_bias, "has_bias", self.cls_name)
        self.reshape = P.Reshape()
        self.shape_op = P.Shape()

        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 2 or weight_init.shape[0] != out_channels or \
                    weight_init.shape[1] != in_channels:
                raise ValueError(f"For '{self.cls_name}', weight init shape error. The ndim of 'weight_init' must "
                                 f"be equal to 2, and the first dim must be equal to 'out_channels', and the "
                                 f"second dim must be equal to 'in_channels'. But got 'weight_init': {weight_init}, "
                                 f"'out_channels': {out_channels}, 'in_channels': {in_channels}.")
        self.weight = Parameter(initializer(
            weight_init, [out_channels, in_channels]), name="weight")

        self.bias = None
        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError(f"For '{self.cls_name}', bias init shape error. The ndim of 'bias_init' must "
                                     f"be equal to 1, and the first dim must be equal to 'out_channels'. But got "
                                     f"'bias_init': {bias_init}, 'out_channels': {out_channels}.")
            self.bias = Parameter(initializer(
                bias_init, [out_channels]), name="bias")
            self.bias_add = P.BiasAdd()

        self.matmul = P.MatMul(transpose_b=True)
        self.activation = get_activation(activation) if isinstance(
            activation, str) else activation
        if activation is not None and not isinstance(self.activation, (Cell, Primitive)):
            raise TypeError(f"For '{self.cls_name}', the 'activation' must be str or Cell or Primitive, but got "
                            f"{type(activation).__name__}.")
        self.activation_flag = self.activation is not None

    def construct(self, x):
        x_shape = self.shape_op(x)
        if len(x_shape) != 2:
            x = self.reshape(x, (-1, x_shape[-1]))
        x = self.matmul(x, self.weight)
        if self.has_bias:
            x = self.bias_add(x, self.bias)
        if self.activation_flag:
            x = self.activation(x)
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (F.shape(x)[-1],)
            x = self.reshape(x, out_shape)
        return x

    def extend_repr(self):
        s = 'input_channels={}, output_channels={}'.format(
            self.in_channels, self.out_channels)
        if self.has_bias:
            s += ', has_bias={}'.format(self.has_bias)
        if self.activation_flag:
            s += ', activation={}'.format(self.activation)
        return s


@constexpr
def _is_equal_one(x):
    if x is None:
        return False
    return F.equal(F.reduce_mean(x), 1.0)


@constexpr
def _dtype_check(x_dtype, prim_name=None):
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    if x_dtype not in [mstype.float32, mstype.float16]:
        raise TypeError(
            f"{msg_prefix} x_dtype must be float32 or float16, but got {x_dtype}.")


@constexpr
def _is_float_dtype(dtype):
    if dtype in [mstype.float32, mstype.float16]:
        return True
    return False


@constexpr
def _need_reduce_all(axis):
    if axis == ():
        return True
    return False


class ClipByNorm(Cell):
    r"""
    Clips tensor values to a maximum :math:`L_2`-norm.

    The output of this layer remains the same if the :math:`L_2`-norm of the input tensor
    is not greater than the argument clip_norm. Otherwise the tensor will be normalized as:

    .. math::
        \text{output}(X) = \frac{\text{clip_norm} * X}{L_2(X)},

    where :math:`L_2(X)` is the :math:`L_2`-norm of :math:`X`.

    Args:
        axis (Union[None, int, tuple(int)]): Compute the L2-norm along the Specific dimension.
                                            Default: None, all dimensions to calculate.

    Inputs:
        - **x** (Tensor) - Tensor of shape N-D. The type must be float32 or float16.
        - **clip_norm** (Tensor) - A scalar Tensor of shape :math:`()` or :math:`(1)`.
          Or a tensor shape can be broadcast to input `x` shape.

    Outputs:
        Tensor, clipped tensor with the same shape as the `x`, whose type is float32.

    Raises:
        TypeError: If `axis` is not one of None, int, tuple.
        TypeError: If dtype of `x` is neither float32 nor float16.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.ClipByNorm()
        >>> x = Tensor(np.random.randint(0, 10, [4, 16]), mindspore.float32)
        >>> clip_norm = Tensor(np.array([100]).astype(np.float32))
        >>> output = net(x, clip_norm)
        >>> print(output.shape)
        (4, 16)

    """

    def __init__(self, axis=None):
        """Initialize ClipByNorm."""
        super(ClipByNorm, self).__init__()
        self.clip_by_norm = inner.ClipByNorm(axis)

    def construct(self, x, clip_norm):
        values_clip = self.clip_by_norm(x, clip_norm)
        return values_clip


class Norm(Cell):
    r"""
    Computes the norm of vectors, currently including Euclidean norm, i.e., :math:`L_2`-norm.

    .. math::

        norm(x) = \sqrt{\sum_{i=1}^{n} (x_i^2)}

    Args:
        axis (Union[tuple, int]): The axis over which to compute vector norms. Default: ().
        keep_dims (bool): If true, the axis indicated in `axis` are kept with size 1. Otherwise,
                   the dimensions in `axis` are removed from the output shape. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor which is not empty. The data type should be float16 or float32.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, output tensor with dimensions in 'axis' reduced to 1 will be returned if 'keep_dims' is True;
        otherwise a Tensor with dimensions in 'axis' removed is returned. The data type is the same with `x`.

    Raises:
        TypeError: If `axis` is neither an int nor a tuple.
        TypeError: If `keep_dims` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.Norm(axis=0)
        >>> x = Tensor(np.array([[4, 4, 9, 1], [2, 1, 3, 6]]), mindspore.float32)
        >>> print(x.shape)
        (2, 4)
        >>> output = net(x)
        >>> print(output)
        [4.472136 4.1231055 9.486833 6.0827627]
        >>> print(output.shape)
        (4,)
        >>> net = nn.Norm(axis=0, keep_dims=True)
        >>> x = Tensor(np.array([[4, 4, 9, 1], [2, 1, 3, 6]]), mindspore.float32)
        >>> print(x.shape)
        (2, 4)
        >>> output = net(x)
        >>> print(output)
        [4.472136 4.1231055 9.486833 6.0827627]
        >>> print(output.shape)
        (1, 4)
        >>> net = nn.Norm(axis=1)
        >>> x = Tensor(np.array([[4, 4, 9, 1], [2, 1, 3, 6]]), mindspore.float32)
        >>> print(x.shape)
        (2, 4)
        >>> output = net(x)
        >>> print(output)
        [10.677078 7.071068]
        >>> print(output.shape)
        (2,)
    """

    def __init__(self, axis=(), keep_dims=False):
        """Initialize Norm."""
        super(Norm, self).__init__()
        Validator.check_value_type(
            "keep_dims", keep_dims, [bool], self.cls_name)
        self.axis = axis
        self.keep_dims = keep_dims
        self.reduce_sum = P.ReduceSum(True)
        self.sqrt = P.Sqrt()
        self.squeeze = P.Squeeze(self.axis)

    def construct(self, x):
        x = self.sqrt(self.reduce_sum(F.square(x), self.axis))

        if not self.keep_dims:
            x = self.squeeze(x)
        return x

    def extend_repr(self):
        return 'axis={}, keep_dims={}'.format(self.axis, self.keep_dims)


class OneHot(Cell):
    """
    Returns a one-hot tensor.

    The locations represented by indices in argument `indices` take value on_value,
    while all other locations take value off_value.

    Note:
        If the input indices is rank :math:`N`, the output will have rank :math:`N+1`. The new
        axis is created at dimension `axis`.

    If `indices` is a scalar, the output shape will be a vector of length `depth`.

    If `indices` is a vector of length `features`, the output shape will be:

    .. code-block::

        features * depth if axis == -1

        depth * features if axis == 0

    If `indices` is a matrix with shape `[batch, features]`, the output shape will be:

    .. code-block::

        batch * features * depth if axis == -1

        batch * depth * features if axis == 1

        depth * batch * features if axis == 0

    Args:
        axis (int): Features x depth if axis is -1, depth x features
                    if axis is 0. Default: -1.
        depth (int): A scalar defining the depth of the one hot dimension. Default: 1.
        on_value (float): A scalar defining the value to fill in output[i][j]
                          when indices[j] = i. Default: 1.0.
        off_value (float): A scalar defining the value to fill in output[i][j]
                           when indices[j] != i. Default: 0.0.
        dtype (:class:`mindspore.dtype`): Data type of 'on_value' and 'off_value', not the
                                          data type of indices. Default: mindspore.float32.

    Inputs:
        - **indices** (Tensor) - A tensor of indices with data type of int32 or int64.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, the one-hot tensor of data type `dtype` with dimension at `axis` expanded to `depth` and filled with
        on_value and off_value. The dimension of the `Outputs` is equal to the dimension of the `indices` plus one.

    Raises:
        TypeError: If `axis` or `depth` is not an int.
        TypeError: If dtype of `indices` is neither int32 nor int64.
        ValueError: If `axis` is not in range [-1, len(indices_shape)].
        ValueError: If `depth` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # 1st sample: add new coordinates at axis 1
        >>> net = nn.OneHot(depth=4, axis=1)
        >>> indices = Tensor([[1, 3], [0, 2]], dtype=mindspore.int32)
        >>> output = net(indices)
        >>> print(output)
        [[[0. 0.]
          [1. 0.]
          [0. 0.]
          [0. 1.]]
         [[1. 0.]
          [0. 0.]
          [0. 1.]
          [0. 0.]]]
        >>> # The results are shown below:
        >>> print(output.shape)
        (2, 4, 2)
        >>> # 2nd sample: add new coordinates at axis 0
        >>> net = nn.OneHot(depth=4, axis=0)
        >>> indices = Tensor([[1, 3], [0, 2]], dtype=mindspore.int32)
        >>> output = net(indices)
        >>> print(output)
        [[[0. 0.]
          [1. 0.]]
         [[1. 0.]
          [0. 0.]]
         [[0. 0.]
          [0. 1.]]
         [[0. 1.]
          [0. 0.]]]
        >>> # The results are shown below:
        >>> print(output.shape)
        (4, 2, 2)
        >>> # 3rd sample: add new coordinates at the last dimension.
        >>> net = nn.OneHot(depth=4, axis=-1)
        >>> indices = Tensor([[1, 3], [0, 2]], dtype=mindspore.int32)
        >>> output = net(indices)
        >>> # The results are shown below:
        >>> print(output)
        [[[0. 1. 0. 0.]
          [0. 0. 0. 1.]]
         [[1. 0. 0. 0.]
          [0. 0. 1. 0.]]]
        >>> print(output.shape)
        (2, 2, 4)
        >>> indices = Tensor([1, 3, 0, 2], dtype=mindspore.int32)
        >>> output = net(indices)
        >>> print(output)
        [[0. 1. 0. 0.]
         [0. 0. 0. 1.]
         [1. 0. 0. 0.]
         [0. 0. 1. 0.]]
        >>> print(output.shape)
        (4, 4)
    """

    def __init__(self, axis=-1, depth=1, on_value=1.0, off_value=0.0, dtype=mstype.float32):
        """Initialize OneHot."""
        super(OneHot, self).__init__()
        self.onehot = P.OneHot(axis)
        self.depth = depth
        self.dtype = dtype
        self.on_value = on_value
        self.off_value = off_value

    def construct(self, indices):
        return self.onehot(indices, self.depth, F.cast(self.on_value, self.dtype), F.cast(self.off_value, self.dtype))


class Pad(Cell):
    r"""
    Pads the input tensor according to the paddings and mode.

    Args:
        paddings (tuple): The shape of parameter `paddings` is (N, 2). N is the rank of input data. All elements of
            paddings are int type. For `D` th dimension of the `x`, paddings[D, 0] indicates how many sizes to be
            extended ahead of the `D` th dimension of the input tensor, and paddings[D, 1] indicates how many sizes to
            be extended behind of the `D` th dimension of the input tensor. The padded size of each dimension D of the
            output is: :math:`paddings[D, 0] + input\_x.dim\_size(D) + paddings[D, 1]`,
            e.g.:

            .. code-block::

                mode = "CONSTANT".
                paddings = [[1,1], [2,2]].
                x = [[1,2,3], [4,5,6], [7,8,9]].
                # The above can be seen: 1st dimension of `x` is 3, 2nd dimension of `x` is 3.
                # Substitute into the formula to get:
                # 1st dimension of output is paddings[0][0] + 3 + paddings[0][1] = 1 + 3 + 1 = 5.
                # 2nd dimension of output is paddings[1][0] + 3 + paddings[1][1] = 2 + 3 + 2 = 7.
                # So the shape of output is (5, 7).

        mode (str): Specifies padding mode. The optional values are "CONSTANT", "REFLECT", "SYMMETRIC".
            Default: "CONSTANT".

    Inputs:
        - **x** (Tensor) - The input tensor.

    Outputs:
        Tensor, the tensor after padding.

        - If `mode` is "CONSTANT", it fills the edge with 0, regardless of the values of the `x`.
          If the `x` is [[1,2,3], [4,5,6], [7,8,9]] and `paddings` is [[1,1], [2,2]], then the
          Outputs is [[0,0,0,0,0,0,0], [0,0,1,2,3,0,0], [0,0,4,5,6,0,0], [0,0,7,8,9,0,0], [0,0,0,0,0,0,0]].
        - If `mode` is "REFLECT", it uses a way of symmetrical copying through the axis of symmetry to fill in.
          If the `x` is [[1,2,3], [4,5,6], [7,8,9]] and `paddings` is [[1,1], [2,2]], then the
          Outputs is [[6,5,4,5,6,5,4], [3,2,1,2,3,2,1], [6,5,4,5,6,5,4], [9,8,7,8,9,8,7], [6,5,4,5,6,5,4]].
        - If `mode` is "SYMMETRIC", the filling method is similar to the "REFLECT". It is also copied
          according to the symmetry axis, except that it includes the symmetry axis. If the `x`
          is [[1,2,3], [4,5,6], [7,8,9]] and `paddings` is [[1,1], [2,2]], then the Outputs is
          [[2,1,1,2,3,3,2], [2,1,1,2,3,3,2], [5,4,4,5,6,6,5], [8,7,7,8,9,9,8], [8,7,7,8,9,9,8]].

    Raises:
        TypeError: If `paddings` is not a tuple.
        ValueError: If length of `paddings` is more than 4 or its shape is not (N, 2).
        ValueError: If `mode` is not one of 'CONSTANT', 'REFLECT', 'SYMMETRIC'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> import mindspore.nn as nn
        >>> import numpy as np
        >>> # If `mode` is "CONSTANT"
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.pad = nn.Pad(paddings=((1, 1), (2, 2)), mode="CONSTANT")
        ...     def construct(self, x):
        ...         return self.pad(x)
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), mindspore.float32)
        >>> pad = Net()
        >>> output = pad(x)
        >>> print(output)
        [[0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 1. 2. 3. 0. 0.]
         [0. 0. 4. 5. 6. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0.]]
        >>> # Another way to call
        >>> pad = ops.Pad(paddings=((1, 1), (2, 2)))
        >>> # From the above code, we can see following:
        >>> # "paddings=((1, 1), (2, 2))",
        >>> # paddings[0][0] = 1, indicates a row of values is filled top of the input data in the 1st dimension.
        >>> # Shown as follows:
        >>> # [[0. 0. 0.]
        >>> #  [1. 2. 3.]
        >>> #  [4. 5. 6.]]
        >>> # paddings[0][1] = 1 indicates a row of values is filled below input data in the 1st dimension.
        >>> # Shown as follows:
        >>> # [[0. 0. 0.]
        >>> #  [1. 2. 3.]
        >>> #  [4. 5. 6.]
        >>> #  [0. 0. 0.]]
        >>> # paddings[1][0] = 2, indicates 2 rows of values is filled in front of input data in the 2nd dimension.
        >>> # Shown as follows:
        >>> # [[0. 0. 0. 0. 0.]
        >>> #  [0. 0. 1. 2. 3.]
        >>> #  [0. 0. 4. 5. 6.]
        >>> #  [0. 0. 0. 0. 0.]]
        >>> # paddings[1][1] = 2, indicates 2 rows of values is filled in front of input data in the 2nd dimension.
        >>> # Shown as follows:
        >>> # [[0. 0. 0. 0. 0. 0. 0.]
        >>> #  [0. 0. 1. 2. 3. 0. 0.]
        >>> #  [0. 0. 4. 5. 6. 0. 0.]
        >>> #  [0. 0. 0. 0. 0. 0. 0.]]
        >>> output = pad(x)
        >>> print(output)
        [[0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 1. 2. 3. 0. 0.]
         [0. 0. 4. 5. 6. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0.]]
        >>> # if mode is "REFLECT"
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.pad = nn.Pad(paddings=((1, 1), (2, 2)), mode="REFLECT")
        ...     def construct(self, x):
        ...         return self.pad(x)
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), mindspore.float32)
        >>> pad = Net()
        >>> output = pad(x)
        >>> print(output)
        [[6. 5. 4. 5. 6. 5. 4.]
         [3. 2. 1. 2. 3. 2. 1.]
         [6. 5. 4. 5. 6. 5. 4.]
         [3. 2. 1. 2. 3. 2. 1.]]
        >>> # if mode is "SYMMETRIC"
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.pad = nn.Pad(paddings=((1, 1), (2, 2)), mode="SYMMETRIC")
        ...     def construct(self, x):
        ...         return self.pad(x)
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6]]), mindspore.float32)
        >>> pad = Net()
        >>> output = pad(x)
        >>> print(output)
        [[2. 1. 1. 2. 3. 3. 2.]
         [2. 1. 1. 2. 3. 3. 2.]
         [5. 4. 4. 5. 6. 6. 5.]
         [5. 4. 4. 5. 6. 6. 5.]]
    """

    def __init__(self, paddings, mode="CONSTANT"):
        """Initialize Pad."""
        super(Pad, self).__init__()
        self.mode = mode
        self.paddings = paddings
        Validator.check_string(
            self.mode, ["CONSTANT", "REFLECT", "SYMMETRIC"], 'mode', self.cls_name)
        if not isinstance(paddings, tuple):
            raise TypeError(f"For '{self.cls_name}', the type of 'paddings' must be tuple, "
                            f"but got {type(paddings).__name__}.")
        for item in paddings:
            if len(item) != 2:
                raise ValueError(f"For '{self.cls_name}', the dimension of 'paddings' must be (n, 2), "
                                 f"but got {paddings}.")
        if len(paddings) > 4:
            raise ValueError(f"For '{self.cls_name}', only 'paddings' up to 4 dims is supported, but got "
                             f"{len(paddings)}.")
        if mode == "CONSTANT":
            self.pad = P.Pad(self.paddings)
        else:
            self.paddings = Tensor(np.array(self.paddings), dtype=mstype.int64)
            self.pad = P.MirrorPad(mode=mode)

    def construct(self, x):
        if self.mode == "CONSTANT":
            x = self.pad(x)
        else:
            x = self.pad(x, self.paddings)
        return x


@constexpr
def bilinear(shape, size, scale, align_corners, prim_name=None):
    """Check input and calculate shape"""
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    if not isinstance(align_corners, bool):
        raise TypeError(
            f"{msg_prefix} type of 'align_corners' must be bool, but got {type(align_corners).__name__}.")
    if size is None and scale is None:
        raise ValueError(f"{msg_prefix} 'size' and 'scale' both none.")
    if size is not None and scale is not None:
        raise ValueError(f"{msg_prefix} 'size' and 'scale' both not none.")
    if size is not None:
        if not isinstance(size, (tuple, list)):
            raise ValueError(
                f"{msg_prefix} 'size' must be tuple or list or None, but got {type(size).__name__}.")
        return size
    ret = (scale * shape[2], scale * shape[3])
    return ret


class ResizeBilinear(Cell):
    r"""
    Samples the input tensor to the given size or scale_factor by using bilinear interpolate.

    Args:
        half_pixel_centers (bool): Whether half pixel center. If set to True, `align_corners` should be False.
            Default: False.

    Inputs:
        - **x** (Tensor) - Tensor to be resized. Input tensor must be a 4-D tensor with shape
          :math:`(batch, channels, height, width)`, with data type of float16 or float32.
        - **size** (Union[tuple[int], list[int], None]): A tuple or list of 2 int elements
          :math:`(new\_height, new\_width)`,the new size of the tensor.
          One and only one of size and scale_factor can be set to None. Default: None.
        - **scale_factor** (int, None): The scale factor of new size of the tensor. The value should be positive
          integer. One and only one of size and scale_factor can be set to None. Default: None.
        - **align_corners** (bool): If true, rescale input by :math:`(new\_height - 1) / (height - 1)`, which exactly
          aligns the 4 corners of images and resized images. If false, rescale by :math:`new\_height / height`.
          Default: False.

    Outputs:
        Resized tensor.
        If size is set, the result is 4-D tensor with shape :math:`(batch, channels, new\_height, new\_width)`,
        and the data type is the same as `x`.
        If scale is set, the result is 4-D tensor with shape
        :math:`(batch, channels, scale\_factor * height, scale\_factor * width)` and the data type is the same as `x`.

    Raises:
        TypeError: If `size` is not one of tuple, list, None.
        TypeError: If `scale_factor` is neither int nor None.
        TypeError: If `align_corners` is not a bool.
        TypeError: If `half_pixel_centers` is not a bool.
        TypeError: If `align_corners` and `half_pixel_centers` are all True.
        TypeError: If `half_pixel_centers` is True and device_target not Ascend.
        TypeError: If dtype of `x` is neither float16 nor float32.
        ValueError: If `size` and `scale_factor` are both None or not None.
        ValueError: If length of shape of `x` is not equal to 4.
        ValueError: If `scale_factor` is an int which is less than 0.
        ValueError: If `size` is a list or tuple whose length is not equal to 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], mindspore.float32)
        >>> resize_bilinear = nn.ResizeBilinear()
        >>> result = resize_bilinear(x, size=(5,5))
        >>> print(x)
        [[[[1. 2. 3. 4.]
           [5. 6. 7. 8.]]]]
        >>> print(result)
        [[[[1.        1.8       2.6       3.4       4.       ]
           [2.6       3.4       4.2000003 5.        5.6000004]
           [4.2       5.0000005 5.8       6.6       7.2      ]
           [5.        5.8       6.6       7.4       8.       ]
           [5.        5.8       6.6       7.4000006 8.       ]]]]
        >>> print(result.shape)
        (1, 1, 5, 5)
    """

    def __init__(self, half_pixel_centers=False):
        """Initialize ResizeBilinear."""
        super(ResizeBilinear, self).__init__()
        self.half_pixel_centers = half_pixel_centers

    def construct(self, x, size=None, scale_factor=None, align_corners=False):
        shape = bilinear(x.shape, size, scale_factor,
                         align_corners, self.cls_name)
        resize_bilinear = P.ResizeBilinear(
            shape, align_corners, self.half_pixel_centers)
        return resize_bilinear(x)


class Unfold(Cell):
    r"""
    Extracts patches from images.
    The input tensor must be a 4-D tensor and the data format is NCHW.

    Args:
        ksizes (Union[tuple[int], list[int]]): The size of sliding window, must be a tuple or a list of integers,
            and the format is [1, ksize_row, ksize_col, 1].
        strides (Union[tuple[int], list[int]]): Distance between the centers of the two consecutive patches,
            must be a tuple or list of int, and the format is [1, stride_row, stride_col, 1].
        rates (Union[tuple[int], list[int]]): In each extracted patch, the gap between the corresponding dimension
            pixel positions, must be a tuple or a list of integers, and the format is [1, rate_row, rate_col, 1].
        padding (str): The type of padding algorithm, is a string whose value is "same" or "valid", not case sensitive.
            Default: "valid".

            - same: Means that the patch can take the part beyond the original image, and this part is filled with 0.

            - valid: Means that the taken patch area must be completely covered in the original image.

    Inputs:
        - **x** (Tensor) - A 4-D tensor whose shape is [in_batch, in_depth, in_row, in_col] and
          data type is number.

    Outputs:
        Tensor, a 4-D tensor whose data type is same as `x`,
        and the shape is [out_batch, out_depth, out_row, out_col] where `out_batch` is the same as the `in_batch`.

        - :math:`out\_depth = ksize\_row * ksize\_col * in\_depth`
        - :math:`out\_row = (in\_row - (ksize\_row + (ksize\_row - 1) * (rate\_row - 1))) // stride\_row + 1`
        - :math:`out\_col = (in\_col - (ksize\_col + (ksize\_col - 1) * (rate\_col - 1))) // stride\_col + 1`

    Raises:
        TypeError: If `ksizes`, `strides` or `rates` is neither a tuple nor list.
        ValueError: If shape of `ksizes`, `strides` or `rates` is not (1, x_row, x_col, 1).
        ValueError: If the second and third element of `ksizes`, `strides` or `rates` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> net = Unfold(ksizes=[1, 2, 2, 1], strides=[1, 2, 2, 1], rates=[1, 2, 2, 1])
        >>> # As stated in the above code:
        >>> # ksize_row = 2, ksize_col = 2, rate_row = 2, rate_col = 2, stride_row = 2, stride_col = 2.
        >>> image = Tensor(np.ones([2, 3, 6, 6]), dtype=mstype.float16)
        >>> # in_batch = 2, in_depth = 3, in_row = 6, in_col = 6.
        >>> # Substituting the formula to get:
        >>> # out_batch = in_batch = 2
        >>> # out_depth = 2 * 2 * 3 = 12
        >>> # out_row = (6 - (2 + (2 - 1) * (2 - 1))) // 2 + 1 = 2
        >>> # out_col = (6 - (2 + (2 - 1) * (2 - 1))) // 2 + 1 = 2
        >>> output = net(image)
        >>> print(output.shape)
        (2, 12, 2, 2)
    """

    def __init__(self, ksizes, strides, rates, padding="valid"):
        """Initialize Unfold."""
        super(Unfold, self).__init__()

        def _check_tuple_or_list(arg_name, arg_val, prim_name):
            Validator.check_value_type(f"{arg_name}s", ksizes, [
                tuple, list], self.cls_name)
            if len(arg_val) != 4 or arg_val[0] != 1 or arg_val[3] != 1:
                raise ValueError(f"For '{prim_name}' the format of '{arg_name}s' must be [1, {arg_name}_row, "
                                 f"{arg_name}_col, 1], but got {arg_val}.")
            if not isinstance(arg_val[1], int) or not isinstance(arg_val[2], int) or arg_val[1] < 1 or arg_val[2] < 1:
                raise ValueError(f"For '{prim_name}' the {arg_name}_row and {arg_name}_col in '{arg_name}s' must be "
                                 f"an positive integer number, but got {arg_name}_row is {arg_val[1]}, "
                                 f"{arg_name}_col is {arg_val[2]}")

        _check_tuple_or_list("ksize", ksizes, self.cls_name)
        _check_tuple_or_list("stride", strides, self.cls_name)
        _check_tuple_or_list("rate", rates, self.cls_name)
        ksizes = ksizes[0], ksizes[3], ksizes[1], ksizes[2]
        strides = strides[0], strides[3], strides[1], strides[2]
        rates = rates[0], rates[3], rates[1], rates[2]
        self.extract_image_patches = inner.ExtractImagePatches(
            ksizes, strides, rates, padding)

    def construct(self, input_x):
        result = self.extract_image_patches(input_x)
        return result


def tril(x_shape, x_dtype, k):
    value = F.cast(P.Tril(diagonal=k)(F.ones(x_shape, x_dtype)), x_dtype)
    return value


class Tril(Cell):
    """
    Returns a tensor, the elements above the specified main diagonal are set to zero.

    Divide the matrix elements into upper and lower triangles along the main diagonal (including diagonals).

    The parameter `k` controls the choice of diagonal.
    If `k` = 0, split along the main diagonal and keep all the elements of the lower triangle.
    If `k` > 0, select the diagonal `k` along the main diagonal upwards, and keep all the elements of the lower
    triangle.
    If `k` < 0, select the diagonal `k` along the main diagonal down, and keep all the elements of the lower
    triangle.

    Inputs:
        - **x** (Tensor) - The input tensor. The data type is
          `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_.
        - **k** (Int) - The index of diagonal. Default: 0. If the dimensions of the input matrix are d1 and d2,
          the range of k should be in [-min(d1, d2)+1, min(d1, d2)-1], and the output value will be the same as the
          input `x` when `k` is out of range.

    Outputs:
        Tensor, has the same shape and type as input `x`.

    Raises:
        TypeError: If `k` is not an int.
        ValueError: If length of shape of `x` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case1: k = 0
        >>> x = Tensor(np.array([[ 1,  2,  3,  4],
        ...                      [ 5,  6,  7,  8],
        ...                      [10, 11, 12, 13],
        ...                      [14, 15, 16, 17]]))
        >>> tril = nn.Tril()
        >>> result = tril(x)
        >>> print(result)
        [[ 1  0  0  0]
         [ 5  6  0  0]
         [10 11 12  0]
         [14 15 16 17]]
        >>> # case2: k = 1
        >>> x = Tensor(np.array([[ 1,  2,  3,  4],
        ...                      [ 5,  6,  7,  8],
        ...                      [10, 11, 12, 13],
        ...                      [14, 15, 16, 17]]))
        >>> tril = nn.Tril()
        >>> result = tril(x, 1)
        >>> print(result)
        [[ 1  2  0  0]
         [ 5  6  7  0]
         [10 11 12 13]
         [14 15 16 17]]
        >>> # case3: k = 2
        >>> x = Tensor(np.array([[ 1,  2,  3,  4],
        ...                      [ 5,  6,  7,  8],
        ...                      [10, 11, 12, 13],
        ...                      [14, 15, 16, 17]]))
        >>> tril = nn.Tril()
        >>> result = tril(x, 2)
        >>> print(result)
        [[ 1  2  3  0]
         [ 5  6  7  8]
         [10 11 12 13]
         [14 15 16 17]]
        >>> # case4: k = -1
        >>> x = Tensor(np.array([[ 1,  2,  3,  4],
        ...                      [ 5,  6,  7,  8],
        ...                      [10, 11, 12, 13],
        ...                      [14, 15, 16, 17]]))
        >>> tril = nn.Tril()
        >>> result = tril(x, -1)
        >>> print(result)
        [[ 0  0  0  0]
         [ 5  0  0  0]
         [10 11  0  0]
         [14 15 16  0]]
    """

    def __init__(self):
        """Initialize Tril."""
        super(Tril, self).__init__()
        self.dtype = P.DType()
        self.mul = P.Mul()
        self.cast = P.Cast()

    def construct(self, x, k=0):
        assist = tril(x.shape, self.dtype(x), k)
        result = self.mul(self.cast(x, mstype.float32),
                          self.cast(assist, mstype.float32))
        return self.cast(result, self.dtype(x))


def triu(x_shape, x_dtype, k):
    value = F.cast(P.Triu(k)(F.ones(x_shape, x_dtype)), x_dtype)
    return value


class Triu(Cell):
    """
    Returns a tensor with elements below the kth diagonal zeroed.

    The upper triangular part of the matrix is defined as the elements on and above the diagonal.

    The parameter `k` controls the diagonal to be considered. If `k` = 0, all elements on and above the main diagonal
    are retained. Positive values do not include as many diagonals above the main diagonal. Similarly,
    negative values include as many diagonals below the main diagonal.

    Inputs:
        - **x** (Tensor) - The input tensor. The data type is Number.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
        - **k** (Int) - The index of diagonal. Default: 0

    Outputs:
        Tensor, has the same type and shape as input `x`.

    Raises:
        TypeError: If `k` is not an int.
        ValueError: If length of shape of `x` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[ 1,  2,  3,  4],
        ...                      [ 5,  6,  7,  8],
        ...                      [10, 11, 12, 13],
        ...                      [14, 15, 16, 17]]))
        >>> triu = nn.Triu()
        >>> result = triu(x)
        >>> print(result)
        [[ 1  2  3  4]
         [ 0  6  7  8]
         [ 0  0 12 13]
         [ 0  0  0 17]]
        >>> x = Tensor(np.array([[ 1,  2,  3,  4],
        ...                      [ 5,  6,  7,  8],
        ...                      [10, 11, 12, 13],
        ...                      [14, 15, 16, 17]]))
        >>> triu = nn.Triu()
        >>> result = triu(x, 1)
        >>> print(result)
        [[ 0  2  3  4]
         [ 0  0  7  8]
         [ 0  0  0 13]
         [ 0  0  0  0]]
        >>> x = Tensor(np.array([[ 1,  2,  3,  4],
        ...                      [ 5,  6,  7,  8],
        ...                      [10, 11, 12, 13],
        ...                      [14, 15, 16, 17]]))
        >>> triu = nn.Triu()
        >>> result = triu(x, 2)
        >>> print(result)
        [[ 0  0  3  4]
         [ 0  0  0  8]
         [ 0  0  0  0]
         [ 0  0  0  0]]
        >>> x = Tensor(np.array([[ 1,  2,  3,  4],
        ...                      [ 5,  6,  7,  8],
        ...                      [10, 11, 12, 13],
        ...                      [14, 15, 16, 17]]))
        >>> triu = nn.Triu()
        >>> result = triu(x, -1)
        >>> print(result)
        [[ 1  2  3  4]
         [ 5  6  7  8]
         [ 0 11 12 13]
         [ 0  0 16 17]]
    """

    def __init__(self):
        """Initialize Triu."""
        super(Triu, self).__init__()
        self.dtype = P.DType()
        self.mul = P.Mul()
        self.cast = P.Cast()

    def construct(self, x, k=0):
        assist = triu(x.shape, self.dtype(x), k)
        result = self.mul(self.cast(x, mstype.float32),
                          self.cast(assist, mstype.float32))
        return self.cast(result, self.dtype(x))


def _get_matrix_diag_assist(x_shape, x_dtype):
    """Get matrix diag assist"""
    base_eye = F.reshape(
        F.eye(x_shape[-1], x_shape[-1], x_dtype), (x_shape[-1] * x_shape[-1],))
    if len(x_shape) == 1:
        assist = F.reshape(base_eye, x_shape + (x_shape[-1],))
    else:
        assist = F.reshape(
            F.tile(base_eye, x_shape[:-1]), x_shape + (x_shape[-1],))
    value = F.cast(assist, x_dtype)
    return value


def _get_matrix_diag_part_assist(x_shape, x_dtype):
    """Get matrix diag part assist"""
    base_eye = F.reshape(
        F.eye(x_shape[-2], x_shape[-1], x_dtype), (x_shape[-2] * x_shape[-1],))
    if len(x_shape) <= 2:
        assist = F.reshape(base_eye, x_shape)
    else:
        assist = F.reshape(F.tile(base_eye, x_shape[:-2]), x_shape)
    value = F.cast(assist, x_dtype)
    return value


class MatrixDiag(Cell):
    r"""
    Returns a batched diagonal tensor with a given batched diagonal values.

    Assume `x` has :math:`k` dimensions :math:`[I, J, K, ..., N]`, then the output is a tensor of rank
    :math:`k+1` with dimensions :math:`[I, J, K, ..., N, N]` where:
    :math:`output[i, j, k, ..., m, n] = 1\{m=n\} * x[i, j, k, ..., n]`.

    Inputs:
        - **x** (Tensor) - The diagonal values. It can be one of the following data types:
          float32, float16, int32, int8, and uint8.
          The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Outputs:
        Tensor, has the same type as input `x`. The shape must be x.shape + (x.shape[-1], ).

    Raises:
        TypeError: If dtype of `x` is not one of float32, float16, int32, int8 or uint8.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.array([1, -1]), mindspore.float32)
        >>> matrix_diag = nn.MatrixDiag()
        >>> output = matrix_diag(x)
        >>> print(x.shape)
        (2,)
        >>> print(output)
        [[ 1.  0.]
         [ 0. -1.]]
        >>> print(output.shape)
        (2, 2)
        >>> x = Tensor(np.array([[1, -1], [1, -1]]), mindspore.float32)
        >>> matrix_diag = nn.MatrixDiag()
        >>> output = matrix_diag(x)
        >>> print(x.shape)
        (2, 2)
        >>> print(output)
        [[[ 1.  0.]
          [ 0. -1.]]
         [[ 1.  0.]
          [ 0. -1.]]]
        >>> print(output.shape)
        (2, 2, 2)
        >>> x = Tensor(np.array([[1, -1, 1], [1, -1, 1]]), mindspore.float32)
        >>> matrix_diag = nn.MatrixDiag()
        >>> output = matrix_diag(x)
        >>> print(x.shape)
        (2, 3)
        >>> print(output)
        [[[ 1.  0.  0.]
          [ 0. -1.  0.]
          [ 0.  0.  1.]]
         [[ 1.  0.  0.]
          [ 0. -1.  0.]
          [ 0.  0.  1.]]]
        >>> print(output.shape)
        (2, 3, 3)
    """

    def __init__(self):
        """Initialize MatrixDiag."""
        super(MatrixDiag, self).__init__()
        self.matrix_diag = inner.MatrixDiag()
        self.dtype = P.DType()

    def construct(self, input_x):
        x_shape = F.shape(input_x)
        x_dtype = self.dtype(input_x)
        assist = _get_matrix_diag_assist(x_shape, x_dtype)
        out_matrix_diag = self.matrix_diag(input_x, assist)
        return out_matrix_diag


class MatrixDiagPart(Cell):
    r"""
    Returns the batched diagonal part of a batched tensor.

    Assume `x` has :math:`k` dimensions :math:`[I, J, K, ..., M, N]`, then the output is a tensor of rank
    :math:`k-1` with dimensions :math:`[I, J, K, ..., min(M, N)]` where:
    :math:`output[i, j, k, ..., n] = x[i, j, k, ..., n, n]`.

    Inputs:
        - **x** (Tensor) - The batched tensor. It can be one of the following data types:
          float32, float16, int32, int8, and uint8.

    Outputs:
        Tensor, has the same type as input `x`. The shape must be x.shape[:-2] + [min(x.shape[-2:])].

    Raises:
        TypeError: If dtype of `x` is not one of float32, float16, int32, int8 or uint8.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> x = Tensor([[[-1, 0], [0, 1]],
        ...             [[-1, 0], [0, 1]],
        ...             [[-1, 0], [0, 1]]], mindspore.float32)
        >>> matrix_diag_part = nn.MatrixDiagPart()
        >>> output = matrix_diag_part(x)
        >>> print(output)
        [[-1.  1.]
         [-1.  1.]
         [-1.  1.]]
        >>> x = Tensor([[-1, 0, 0, 1],
        ...             [-1, 0, 0, 1],
        ...             [-1, 0, 0, 1],
        ...             [-1, 0, 0, 1]], mindspore.float32)
        >>> matrix_diag_part = nn.MatrixDiagPart()
        >>> output = matrix_diag_part(x)
        >>> print(output)
        [-1.  0.  0.  1.]
    """

    def __init__(self):
        """Initialize MatrixDiagPart."""
        super(MatrixDiagPart, self).__init__()
        self.matrix_diag_part = inner.MatrixDiagPart()
        self.dtype = P.DType()

    def construct(self, input_x):
        x_shape = F.shape(input_x)
        x_dtype = self.dtype(input_x)
        assist = _get_matrix_diag_part_assist(x_shape, x_dtype)
        out_matrix_diag_part = self.matrix_diag_part(input_x, assist)
        return out_matrix_diag_part


class MatrixSetDiag(Cell):
    r"""
    Modifies the batched diagonal part of a batched tensor.

    Assume `x` has :math:`k+1` dimensions :math:`[I, J, K, ..., M, N]` and `diagonal` has :math:`k`
    dimensions :math:`[I, J, K, ..., min(M, N)]`, the output is a tensor of rank :math:`k+1` with dimensions
    :math:`[I, J, K, ..., M, N]`, where:

    .. math::
        output[i, j, k, ..., m, n] = diagonal[i, j, k, ..., n]\ for\ m == n

    .. math::
        output[i, j, k, ..., m, n] = x[i, j, k, ..., m, n]\ for\ m != n

    Inputs:
        - **x** (Tensor) - The batched tensor. Rank k+1, where k >= 1. It can be one of the following data types:
          float32, float16, int32, int8, and uint8.
        - **diagonal** (Tensor) - The diagonal values. Must have the same type as input `x`. Rank k, where k >= 1.

    Outputs:
        Tensor, has the same type and shape as input `x`.

    Raises:
        TypeError: If dtype of `x` or `diagonal` is not one of float32, float16, int32, int8 or uint8.
        ValueError: If length of shape of `x` is less than 2.
        ValueError: If x_shape[-2] < x_shape[-1] and x_shape[:-1] != diagonal_shape.
        ValueError: If x_shape[-2] >= x_shape[-1] and x_shape[:-2] + x_shape[-1:] != diagonal_shape.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor([[[-1, 0], [0, 1]], [[-1, 0], [0, 1]], [[-1, 0], [0, 1]]], mindspore.float32)
        >>> diagonal = Tensor([[-1., 2.], [-1., 1.], [-1., 1.]], mindspore.float32)
        >>> matrix_set_diag = nn.MatrixSetDiag()
        >>> output = matrix_set_diag(x, diagonal)
        >>> print(output)
        [[[-1.  0.]
          [ 0.  2.]]
         [[-1.  0.]
          [ 0.  1.]]
         [[-1.  0.]
          [ 0.  1.]]]
    """

    def __init__(self):
        """Initialize MatrixSetDiag."""
        super(MatrixSetDiag, self).__init__()
        self.matrix_set_diag = inner.MatrixSetDiag()
        self.dtype = P.DType()

    def construct(self, input_x, diagonal):
        x_shape = F.shape(input_x)
        x_dtype = self.dtype(input_x)
        assist = _get_matrix_diag_part_assist(x_shape, x_dtype)
        out_matrix_set_diag = self.matrix_set_diag(input_x, diagonal, assist)
        return out_matrix_set_diag


@constexpr
def _check_input_dim(axis, dim, cls_name):
    Validator.check_int_range(axis, -dim, dim, Rel.INC_LEFT, 'axis', cls_name)


class Roll(Cell):
    """
    Rolls the elements of a tensor along an axis.

    The elements are shifted positively (towards larger indices) by the offset of `shift` along the dimension of `axis`.
    Negative `shift` values will shift elements in the opposite direction. Elements that roll passed the last position
    will wrap around to the first and vice versa. Multiple shifts along multiple axes may be specified.

    Args:
        shift (Union[list(int), tuple(int), int]): Specifies the number of places by which elements are shifted
            positively (towards larger indices) along the specified dimension. Negative shifts will roll the elements
            in the opposite direction.
        axis (Union[list(int), tuple(int), int]): Specifies the dimension indexes of shape to be rolled.

    Inputs:
        - **input_x** (Tensor) - Input tensor.

    Outputs:
        Tensor, has the same shape and type as `input_x`.

    Raises:
        TypeError: If `shift` is not an int, a tuple or a list.
        TypeError: If `axis` is not an int, a tuple or a list.
        TypeError: If element of `shift` is not an int.
        TypeError: If element of `axis` is not an int.
        ValueError: If axis is out of the range [-len(input_x.shape), len(input_x.shape)).
        ValueError: If length of shape of `shift` is not equal to length of shape of `axis`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> input_x = Tensor(np.array([0, 1, 2, 3, 4]).astype(np.float32))
        >>> op = nn.Roll(shift=2, axis=0)
        >>> output = op(input_x)
        >>> print(output)
        [3. 4. 0. 1. 2.]
        >>> input_x = Tensor(np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]).astype(np.float32))
        >>> op = nn.Roll(shift=[1, -2], axis=[0, 1])
        >>> output = op(input_x)
        >>> print(output)
        [[7. 8. 9. 5. 6.]
         [2. 3. 4. 0. 1.]]
    """

    def __init__(self, shift, axis):
        """Initialize Roll"""
        super(Roll, self).__init__()
        Validator.check_value_type(
            "shift", shift, [int, tuple, list], self.cls_name)
        Validator.check_value_type(
            "axis", axis, [int, tuple, list], self.cls_name)
        self.shape_op = P.Shape()
        self.shift = shift
        self.axis = axis
        self.op_list = []
        self.gpu = False

        if not isinstance(self.axis, (list, tuple)):
            self.axis = [self.axis]
        if not isinstance(self.shift, (list, tuple)):
            self.shift = [self.shift]
        if context.get_context("device_target") == "GPU":
            Validator.check_int(len(self.shift), 1, Rel.GE, "shift", "Roll")
            Validator.check_int(len(self.axis), 1, Rel.GE, "axis", "Roll")
            for s_axis in self.axis:
                Validator.check_is_int(s_axis, "axis", "Roll")
            for s_shift in self.shift:
                Validator.check_is_int(s_shift, "shift", "Roll")
            self.roll = P.Roll(self.shift, self.axis)
            self.gpu = True
            if len(self.shift) != len(self.axis):
                raise ValueError(f"For '{self.cls_name}', the shape of 'shift' and the shape of 'axis' must be "
                                 f"the same, but got the length of 'shift' {len(self.shift)} "
                                 f"and the length of 'axis' {len(self.axis)}.")
        else:
            if not isinstance(self.axis, (list, tuple)):
                self.op_list.append(
                    (P.Roll(shift=self.shift, axis=0), self.axis))
            else:
                if len(self.shift) != len(self.axis):
                    raise ValueError(f"For '{self.cls_name}', the shape of 'shift' and the shape of 'axis' must be "
                                     f"the same, but got the length of 'shift' {len(self.shift)} "
                                     f"and the length of 'axis' {len(self.axis)}.")
                for idx, _ in enumerate(self.axis):
                    self.op_list.append(
                        (P.Roll(shift=self.shift[idx], axis=0), self.axis[idx]))

    def construct(self, input_x):
        dim = len(self.shape_op(input_x))
        if self.gpu:
            output = self.roll(input_x)
        else:
            for single_op_roll, single_axis in self.op_list:
                _check_input_dim(single_axis, dim, self.cls_name)
                if single_axis < 0:
                    single_axis += dim
                transpose_perm = []
                for i in range(dim):
                    transpose_perm.append(i)
                transpose_perm[0], transpose_perm[single_axis] = single_axis, 0

                input_x = input_x.transpose(transpose_perm)
                input_x = single_op_roll(input_x)
                input_x = input_x.transpose(transpose_perm)
            output = input_x
        return output


class Unflatten(Cell):
    r"""
    Summary:
        Unflattens a Tensor dim according to `axis` and `unflattened_size`.

    Args:
        axis (int): specifies the dimension of the input Tensor to be unflattened.
        unflattened_size (Union(tuple[int], list[int])): the new shape of the unflattened dimension of
            the Tensor and it can be a tuple of ints or a list of ints. The product of `unflattened_size`
            must equal to input_shape[axis].

    Inputs:
        - **input** (Tensor) - The input Tensor to be unflattened.

    Outputs:
        Tensor that has been unflattend.

    Raises:
        TypeError: If `axis` is not int.
        TypeError: If `unflattened_size` is neither tuple of ints nor list of ints.
        TypeError: The product of `unflattened_size` does not equal to input_shape[axis].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input = Tensor(np.arange(0, 100).reshape(2, 10, 5), mindspore.float32)
        >>> net = nn.Unflatten(1, (2, 5))
        >>> output = net(input)
        >>> print(f"before unflatten the input shape is {input.shape}")
        before unflatten the input shape is  (2, 10, 5)
        >>> print(f"after unflatten the output shape is {output.shape}")
        after unflatten the output shape is (2, 2, 5, 5)
    """

    def __init__(self, axis, unflattened_size):
        """Initialize Unflatten."""
        super(Unflatten, self).__init__()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        Validator.check_is_int(axis, 'axis', 'Unflatten')
        Validator.check_value_type(
            'unflattended_size', unflattened_size, (list, tuple), 'Unflatten')
        self.axis = axis
        if isinstance(unflattened_size, list):
            unflattened_size = tuple(unflattened_size)
        self.unflattened_size = unflattened_size

    def construct(self, input_x):
        input_shape = self.shape(input_x)
        new_shape = tuple()
        new_shape += input_shape[: self.axis]
        new_shape += self.unflattened_size
        if self.axis != -1:
            new_shape += input_shape[self.axis + 1:]
        return self.reshape(input_x, new_shape)

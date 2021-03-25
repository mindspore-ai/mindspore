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
import math
import numpy as np
import mindspore.common.dtype as mstype
from mindspore.ops.composite.multitype_ops import _constexpr_utils as const_utils
from mindspore.common.seed import _get_graph_seed
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.functional import identity
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.primitive import constexpr, Primitive
from mindspore.common.parameter import Parameter
from mindspore._extends import cell_attr_register
from mindspore._checkparam import Rel, Validator
from ..cell import Cell
from .activation import get_activation

__all__ = ['Dropout', 'Flatten', 'Dense', 'ClipByNorm', 'Norm', 'OneHot', 'Pad', 'Unfold',
           'Tril', 'Triu', 'ResizeBilinear', 'MatrixDiag', 'MatrixDiagPart', 'MatrixSetDiag', 'L1Regularizer']


class L1Regularizer(Cell):
    r"""
    Applies l1 regularization to weights.

    l1 regularization makes weights sparsity

    .. math::
        \text{loss}=\lambda * \text{reduce_sum}(\text{abs}(\omega))

    Note:
        scale(regularization factor) should be a number which greater than 0

    Args:
        scale (int, float): l1 regularization factor which greater than 0.

    Inputs:
        - **weights** (Tensor) - The input tensor

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
        super(L1Regularizer, self).__init__()
        Validator.check_value_type("scale", scale, [int, float], self.cls_name)
        if scale <= 0:
            raise ValueError("scale should be a number which greater than 0")
        if math.isinf(scale) or math.isnan(scale):
            raise ValueError("scale can not be INF or NAN")
        self.abs = P.Abs()
        self.reduce_sum = P.ReduceSum()
        self.scale = Tensor(scale, dtype=mstype.float32)

    def construct(self, weights):
        const_utils.check_type_valid(F.dtype(weights), mstype.number_type, 'weights')
        l1_regularization = self.scale * self.reduce_sum(self.abs(weights))
        return l1_regularization


class Dropout(Cell):
    r"""
    Dropout layer for the input.

    Randomly set some elements of the input tensor to zero with probability :math:`1 - keep\_prob` during training
    using samples from a Bernoulli distribution.

    The outputs are scaled by a factor of :math:`\frac{1}{keep\_prob}`    during training so
    that the output layer remains at a similar scale. During inference, this
    layer returns the same tensor as the input.

    This technique is proposed in paper `Dropout: A Simple Way to Prevent Neural Networks from Overfitting
    <http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf>`_ and proved to be effective to reduce
    over-fitting and prevents neurons from co-adaptation. See more details in `Improving neural networks by
    preventing co-adaptation of feature detectors
    <https://arxiv.org/pdf/1207.0580.pdf>`_.

    Note:
        Each channel will be zeroed out independently on every construct call.

    Args:
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. E.g. rate=0.9,
                   dropping out 10% of input units. Default: 0.5.
        dtype (:class:`mindspore.dtype`): Data type of input. Default: mindspore.float32.

    Inputs:
        - **input** (Tensor) - The input of Dropout with data type of float16 or float32.

    Outputs:
        Tensor, output tensor with the same shape as the input.

    Raises:
        TypeError: If `keep_prob` is not a float.
        TypeError: If dtype of `input` is not neither float16 nor float32.
        ValueError: If `keep_prob` is not in range (0, 1].
        ValueError: If length of shape of `input` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.ones([2, 2, 3]), mindspore.float32)
        >>> net = nn.Dropout(keep_prob=0.8)
        >>> net.set_train()
        Dropout<keep_prob=0.8>
        >>> output = net(x)
        >>> print(output.shape)
        (2, 2, 3)
    """

    def __init__(self, keep_prob=0.5, dtype=mstype.float32):
        super(Dropout, self).__init__()
        if keep_prob <= 0 or keep_prob > 1:
            raise ValueError("dropout probability should be a number in range (0, 1], but got {}".format(keep_prob))
        Validator.check_subclass("dtype", dtype, mstype.number_type, self.cls_name)
        Validator.check_value_type('keep_prob', keep_prob, [float], self.cls_name)
        self.keep_prob = keep_prob
        seed0, seed1 = _get_graph_seed(0, "dropout")
        self.seed0 = seed0
        self.seed1 = seed1
        self.dropout = P.Dropout(keep_prob, seed0, seed1)

    def construct(self, x):
        if not self.training:
            return x

        if self.keep_prob == 1:
            return x

        out, _ = self.dropout(x)
        return out

    def extend_repr(self):
        return 'keep_prob={}'.format(self.keep_prob)


class Flatten(Cell):
    r"""
    Flatten layer for the input.

    Flattens a tensor without changing dimension of batch size on the 0-th axis.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, \ldots)` to be flattened.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, X)`, where :math:`X` is
        the product of the remaining dimensions.

    Raises:
        TypeError: If `input` is not a subclass of Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input = Tensor(np.array([[[1.2, 1.2], [2.1, 2.1]], [[2.2, 2.2], [3.2, 3.2]]]), mindspore.float32)
        >>> net = nn.Flatten()
        >>> output = net(input)
        >>> print(output)
        [[1.2 1.2 2.1 2.1]
         [2.2 2.2 3.2 3.2]]
    """

    def __init__(self):
        super(Flatten, self).__init__()

    def construct(self, x):
        return F.reshape(x, (F.shape(x)[0], -1))

@constexpr
def check_dense_input_shape(x):
    if len(x) < 2:
        raise ValueError('For Dense, the dimension of input should not be less than 2, while the input dimension is '
                         + f'{len(x)}.')

class Dense(Cell):
    r"""
    The dense connected layer.

    Applies dense connected layer for the input. This layer implements the operation as:

    .. math::
        \text{outputs} = \text{activation}(\text{inputs} * \text{kernel} + \text{bias}),

    where :math:`\text{activation}` is the activation function passed as the activation
    argument (if passed in), :math:`\text{kernel}` is a weight matrix with the same
    data type as the inputs created by the layer, and :math:`\text{bias}` is a bias vector
    with the same data type as the inputs created by the layer (only if has_bias is True).

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input x. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (Union[str, Cell, Primitive]): activate function applied to the output of the fully connected layer,
            eg. 'ReLU'.Default: None.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, in\_channels)`.

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
        >>> input = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), mindspore.float32)
        >>> net = nn.Dense(3, 4)
        >>> output = net(input)
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
        super(Dense, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        self.has_bias = Validator.check_bool(has_bias)
        self.reshape = P.Reshape()
        self.shape_op = P.Shape()


        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 2 or weight_init.shape[0] != out_channels or \
                    weight_init.shape[1] != in_channels:
                raise ValueError("Weight init shape error.")
        self.weight = Parameter(initializer(weight_init, [out_channels, in_channels]), name="weight")

        self.bias = None
        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError("Bias init shape error.")
            self.bias = Parameter(initializer(bias_init, [out_channels]), name="bias")
            self.bias_add = P.BiasAdd()

        self.matmul = P.MatMul(transpose_b=True)
        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        if activation is not None and not isinstance(self.activation, (Cell, Primitive)):
            raise TypeError("The activation must be str or Cell or Primitive,"" but got {}.".format(activation))
        self.activation_flag = self.activation is not None

    def construct(self, x):
        x_shape = self.shape_op(x)
        check_dense_input_shape(x_shape)
        if len(x_shape) != 2:
            x = self.reshape(x, (-1, x_shape[-1]))
        x = self.matmul(x, self.weight)
        if self.has_bias:
            x = self.bias_add(x, self.bias)
        if self.activation_flag:
            x = self.activation(x)
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (-1,)
            x = self.reshape(x, out_shape)
        return x

    def extend_repr(self):
        s = 'input_channels={}, output_channels={}'.format(self.in_channels, self.out_channels)
        if self.has_bias:
            s += ', has_bias={}'.format(self.has_bias)
        if self.activation_flag:
            s += ', activation={}'.format(self.activation)
        return s


@constexpr
def _is_equal_one(x):
    if x is None:
        return False
    return bool(x.asnumpy().mean() == 1.0)


@constexpr
def _dtype_check(x_dtype):
    if x_dtype not in [mstype.float32, mstype.float16]:
        raise TypeError("The input type must be float32 or float16.")


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
        - **input** (Tensor) - Tensor of shape N-D. The type must be float32 or float16.
        - **clip_norm** (Tensor) - A scalar Tensor of shape :math:`()` or :math:`(1)`.
          Or a tensor shape can be broadcast to input shape.

    Outputs:
        Tensor, clipped tensor with the same shape as the input, whose type is float32.

    Raises:
        TypeError: If `axis` is not one of None, int, tuple.
        TypeError: If dtype of `input` is neither float32 nor float16.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.ClipByNorm()
        >>> input = Tensor(np.random.randint(0, 10, [4, 16]), mindspore.float32)
        >>> clip_norm = Tensor(np.array([100]).astype(np.float32))
        >>> output = net(input, clip_norm)
        >>> print(output.shape)
        (4, 16)

    """

    def __init__(self, axis=None):
        super(ClipByNorm, self).__init__()
        if axis is None:
            axis = ()
        if isinstance(axis, tuple):
            for idx, item in enumerate(axis):
                Validator.check_value_type("axis[%d]" % idx, item, [int], self.cls_name)
        self.axis = Validator.check_value_type('axis', axis, [int, tuple], self.cls_name)
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.select_ = P.Select()
        self.greater_ = P.Greater()
        self.cast = P.Cast()
        self.sqrt = P.Sqrt()
        self.max_op = P.Maximum()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.fill = P.Fill()
        self.expand_dims = P.ExpandDims()
        self.dtype = P.DType()

    def construct(self, x, clip_norm):
        mul_x = F.square(x)
        l2sum = self.cast(self.reduce_sum(mul_x, self.axis), mstype.float32)
        cond = self.greater_(l2sum, 0)
        ones_ = self.fill(self.dtype(cond), self.shape(cond), 1.0)
        l2sum_safe = self.select_(cond, l2sum, self.cast(ones_, self.dtype(l2sum)))
        l2norm = self.select_(cond, self.sqrt(l2sum_safe), l2sum)

        _dtype_check(self.dtype(x))
        if _is_equal_one(clip_norm):
            intermediate = x
        else:
            intermediate = x * clip_norm

        max_norm = self.max_op(l2norm, clip_norm)
        if _need_reduce_all(self.axis):
            max_norm = self.expand_dims(max_norm, -1)
        values_clip = self.cast(intermediate, mstype.float32) / max_norm
        values_clip = self.reshape(values_clip, self.shape(x))
        values_clip = identity(values_clip)
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
        - **input** (Tensor) - Tensor which is not empty.

    Outputs:
        Tensor, output tensor with dimensions in 'axis' reduced to 1 will be returned if 'keep_dims' is True;
        otherwise a Tensor with dimensions in 'axis' removed is returned.

    Raises:
        TypeError: If `axis` is neither an int nor tuple.
        TypeError: If `keep_dims` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.Norm(axis=0)
        >>> input = Tensor(np.array([[4, 4, 9, 1], [2, 1, 3, 6]]), mindspore.float32)
        >>> output = net(input)
        >>> print(output)
        [4.472136 4.1231055 9.486833 6.0827627]
    """

    def __init__(self, axis=(), keep_dims=False):
        super(Norm, self).__init__()
        Validator.check_value_type("keep_dims", keep_dims, [bool], self.cls_name)
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
        - **indices** (Tensor) - A tensor of indices with data type of int32 or int64 and arbitrary shape.

    Outputs:
        Tensor, the one-hot tensor of data type `dtype` with dimension at `axis` expanded to `depth` and filled with
        on_value and off_value.

    Raises:
        TypeError: If `axis` or `depth` is not an int.
        TypeError: If dtype of `indices` is neither int32 nor int64.
        ValueError: If `axis` is not in range [-1, len(indices_shape)].
        ValueError: If `depth` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
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
    """

    def __init__(self, axis=-1, depth=1, on_value=1.0, off_value=0.0, dtype=mstype.float32):
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
            paddings are int type. For `D` th dimension of input, paddings[D, 0] indicates how many sizes to be
            extended ahead of the `D` th dimension of the input tensor, and paddings[D, 1] indicates how many sizes to
            be extended behind of the `D` th dimension of the input tensor. The padded size of each dimension D of the
            output is: :math:`paddings[D, 0] + input\_x.dim\_size(D) + paddings[D, 1]`

        mode (str): Specifies padding mode. The optional values are "CONSTANT", "REFLECT", "SYMMETRIC".
            Default: "CONSTANT".

    Inputs:
        - **input_x** (Tensor) - The input tensor.

    Outputs:
        Tensor, the tensor after padding.

        - If `mode` is "CONSTANT", it fills the edge with 0, regardless of the values of the `input_x`.
          If the `input_x` is [[1,2,3], [4,5,6], [7,8,9]] and `paddings` is [[1,1], [2,2]], then the
          Outputs is [[0,0,0,0,0,0,0], [0,0,1,2,3,0,0], [0,0,4,5,6,0,0], [0,0,7,8,9,0,0], [0,0,0,0,0,0,0]].
        - If `mode` is "REFLECT", it uses a way of symmetrical copying through the axis of symmetry to fill in.
          If the `input_x` is [[1,2,3], [4,5,6], [7,8,9]] and `paddings` is [[1,1], [2,2]], then the
          Outputs is [[6,5,4,5,6,5,4], [3,2,1,2,3,2,1], [6,5,4,5,6,5,4], [9,8,7,8,9,8,7], [6,5,4,5,6,5,4]].
        - If `mode` is "SYMMETRIC", the filling method is similar to the "REFLECT". It is also copied
          according to the symmetry axis, except that it includes the symmetry axis. If the `input_x`
          is [[1,2,3], [4,5,6], [7,8,9]] and `paddings` is [[1,1], [2,2]], then the Outputs is
          [[2,1,1,2,3,3,2], [2,1,1,2,3,3,2], [5,4,4,5,6,6,5], [8,7,7,8,9,9,8], [8,7,7,8,9,9,8]].

    Raises:
        TypeError: If `paddings` is not a tuple.
        ValueError: If length of `paddings` is more than 4 or its shape is not (n, 2).
        ValueError: If `mode` is not one of 'CONSTANT', 'REFLECT', 'SYMMETRIC'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore.ops import operations as P
        >>> import mindspore.nn as nn
        >>> import numpy as np
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.pad = nn.Pad(paddings=((1, 1), (2, 2)), mode="CONSTANT")
        ...     def construct(self, x):
        ...         return self.pad(x)
        >>> x = np.array([[0.3, 0.5, 0.2], [0.5, 0.7, 0.3]], dtype=np.float32)
        >>> pad = Net()
        >>> output = pad(Tensor(x))
        >>> print(output)
        [[0.         0.         0.         0.         0.         0.        0.         ]
         [0.         0.         0.3        0.5        0.2        0.        0.         ]
         [0.         0.         0.5        0.7        0.3        0.        0.         ]
         [0.         0.         0.         0.         0.         0.        0.         ]]
    """

    def __init__(self, paddings, mode="CONSTANT"):
        super(Pad, self).__init__()
        self.mode = mode
        self.paddings = paddings
        Validator.check_string(self.mode, ["CONSTANT", "REFLECT", "SYMMETRIC"], 'mode', self.cls_name)
        if not isinstance(paddings, tuple):
            raise TypeError('Paddings must be tuple type.')
        for item in paddings:
            if len(item) != 2:
                raise ValueError('The shape of paddings must be (n, 2).')
        if len(paddings) > 4:
            raise ValueError('Only padding up to 4 dims is supported')
        if mode == "CONSTANT":
            self.pad = P.Pad(self.paddings)
        else:
            self.paddings = Tensor(np.array(self.paddings))
            self.pad = P.MirrorPad(mode=mode)

    def construct(self, x):
        if self.mode == "CONSTANT":
            x = self.pad(x)
        else:
            x = self.pad(x, self.paddings)
        return x


@constexpr
def bilinear(shape, size, scale, align_corners):
    """Check input and calculate shape"""
    if not isinstance(align_corners, bool):
        raise TypeError("align_corners should be type boolean")
    if size is None and scale is None:
        raise ValueError("size and scale both none")
    if size is not None and scale is not None:
        raise ValueError("size and scale both not none")
    if size is not None:
        if not isinstance(size, (tuple, list)):
            raise ValueError("size must be tuple or list")
        Validator.check_int(len(size), 2, Rel.EQ, "size", "bilinear")
        Validator.check_int(size[0], 1, Rel.GE, "size[0]", "bilinear")
        Validator.check_int(size[1], 1, Rel.GE, "size[1]", "bilinear")
        return size
    Validator.check_int(scale, 1, Rel.GE, "scale factor", "bilinear")
    ret = (scale * shape[2], scale * shape[3])
    return ret


class ResizeBilinear(Cell):
    r"""
    Samples the input tensor to the given size or scale_factor by using bilinear interpolate.

    Inputs:
        - **x** (Tensor) - Tensor to be resized. Input tensor must be a 4-D tensor with shape:
          math:`(batch, channels, height, width)`, with data type of float16 or float32.
        - **size** (Union[tuple[int], list[int]]): A tuple or list of 2 int elements '(new_height, new_width)',
          the new size of the tensor. One and only one of size and scale_factor can be set to None. Default: None.
        - **scale_factor** (int): The scale factor of new size of the tensor. The value should be positive integer.
          One and only one of size and scale_factor can be set to None. Default: None.
        - **align_corners** (bool): If true, rescale input by '(new_height - 1) / (height - 1)', which exactly aligns
          the 4 corners of images and resized images. If false, rescale by 'new_height / height'. Default: False.

    Outputs:
        Resized tensor.
        If size is set, the result is 4-D tensor with shape:math:`(batch, channels, new_height, new_width)`
        in float32.
        If scale is set, the result is 4-D tensor with shape:math:`(batch, channels, scale_factor * height,
        scale_factor * width)` in float32

    Raises:
        TypeError: If `size` is not one of tuple, list, None.
        TypeError: If `scale_factor` is neither int nor None.
        TypeError: If `align_corners` is not a bool.
        TypeError: If dtype of `x` is neither float16 nor float32.
        ValueError: If `size` and `scale_factor` are both None or not None.
        ValueError: If length of shape of `x` is not equal to 4.
        ValueError: If `scale_factor` is an int which is less than 1.
        ValueError: If `size` is a list or tuple whose length is not equal to 2.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> tensor = Tensor([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], mindspore.float32)
        >>> resize_bilinear = nn.ResizeBilinear()
        >>> result = resize_bilinear(tensor, size=(5,5))
        >>> print(result.shape)
        (1, 1, 5, 5)
    """

    def __init__(self):
        super(ResizeBilinear, self).__init__()

    def construct(self, x, size=None, scale_factor=None, align_corners=False):
        shape = bilinear(x.shape, size, scale_factor, align_corners)
        resize_bilinear = P.ResizeBilinear(shape, align_corners)
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
        - **input_x** (Tensor) - A 4-D tensor whose shape is [in_batch, in_depth, in_row, in_col] and
          data type is number.

    Outputs:
        Tensor, a 4-D tensor whose data type is same as `input_x`,
        and the shape is [out_batch, out_depth, out_row, out_col] where `out_batch` is the same as the `in_batch`.

        :math:`out\_depth = ksize\_row * ksize\_col * in\_depth`

        :math:`out\_row = (in\_row - (ksize\_row + (ksize\_row - 1) * (rate\_row - 1))) // stride\_row + 1`

        :math:`out\_col = (in\_col - (ksize\_col + (ksize\_col - 1) * (rate\_col - 1))) // stride\_col + 1`

    Raises:
        TypeError: If `ksizes`, `strides` or `rates` is neither a tuple nor list.
        ValueError: If shape of `ksizes`, `strides` or `rates` is not (1, x_row, x_col, 1).
        ValueError: If the second and third element of `ksizes`, `strides` or `rates` is less than 1.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> net = Unfold(ksizes=[1, 2, 2, 1], strides=[1, 2, 2, 1], rates=[1, 2, 2, 1])
        >>> image = Tensor(np.ones([2, 3, 6, 6]), dtype=mstype.float16)
        >>> output = net(image)
        >>> print(output.shape)
        (2, 12, 2, 2)
    """

    def __init__(self, ksizes, strides, rates, padding="valid"):
        super(Unfold, self).__init__()

        def _check_tuple_or_list(arg_name, arg_val, prim_name):
            Validator.check_value_type(f"{arg_name}s", ksizes, [tuple, list], self.cls_name)
            if len(arg_val) != 4 or arg_val[0] != 1 or arg_val[3] != 1:
                raise ValueError(f"For \'{prim_name}\' the format of {arg_name}s should be [1, {arg_name}_row, "
                                 f"{arg_name}_col, 1], but got {arg_val}.")
            if not isinstance(arg_val[1], int) or not isinstance(arg_val[2], int) or arg_val[1] < 1 or arg_val[2] < 1:
                raise ValueError(f"For '{prim_name}' the {arg_name}_row and {arg_name}_col in {arg_name}s should be an "
                                 f"positive integer number, but got {arg_name}_row is {arg_val[1]}, {arg_name}_col "
                                 f"is {arg_val[2]}")

        _check_tuple_or_list("ksize", ksizes, self.cls_name)
        _check_tuple_or_list("stride", strides, self.cls_name)
        _check_tuple_or_list("rate", rates, self.cls_name)
        ksizes = ksizes[0], ksizes[3], ksizes[1], ksizes[2]
        strides = strides[0], strides[3], strides[1], strides[2]
        rates = rates[0], rates[3], rates[1], rates[2]
        self.extract_image_patches = inner.ExtractImagePatches(ksizes, strides, rates, padding)

    def construct(self, input_x):
        result = self.extract_image_patches(input_x)
        return result


@constexpr
def tril(x_shape, x_dtype, k):
    Validator.check_int(len(x_shape), 1, Rel.GE, "x rank", "tril")
    Validator.check_is_int(k, "k value", "tril")
    mask = np.tril(np.ones(x_shape), k)
    return Tensor(mask, x_dtype)


class Tril(Cell):
    """
    Returns a tensor with elements above the kth diagonal zeroed.

    Inputs:
        - **x** (Tensor) - The input tensor.
        - **k** (Int) - The index of diagonal. Default: 0

    Outputs:
        Tensor, has the same type as input `x`.

    Raises:
        TypeError: If `k` is not an int.
        ValueError: If length of shape of `x` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1, 2], [3, 4]]))
        >>> tril = nn.Tril()
        >>> result = tril(x)
        >>> print(result)
        [[1   0]
         [3   4]]
    """

    def __init__(self):
        super(Tril, self).__init__()
        self.dtype = P.DType()
        self.mul = P.Mul()
        self.cast = P.Cast()

    def construct(self, x, k=0):
        assist = tril(x.shape, self.dtype(x), k)
        result = self.mul(self.cast(x, mstype.float32), self.cast(assist, mstype.float32))
        return self.cast(result, self.dtype(x))


@constexpr
def triu(x_shape, x_dtype, k):
    Validator.check_int(len(x_shape), 1, Rel.GE, "x rank", "triu")
    Validator.check_is_int(k, "k value", "triu")
    mask = np.triu(np.ones(x_shape), k)
    return Tensor(mask, x_dtype)


class Triu(Cell):
    """
    Returns a tensor with elements below the kth diagonal zeroed.

    Inputs:
        - **x** (Tensor) - The input tensor.
        - **k** (Int) - The index of diagonal. Default: 0

    Outputs:
        Tensor, has the same type as input `x`.

    Raises:
        TypeError: If `k` is not an int.
        ValueError: If length of shape of `x` is less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1, 2], [3, 4]]))
        >>> triu = nn.Triu()
        >>> result = triu(x)
        >>> print(result)
        [[1 2]
         [0 4]]
    """

    def __init__(self):
        super(Triu, self).__init__()
        self.dtype = P.DType()
        self.mul = P.Mul()
        self.cast = P.Cast()

    def construct(self, x, k=0):
        assist = triu(x.shape, self.dtype(x), k)
        result = self.mul(self.cast(x, mstype.float32), self.cast(assist, mstype.float32))
        return self.cast(result, self.dtype(x))


@constexpr
def _get_matrix_diag_assist(x_shape, x_dtype):
    Validator.check_int(len(x_shape), 1, Rel.GE, "x rank", "_get_matrix_diag_assist")
    base_eye = np.eye(x_shape[-1], x_shape[-1]).reshape(-1)
    assist = np.tile(base_eye, x_shape[:-1]).reshape(x_shape + (x_shape[-1],))
    return Tensor(assist, x_dtype)


@constexpr
def _get_matrix_diag_part_assist(x_shape, x_dtype):
    Validator.check_int(len(x_shape), 2, Rel.GE, "x rank", "_get_matrix_diag_part_assist")
    base_eye = np.eye(x_shape[-2], x_shape[-1]).reshape(-1)
    assist = np.tile(base_eye, x_shape[:-2]).reshape(x_shape)
    return Tensor(assist, x_dtype)


class MatrixDiag(Cell):
    r"""
    Returns a batched diagonal tensor with a given batched diagonal values.

    Assume `x` has :math:`k` dimensions :math:`[I, J, K, ..., N]`, then the output is a tensor of rank
    :math:`k+1` with dimensions :math:`[I, J, K, ..., N, N]` where:
    :math:`output[i, j, k, ..., m, n] = 1\{m=n\} * x[i, j, k, ..., n]`

    Inputs:
        - **x** (Tensor) - The diagonal values. It can be one of the following data types:
          float32, float16, int32, int8, and uint8.

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
        >>> print(output)
        [[ 1.  0.]
         [ 0. -1.]]
    """

    def __init__(self):
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
    :math:`output[i, j, k, ..., n] = x[i, j, k, ..., n, n]`

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
        >>> x = Tensor([[[-1, 0], [0, 1]], [[-1, 0], [0, 1]], [[-1, 0], [0, 1]]], mindspore.float32)
        >>> matrix_diag_part = nn.MatrixDiagPart()
        >>> output = matrix_diag_part(x)
        >>> print(output)
        [[-1.  1.]
         [-1.  1.]
         [-1.  1.]]
    """

    def __init__(self):
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
    dimensions :math:`[I, J, K, ..., min(M, N)]`. Then the output is a tensor of rank :math:`k+1` with dimensions
    :math:`[I, J, K, ..., M, N]` where:

    .. math::
        output[i, j, k, ..., m, n] = diagnoal[i, j, k, ..., n]\ for\ m == n

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
        super(MatrixSetDiag, self).__init__()
        self.matrix_set_diag = inner.MatrixSetDiag()
        self.dtype = P.DType()

    def construct(self, input_x, diagonal):
        x_shape = F.shape(input_x)
        x_dtype = self.dtype(input_x)
        assist = _get_matrix_diag_part_assist(x_shape, x_dtype)
        out_matrix_set_diag = self.matrix_set_diag(input_x, diagonal, assist)
        return out_matrix_set_diag

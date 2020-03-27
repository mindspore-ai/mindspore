# Copyright 2020 Huawei Technologies Co., Ltd
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
import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore._checkparam import check_int_positive, check_bool
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.functional import identity
from mindspore.common.parameter import Parameter
from mindspore._extends import cell_attr_register
from ..cell import Cell
from .activation import get_activation
from ..._checkparam import ParamValidator as validator


class Dropout(Cell):
    r"""
    Dropout layer for the input.

    Randomly set some elements of the input tensor to zero with probability :math:`1 - keep\_prob` during training
    using samples from a Bernoulli distribution.

    Note:
        Each channel will be zeroed out independently on every construct call.

        The outputs are scaled by a factor of :math:`\frac{1}{keep\_prob}` during training so
        that the output layer remains at a similar scale. During inference, this
        layer returns the same tensor as the input.

        This technique is proposed in paper `Dropout: A Simple Way to Prevent Neural Networks from Overfitting
        <http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf>`_ and proved to be effective to reduce
        over-fitting and prevents neurons from co-adaptation. See more details in `Improving neural networks by
        preventing co-adaptation of feature detectors
        <https://arxiv.org/pdf/1207.0580.pdf>`_.

    Args:
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. E.g. rate=0.9,
                   dropping out 10% of input units. Default: 0.5.
        seed0 (int): The first random seed. Default: 0.
        seed1 (int): The second random seed. Default: 0.
        dtype (:class:`mindspore.dtype`): Data type of input. Default: mindspore.float32.

    Raises:
        ValueError: If keep_prob is not in range (0, 1).

    Inputs:
        - **input** (Tensor) - An N-D Tensor.

    Outputs:
        Tensor, output tensor with the same shape as the input.

    Examples:
        >>> x = Tensor(np.ones([20, 16, 50]), mindspore.float32)
        >>> net = nn.Dropout(keep_prob=0.8)
        >>> net(x)
    """
    def __init__(self, keep_prob=0.5, seed0=0, seed1=0, dtype=mstype.float32):
        super(Dropout, self).__init__()
        if keep_prob <= 0 or keep_prob > 1:
            raise ValueError("dropout probability should be a number in range (0, 1], but got {}".format(keep_prob))
        validator.check_subclass("dtype", dtype, mstype.number_type)
        self.keep_prob = Tensor(keep_prob)
        self.seed0 = seed0
        self.seed1 = seed1
        self.dtype = dtype
        self.get_shape = P.Shape()
        self.dropout_gen_mask = P.DropoutGenMask(Seed0=seed0, Seed1=seed1)
        self.dropout_do_mask = P.DropoutDoMask()
        self.cast = P.Cast()

    def construct(self, x):
        shape = self.get_shape(x)
        dtype = P.DType()(x)
        keep_prob = self.cast(self.keep_prob, dtype)
        output = self.dropout_gen_mask(shape, keep_prob)
        return self.dropout_do_mask(x, output, keep_prob)

    def extend_repr(self):
        str_info = 'keep_prob={}, Seed0={}, Seed1={}, dtype={}' \
                   .format(self.keep_prob, self.seed0, self.seed1, self.dtype)
        return str_info


class Flatten(Cell):
    r"""
    Flatten layer for the input.

    Flattens a tensor without changing dimension of batch size on the 0-th axis.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, \ldots)` to be flattened.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(N, X)`, where :math:`X` is
        the product of the remaining dimensions.

    Examples:
        >>> net = nn.Flatten()
        >>> input = Tensor(np.array([[[1.2, 1.2], [2.1, 2.1]], [[2.2, 2.2], [3.2, 3.2]]]), mindspore.float32)
        >>> input.shape()
        (2, 2, 2)
        >>> net(input)
        [[1.2 1.2 2.1 2.1]
         [2.2 2.2 3.2 3.2]]
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def construct(self, x):
        return F.reshape(x, (F.shape(x)[0], -1))


class Dense(Cell):
    r"""
    The fully connected layer.

    Applies dense-connected layer for the input. This layer implements the operation as:

    .. math::
        \text{outputs} = \text{activation}(\text{inputs} * \text{kernel} + \text{bias}),

    where :math:`\text{activation}` is the activation function passed as the activation
    argument (if passed in), :math:`\text{activation}` is a weight matrix with the same
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
        activation (str): Regularizer function applied to the output of the layer, eg. 'relu'. Default: None.

    Raises:
        ValueError: If weight_init or bias_init shape is incorrect.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(N, in_channels)`.

    Outputs:
        Tensor of shape :math:`(N, out_channels)`.

    Examples:
        >>> net = nn.Dense(3, 4)
        >>> input = Tensor(np.random.randint(0, 255, [2, 3]), mindspore.float32)
        >>> net(input)
        [[ 2.5246444   2.2738023   0.5711005  -3.9399147 ]
         [ 1.0739875   4.0155234   0.94188046 -5.459526  ]]
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
        self.in_channels = check_int_positive(in_channels)
        self.out_channels = check_int_positive(out_channels)
        self.has_bias = check_bool(has_bias)

        if isinstance(weight_init, Tensor):
            if weight_init.dim() != 2 or weight_init.shape()[0] != out_channels or \
               weight_init.shape()[1] != in_channels:
                raise ValueError("weight_init shape error")

        self.weight = Parameter(initializer(weight_init, [out_channels, in_channels]), name="weight")

        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.dim() != 1 or bias_init.shape()[0] != out_channels:
                    raise ValueError("bias_init shape error")

            self.bias = Parameter(initializer(bias_init, [out_channels]), name="bias")

        self.matmul = P.MatMul(transpose_b=True)
        self.bias_add = P.BiasAdd()

        self.activation = get_activation(activation)
        self.activation_flag = self.activation is not None

    def construct(self, x):
        output = self.matmul(x, self.weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        if self.activation_flag:
            return self.activation(output)
        return output

    def extend_repr(self):
        str_info = 'in_channels={}, out_channels={}, weight={}, has_bias={}' \
                   .format(self.in_channels, self.out_channels, self.weight, self.has_bias)
        if self.has_bias:
            str_info = str_info + ', bias={}'.format(self.bias)

        if self.activation_flag:
            str_info = str_info + ', activation={}'.format(self.activation)

        return str_info


class ClipByNorm(Cell):
    r"""
    Clips tensor values to a maximum :math:`L_2`-norm.

    The output of this layer remains the same if the :math:`L_2`-norm of the input tensor
    is not greater than the argument clip_norm. Otherwise the tensor will be normalized as:

    .. math::
        \text{output}(X) = \frac{\text{clip_norm} * X}{L_2(X)},

    where :math:`L_2(X)` is the :math:`L_2`-norm of :math:`X`.

    Inputs:
        - **input** (Tensor) - Tensor of shape N-D.
        - **clip_norm** (Tensor) - A scalar Tensor of shape :math:`()` or :math:`(1)` and of
          the same type as the input Tensor.

    Outputs:
        Tensor, clipped tensor with the same shape as the input.

    Examples:
        >>> net = nn.ClipByNorm()
        >>> input = Tensor(np.random.randint(0, 10, [4, 16]), mindspore.float32)
        >>> clip_norm = Tensor(np.array([100]).astype(np.float32))
        >>> net(input, clip_norm)

    """
    def __init__(self):
        super(ClipByNorm, self).__init__()
        self.reduce_sum = P.ReduceSum(keep_dims=True)
        self.select_ = P.Select()
        self.greater_ = P.Greater()
        self.axis = ()
        self.cast = P.Cast()
        self.zero = Tensor(np.array([0.0]).astype(np.float32))
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
        cond = self.greater_(l2sum, self.zero)
        ones_ = self.fill(self.dtype(cond), self.shape(cond), 1.0)

        l2sum_safe = self.select_(cond, l2sum, self.cast(ones_, self.dtype(l2sum)))
        l2norm = self.select_(cond, self.sqrt(l2sum_safe), l2sum)

        intermediate = x * clip_norm
        max_norm = self.max_op(l2norm, clip_norm)
        values_clip = self.cast(intermediate, mstype.float32) / self.expand_dims(max_norm, -1)
        values_clip = self.reshape(values_clip, self.shape(x))
        values_clip = identity(values_clip)
        return values_clip


class Norm(Cell):
    """
    Computes the norm of vectors, currently including Euclidean norm, i.e., :math:`L_2`-norm.

    Args:
        axis (tuple): The axis over which to compute vector norms. Default: ().
        keep_dims (bool): If True, the axis indicated in `axis` are kept with size 1. Otherwise,
                   the dimensions in `axis` are removed from the output shape. Default: False.

    Inputs:
        - **input** (Tensor) - Tensor which is not empty.

    Outputs:
        Tensor, output tensor with dimensions in 'axis' reduced to 1 will be returned if 'keep_dims' is True;
        otherwise a Tensor with dimensions in 'axis' removed is returned.

    Examples:
        >>> net = nn.Norm(axis=0)
        >>> input = Tensor(np.random.randint(0, 10, [4, 16]), mindspore.float32)
        >>> net(input)
    """
    def __init__(self, axis=(), keep_dims=False):
        super(Norm, self).__init__()
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
        str_info = 'axis={}, keep_dims={}'.format(self.axis, self.keep_dims)
        return str_info


class OneHot(Cell):
    """
    Returns a one-hot tensor.

    The locations represented by indices in argument 'indices' take value on_value,
    while all other locations take value off_value.

    Note:
        If the input indices is rank :math:`N`, the output will have rank :math:`N+1`. The new
        axis is created at dimension `axis`.

    Args:
        axis (int): Features x depth if axis == -1, depth x features
                    if axis == 0. Default: -1.
        depth (int): A scalar defining the depth of the one hot dimension. Default: 1.
        on_value (float): A scalar defining the value to fill in output[i][j]
                          when indices[j] = i. Default: 1.0.
        off_value (float): A scalar defining the value to fill in output[i][j]
                           when indices[j] != i. Default: 0.0.
        dtype (:class:`mindspore.dtype`): Default: mindspore.float32.

    Inputs:
        - **indices** (Tensor) - A tensor of indices of data type mindspore.int32 and arbitrary shape.

    Outputs:
        Tensor, the one-hot tensor of data type 'dtype' with dimension at 'axis' expanded to 'depth' and filled with
        on_value and off_value.

    Examples:
        >>> net = nn.OneHot(depth=4, axis=1)
        >>> indices = Tensor([[1, 3], [0, 2]], dtype=mindspore.int32)
        >>> net(indices)
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
        self.on_value = Tensor(on_value, dtype)
        self.off_value = Tensor(off_value, dtype)

    def construct(self, indices):
        return self.onehot(indices, self.depth, self.on_value, self.off_value)

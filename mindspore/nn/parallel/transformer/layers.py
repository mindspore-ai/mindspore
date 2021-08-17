# Copyright 2021 Huawei Technologies Co., Ltd
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
The basic layer of the Transformer Networks. This is an experimental interface that is subject to
change and/or deletion.
"""

from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore._extends import cell_attr_register
from mindspore.nn.cell import Cell
from mindspore.nn.layer import Dense


class _LayerNorm(Cell):
    r"""
        A self-defined layer norm operation using reduce sum and reduce mean

        Args:
            normalized_shape (tuple): The shape of the input tensor
            eps (float): The epsilon value of the denominator. Default 1e-5.
            param_init_type: The param init type.
        Inputs:
            - **x** (Tensor) - Tensor of shape :math:`(batch, seq\_length, hidden\_size)`.

        Outputs:
            Tensor of shape :math:`(batch, seq_length, hidden_size)`.
    """

    def __init__(self, normalized_shape, eps=1e-5, param_init_type=mstype.float32):
        super(_LayerNorm, self).__init__()
        if param_init_type not in [mstype.float32, mstype.float16]:
            raise TypeError(f"param type should in [float32, float16], but found type {type(param_init_type)}")
        self.gamma = Parameter(initializer('ones', normalized_shape, param_init_type), name="gamma",
                               parallel_optimizer=False)
        self.beta = Parameter(initializer('zeros', normalized_shape, param_init_type), name="beta",
                              parallel_optimizer=False)
        self.mean = P.ReduceMean(keep_dims=True)
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.sub1 = P.Sub()
        self.sub2 = P.Sub()
        self.add = P.TensorAdd()
        self.eps = eps
        self.mul = P.Mul()
        self.add2 = P.TensorAdd()
        self.real_div = P.RealDiv()

    def construct(self, x):
        r"""
          x : batch x seq_length x hidden_size
        """
        mean = self.mean(x, -1)
        diff = self.sub1(x, mean)
        variance = self.mean(self.square(diff), -1)
        variance_eps = self.sqrt(self.add(variance, self.eps))
        output = self.real_div(diff, variance_eps)
        output = self.add2(self.mul(output, self.gamma), self.beta)
        return output

    def shard(self, strategy):
        r"""
        Set the shard for the layer norm. the strategy size should be equal to the inputs.

        Note:
            It is valid only in semi auto parallel or auto parallel mode.
            In other parallel modes, strategies set here will be ignored.

        Args:
            strategy (tuple): The strategy for the dropout. Should be the same shape as the inputs.
        Examples:
            >>> net = nn.parallel.transformer.LayerNorm(normalized_shape=(1024, 10))
            >>> net.shard(((10, 2, 1),))
        """
        self.mean.shard(strategy)
        self.square.shard(strategy)
        self.sqrt.shard(strategy)
        self.sub1.shard((strategy[0], strategy[0]))
        self.sub2.shard((strategy[0], strategy[0]))
        self.add.shard((strategy[0], ()))
        self.mul.shard((strategy[0], (1,)))
        self.add2.shard((strategy[0], (1,)))
        self.real_div.shard((strategy[0], strategy[0]))

        return self


class _Linear(Dense):
    r"""
    The dense connected layer. Once the parallel mode is enabled, the input shape should be
    3-D tensor.

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
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        activation (str): activate function applied to the output of the fully connected layer,
            eg. 'ReLU'.Default: None.
        compute_dtype (mstype): The computation type. Default: mstype.float16
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
        ``Ascend`` ``GPU``
    """

    @cell_attr_register(attrs=['has_bias', 'in_channels', 'out_channels', 'shard_output', 'activation'])
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None,
                 transpose_b=True,
                 param_init_type=mstype.float32,
                 compute_dtype=mstype.float16):
        super(_Linear, self).__init__(in_channels=in_channels,
                                      out_channels=out_channels,
                                      weight_init=weight_init,
                                      bias_init=bias_init,
                                      has_bias=has_bias,
                                      activation=activation)
        if param_init_type not in [mstype.float32, mstype.float16]:
            raise TypeError(f"param type should in [float32, float16], but found type {type(param_init_type)}")
        if activation and not isinstance(activation, str):
            raise ValueError("Activation can only be str, but found type {}".format(activation))
        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 2 or weight_init.shape[0] != out_channels or \
                    weight_init.shape[1] != in_channels:
                raise ValueError("Weight init shape error.")
        if transpose_b:
            weight_shape = [out_channels, in_channels]
        else:
            weight_shape = [in_channels, out_channels]
        self.weight = Parameter(initializer(weight_init, weight_shape, param_init_type), name="weight")
        self.matmul = P.MatMul(transpose_b=transpose_b)
        self.bias = None
        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError("Bias init shape error.")
            self.bias = Parameter(initializer(bias_init, [out_channels], param_init_type), name="bias")
            self.bias_add = P.BiasAdd()
        self.act_name = activation
        self.dtype = compute_dtype
        self.cast = P.Cast()
        self.has_bias = self.has_bias

    def construct(self, x):
        out_shape = P.Shape()(x)[:-1] + (self.out_channels,)
        x = P.Reshape()(x, (-1, self.in_channels))
        weight = self.cast(self.weight, self.dtype)
        x = self.matmul(x, weight)
        x = self.bias_add(x, self.cast(self.bias, self.dtype))
        output = P.Reshape()(x, out_shape)
        if self.activation_flag:
            output = self.activation(output)
        return output

    def shard(self, strategy_matmul, strategy_bias=None, strategy_activation=None):
        r"""
         Set the shard for the linear. the strategy size should be equal to the inputs.

         Note:
            It is valid only in semi auto parallel or auto parallel mode.
            In other parallel modes, strategies set here will be ignored.

         Args:
             strategy_matmul (tuple): The strategy for the matmul. Should be the same shape as the inputs.
             strategy_bias (tuple): The strategy for the bias_add. Should be the same shape as the inputs.
             strategy_activation (tuple): The strategy for the strategy_activation. Should be the same shape as
                the inputs.
         """
        self.matmul.shard(strategy_matmul)
        if self.has_bias:
            self.bias_add.shard(strategy_bias)
        if self.activation_flag:
            getattr(self.activation, self.act_name).shard(strategy_activation)

        return self

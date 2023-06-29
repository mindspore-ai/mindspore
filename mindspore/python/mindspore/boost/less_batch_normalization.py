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
"""less Batch Normalization"""
from __future__ import absolute_import

import numpy as np
from mindspore.nn.cell import Cell
from mindspore.nn.layer import Dense
from mindspore.ops import operations as P
from mindspore.common import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer


__all__ = ["CommonHeadLastFN", "LessBN"]


class CommonHeadLastFN(Cell):
    r"""
    The last full Normalization layer.

    This layer implements the operation as:

    .. math::
        \text{inputs} = \text{norm}(\text{inputs})
        \text{kernel} = \text{norm}(\text{kernel})
        \text{outputs} = \text{multiplier} * (\text{inputs} * \text{kernel} + \text{bias}),

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input x. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: ``True``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), mindspore.float32)
        >>> net = CommonHeadLastFN(3, 4)
        >>> output = net(input)
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True):
        super(CommonHeadLastFN, self).__init__()
        weight_shape = [out_channels, in_channels]
        self.weight = Parameter(initializer(weight_init, weight_shape), requires_grad=True, name='weight')
        self.x_norm = P.L2Normalize(axis=1)
        self.w_norm = P.L2Normalize(axis=1)
        self.fc = P.MatMul(transpose_a=False, transpose_b=True)
        self.multiplier = Parameter(Tensor(np.ones([1]), mstype.float32), requires_grad=True, name='multiplier')
        self.has_bias = has_bias
        if self.has_bias:
            bias_shape = [out_channels]
            self.bias_add = P.BiasAdd()
            self.bias = Parameter(initializer(bias_init, bias_shape), requires_grad=True, name='bias')

    def construct(self, x):
        x = self.x_norm(x)
        w = self.w_norm(self.weight)
        x = self.fc(x, w)
        if self.has_bias:
            x = self.bias_add(x, self.bias)
        x = self.multiplier * x
        return x


class LessBN(Cell):
    """
    Reduce the number of BN automatically to improve the network performance
    and ensure the network accuracy.

    Args:
        network (Cell): Network to be modified.
        fn_flag (bool): Replace FC with FN. Default: ``False`` .

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, Parameter, nn
        >>> import mindspore.ops as ops
        >>> from mindspore.nn import WithLossCell
        >>> from mindspore import dtype as mstype
        >>> from mindspore import boost
        >>>
        >>> class Net(nn.Cell):
        ...    def __init__(self, in_features, out_features):
        ...        super(Net, self).__init__()
        ...        self.weight = Parameter(Tensor(np.ones([in_features, out_features]).astype(np.float32)),
        ...                                name='weight')
        ...        self.matmul = ops.MatMul()
        ...
        ...    def construct(self, x):
        ...        output = self.matmul(x, self.weight)
        ...        return output
        >>> size, in_features, out_features = 16, 16, 10
        >>> net = Net(in_features, out_features)
        >>> loss = nn.MSELoss()
        >>> optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> net_with_loss = WithLossCell(net, loss)
        >>> inputs = Tensor(np.ones([size, in_features]).astype(np.float32))
        >>> label = Tensor(np.zeros([size, out_features]).astype(np.float32))
        >>> train_network = boost.LessBN(net_with_loss)
        >>> output = train_network(inputs, label)
    """

    def __init__(self, network, fn_flag=False):
        super(LessBN, self).__init__()
        self.network = network
        self.network.set_boost("less_bn")
        self.network.update_cell_prefix()
        if fn_flag:
            self._convert_to_less_bn_net(self.network)
        self.network.add_flags(defer_inline=True)

    @staticmethod
    def _convert_dense(subcell):
        """
        convert dense cell to FN cell
        """
        prefix = subcell.param_prefix
        new_subcell = CommonHeadLastFN(subcell.in_channels,
                                       subcell.out_channels,
                                       subcell.weight,
                                       subcell.bias,
                                       False)
        new_subcell.update_parameters_name(prefix + '.')

        return new_subcell

    def construct(self, *inputs):
        return self.network(*inputs)

    def _convert_to_less_bn_net(self, net):
        """
        convert network to less_bn network
        """
        cells = net.name_cells()
        dense_name = []
        dense_list = []
        for name in cells:
            subcell = cells[name]
            if subcell == net:
                continue
            if isinstance(subcell, (Dense)):
                dense_name.append(name)
                dense_list.append(subcell)
            else:
                self._convert_to_less_bn_net(subcell)

        if dense_list:
            new_subcell = LessBN._convert_dense(dense_list[-1])
            net.insert_child_to_cell(dense_name[-1], new_subcell)

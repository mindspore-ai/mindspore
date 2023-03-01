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
"""Aggregator."""
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore._checkparam import Validator
from mindspore._extends import cell_attr_register
from mindspore.common.initializer import initializer
from mindspore.nn.layer.activation import get_activation
from mindspore.ops import functional as F
from mindspore.ops import operations as P


class GNNFeatureTransform(nn.Cell):
    r"""
    The GNN featuren transform layer for input.

    Applies linear transformation for the input feature. This layer implements the operation as:

    .. math::
        \text{outputs} = \text{inputs} * \text{kernel} + \text{bias},

    where :math:`\text{activation}` is the activation function passed as the activation
    argument (if passed in),:math:`\text{activation}` is a weight matrix with the same
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

    Raises:
        ValueError: If weight_init or bias_init shape is incorrect.

    Inputs:
        - **input_x** (Tensor) - The first tensor to be multiplied. The shape of the tensor is :math:`(*B, N, C)`,
        where :math:`*B` represents the batch size which can be multidimensional, :math:`N` and :math:`C` are the
        size of the last two dimensions. If `transpose_a` is True, its shape should be :math:`(*B, C, N)`.

    Outputs:
        Tensor, the shape of the output tensor is :math:`(*B, N, M)`.

    Examples:
        >>> net = nn.GNNFeatureTransform(3, 4)
        >>> input = Tensor(np.random.randint(0, 255, [2, 3]), mindspore.float32)
        >>> net(input)
        [[ 2.5246444   2.2738023   0.5711005  -3.9399147 ]
         [ 1.0739875   4.0155234   0.94188046 -5.459526  ]]
    """

    @cell_attr_register
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True):
        super(GNNFeatureTransform, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels)
        self.out_channels = Validator.check_positive_int(out_channels)
        self.has_bias = Validator.check_bool(has_bias)

        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 2 or weight_init.shape[0] != out_channels or \
                    weight_init.shape[1] != in_channels:
                raise ValueError("weight_init shape error")

        self.weight = Parameter(initializer(weight_init, [out_channels, in_channels]), name="weight")

        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError("bias_init shape error")

            self.bias = Parameter(initializer(bias_init, [out_channels]), name="bias")

        self.matmul = P.MatMul(transpose_b=True)
        self.bias_add = P.BiasAdd()

    def construct(self, x):
        tensor_shape = F.shape(x)
        input_feature = F.reshape(x, (tensor_shape[0] * tensor_shape[1], tensor_shape[2]))
        output = self.matmul(input_feature, self.weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        output = F.reshape(output, (tensor_shape[0], tensor_shape[1], self.out_channels))
        return output

    def extend_repr(self):
        s = 'in_channels={}, out_channels={}'.format(self.in_channels, self.out_channels)
        if self.has_bias:
            s += ', has_bias={}'.format(self.has_bias)
        return s


class _BaseAggregator(nn.Cell):
    """
    Base Aggregator of GNN

    Args:
        feature_in_dim (int): Node or edge input feature dim.
        feature_out_dim (int): Node or edge outpout feature dim.
        use_fc (bool): Specifies whether a linear transformation before message is aggregated. Default: True
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input x. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        dropout_ratio (float): The keep rate of dropout layer, greater than 0 and less equal than 1. Default: None.
        activation (str): Regularizer function applied to the output of the layer, eg. 'relu'. Default: None.

    Examples:
        >>> class MyAggregator(_BaseAggregator):
        >>>    def __init__(self):
        >>>        super(MyAggregator, self).__init__(self, feature_in_dim, feature_out_dim)
        >>>        self.reduce_mean = P.ReduceSum()
        >>>
        >>>    def construct(self, x):
        >>>        return self.reduce_mean(x, 1)
    """

    def __init__(self,
                 feature_in_dim,
                 feature_out_dim,
                 use_fc=True,
                 weight_init="normal",
                 bias_init="zeros",
                 has_bias=True,
                 dropout_ratio=None,
                 activation=None):
        super(_BaseAggregator, self).__init__()
        self.in_dim = feature_in_dim
        self.out_dim = feature_out_dim
        self.use_fc = use_fc
        if self.use_fc:
            self.weight_init = weight_init
            self.bias_init = bias_init
            self.has_bias = has_bias
            self.fc = GNNFeatureTransform(self.in_dim,
                                          self.out_dim,
                                          weight_init=self.weight_init,
                                          bias_init=self.bias_init,
                                          has_bias=self.has_bias)
        self.dropout_ratio = dropout_ratio
        if self.dropout_ratio is not None:
            self.dropout = nn.Dropout(p=1.0 - self.dropout_ratio)
        self.dropout_flag = self.dropout_ratio is not None
        self.activation = get_activation(activation)
        self.activation_flag = self.activation is not None

    def construct(self, **kward):
        """Must be overridden by all subclasses."""
        raise NotImplementedError


class MeanAggregator(_BaseAggregator):
    """
    Mean Aggregator of GNN

    Args:
        feature_in_dim (int): Node or edge input feature dim.
        feature_out_dim (int): Node or edge outpout feature dim.
        use_fc (bool): Specifies whether a linear transformation before message is aggregated. Default: True
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype
            is same as input x. The values of str refer to the function `initializer`. Default: 'normal'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input x. The values of str refer to the function `initializer`. Default: 'zeros'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: True.
        dropout_ratio (float): The keep rate of dropout layer, greater than 0 and less equal than 1. Default: None.
        activation (str): Regularizer function applied to the output of the layer, eg. 'relu'. Default: None.

    Examples:
        >>> net = MeanAggregator(32, 64, activation="relu", dropout=0.5)
        >>> input_data = Tensor(np.array(np.random.rand(32, 3, 32), dtypy=np.float32))
        >>> output = net(input_data)
    """

    def __init__(self,
                 feature_in_dim,
                 feature_out_dim,
                 use_fc=True,
                 weight_init="normal",
                 bias_init="zeros",
                 has_bias=True,
                 dropout_ratio=None,
                 activation=None):
        super(MeanAggregator, self).__init__(
            feature_in_dim,
            feature_out_dim,
            use_fc,
            weight_init,
            bias_init,
            has_bias,
            dropout_ratio,
            activation)
        self.reduce_mean = P.ReduceMean(keep_dims=False)

    def construct(self, input_feature):
        if self.use_fc:
            input_feature = self.fc(input_feature)
        if self.dropout_flag:
            input_feature = self.dropout(input_feature)
        if self.activation_flag:
            input_feature = self.activation(input_feature)
        output_feature = self.reduce_mean(input_feature, 1)
        return output_feature


class AttentionHead(nn.Cell):
    """
    Attention Head for Graph Attention Networks.

    Args:
        in_channel (int): The number of input channel, input feature dim.
        out_channel (int): The number of output channel, output feature dim.
        in_drop_ratio (float): Input feature dropout ratio, default 0.0.
        coef_drop_ratio (float): Coefficient dropout ratio, default 0.0.
        residual (bool): Whether to use residual connection, default False.
        coef_activation (Cell): The attention coefficient activation function,
            default nn.LeakyReLU().
        activation (Cell): The output activation function, default nn.ELU().

    Inputs:
        - **input_feature** (Tensor) - Tensor of shape : (batch_size, num_nodes, feature_dim).
        - **bias_mat** (Tensor) - Tensor of shape : (batch_size, num_nodes, num_nodes).

    Examples:
        >>> head = AttentionHead(1433,
                                 8,
                                 in_drop_ratio=0.6,
                                 coef_drop_ratio=0.6,
                                 residual=False)
        >>> input_data = Tensor(np.array(np.random.rand(1, 2708, 1433), dtypy=np.float32))
        >>> output = net(input_data)
    """

    def __init__(self,
                 in_channel,
                 out_channel,
                 in_drop_ratio=0.0,
                 coef_drop_ratio=0.0,
                 residual=False,
                 coef_activation=nn.LeakyReLU(),
                 activation=nn.ELU()):
        super(AttentionHead, self).__init__()
        self.in_channel = Validator.check_positive_int(in_channel)
        self.out_channel = Validator.check_positive_int(out_channel)
        self.in_drop_ratio = in_drop_ratio
        self.in_drop = nn.Dropout(p=in_drop_ratio)
        self.in_drop_2 = nn.Dropout(p=in_drop_ratio)
        self.feature_transform = GNNFeatureTransform(
            in_channels=self.in_channel,
            out_channels=self.out_channel,
            has_bias=False)

        self.f_1_transform = GNNFeatureTransform(
            in_channels=self.out_channel,
            out_channels=1)
        self.f_2_transform = GNNFeatureTransform(
            in_channels=self.out_channel,
            out_channels=1)
        self.softmax = nn.Softmax()

        self.coef_drop = nn.Dropout(p=coef_drop_ratio)
        self.batch_matmul = P.BatchMatMul()
        self.bias_add = P.BiasAdd()
        self.bias = Parameter(initializer('zeros', self.out_channel), name='bias')
        self.residual = Validator.check_bool(residual)
        if self.residual:
            if in_channel != out_channel:
                self.residual_transform_flag = True
                self.residual_transform = GNNFeatureTransform(
                    in_channels=self.in_channel,
                    out_channels=self.out_channel)
            else:
                self.residual_transform = None
        self.coef_activation = coef_activation
        self.activation = activation

    def construct(self, input_feature, bias_mat):
        input_feature = self.in_drop(input_feature)

        feature = self.feature_transform(input_feature)
        # self attention following the author
        f_1 = self.f_1_transform(feature)
        f_2 = self.f_2_transform(feature)
        logits = f_1 + P.Transpose()(f_2, (0, 2, 1))
        logits = self.coef_activation(logits) + bias_mat
        coefs = self.softmax(logits)

        coefs = self.coef_drop(coefs)
        feature = self.in_drop_2(feature)

        ret = self.batch_matmul(coefs, feature)
        ret = P.Squeeze(0)(ret)
        ret = self.bias_add(ret, self.bias)
        ret = P.ExpandDims()(ret, 0)
        # residual connection
        if self.residual:
            if self.residual_transform_flag:
                res = self.residual_transform(input_feature)
                ret = ret + res
            else:
                ret = ret + input_feature
        # activation
        if self.activation is not None:
            ret = self.activation(ret)
        return ret


class AttentionAggregator(nn.Cell):
    """
    Attention Head for Graph Attention Networks，can be regarded as one
        GAT layer.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        num_heads (int): Number of attention heads for this layer, default 1.
        in_drop_ratio (float): Input feature dropout ratio, default 0.0.
        coef_drop_ratio (float): Coefficient dropout ratio, default 0.0.
        activation (Cell): The output activation function, default nn.ELU().
        residual (bool): Whether to use residual connection, default False.
        output_transform (str['concat', 'sum']): output transform for a layer,
            default 'concat'

    Inputs:
        - **input_feature** (Tensor) - Tensor of shape : (batch_size, num_nodes, feature_dim).
        - **bias_mat** (Tensor) - Tensor of shape : (batch_size, num_nodes, num_nodes).

    Examples:
        >>> input_data = Tensor(np.array(np.random.rand(1, 2708, 1433), dtype=np.float32))
        >>> biases = Tensor(np.array(np.random.rand(1, 2708, 2708), dtype=np.float32))
        >>> net = AttentionAggregator(1433,
                                      8,
                                      8)
        >>> net(input_data, biases)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads=1,
                 in_drop=0.0,
                 coef_drop=0.0,
                 activation=nn.ELU(),
                 residual=False,
                 output_transform='concat'):
        super(AttentionAggregator, self).__init__()
        self.num_heads = num_heads
        self.attns = []
        for _ in range(num_heads):
            self.attns.append(AttentionHead(in_channels,
                                            out_channels,
                                            in_drop_ratio=in_drop,
                                            coef_drop_ratio=coef_drop,
                                            activation=activation,
                                            residual=residual))
        self.attns = nn.layer.CellList(self.attns)
        if output_transform == 'concat':
            self.out_trans = P.Concat(-1)
        elif output_transform == 'sum':
            self.out_trans = P.AddN()
        else:
            raise ValueError

    def construct(self, input_data, bias_mat):
        res = ()
        for i in range(self.num_heads):
            res += (self.attns[i](input_data, bias_mat),)
        return self.out_trans(res)

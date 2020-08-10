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
"""GCN."""
import numpy as np
from mindspore import nn
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore.nn.layer.activation import get_activation


def glorot(shape):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = np.random.uniform(-init_range, init_range, shape).astype(np.float32)
    return Tensor(initial)


class GraphConvolution(nn.Cell):
    """
    GCN graph convolution layer.

    Args:
        feature_in_dim (int): The input feature dimension.
        feature_out_dim (int): The output feature dimension.
        dropout_ratio (float): Dropout ratio for the dropout layer. Default: None.
        activation (str): Activation function applied to the output of the layer, eg. 'relu'. Default: None.

    Inputs:
        - **adj** (Tensor) - Tensor of shape :math:`(N, N)`.
        - **input_feature** (Tensor) - Tensor of shape :math:`(N, C)`.

    Outputs:
        Tensor, output tensor.
    """

    def __init__(self,
                 feature_in_dim,
                 feature_out_dim,
                 dropout_ratio=None,
                 activation=None):
        super(GraphConvolution, self).__init__()
        self.in_dim = feature_in_dim
        self.out_dim = feature_out_dim
        self.weight_init = glorot([self.out_dim, self.in_dim])
        self.fc = nn.Dense(self.in_dim,
                           self.out_dim,
                           weight_init=self.weight_init,
                           has_bias=False)
        self.dropout_ratio = dropout_ratio
        if self.dropout_ratio is not None:
            self.dropout = nn.Dropout(keep_prob=1-self.dropout_ratio)
        self.dropout_flag = self.dropout_ratio is not None
        self.activation = get_activation(activation)
        self.activation_flag = self.activation is not None
        self.matmul = P.MatMul()

    def construct(self, adj, input_feature):
        """
        GCN graph convolution layer.
        """
        dropout = input_feature
        if self.dropout_flag:
            dropout = self.dropout(dropout)

        fc = self.fc(dropout)
        output_feature = self.matmul(adj, fc)

        if self.activation_flag:
            output_feature = self.activation(output_feature)
        return output_feature


class GCN(nn.Cell):
    """
    GCN architecture.

    Args:
        config (ConfigGCN): Configuration for GCN.
        adj (numpy.ndarray): Numbers of block in different layers.
        feature (numpy.ndarray): Input channel in each layer.
        output_dim (int): The number of output channels, equal to classes num.
    """

    def __init__(self, config, adj, feature, output_dim):
        super(GCN, self).__init__()
        self.adj = Tensor(adj)
        self.feature = Tensor(feature)
        input_dim = feature.shape[1]
        self.layer0 = GraphConvolution(input_dim, config.hidden1, activation="relu", dropout_ratio=config.dropout)
        self.layer1 = GraphConvolution(config.hidden1, output_dim, dropout_ratio=None)

    def construct(self):
        output0 = self.layer0(self.adj, self.feature)
        output1 = self.layer1(self.adj, output0)
        return output1

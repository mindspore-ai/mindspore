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
"""Graph Attention Networks."""
import mindspore.nn as nn
from mindspore import _checkparam as Validator

from aggregator import AttentionAggregator


class GAT(nn.Cell):
    """
    Graph Attention Network

    Args:
        ftr_dims (int): Initial feature dimensions.
        num_class (int): Num of class to identify.
        num_nodes (int): Num of nodes in this graph.
        hidden_units (list[int]): Num of hidden units at each layer.
        num_heads (list[int]): Num of heads at each layer.
        attn_drop (float): Drop out ratio of attention coefficient,
            default 0.0.
        ftr_drop (float): Drop out ratio of feature, default 0.0.
        activation (Cell): Activation Function for output layer, default
            nn.Elu().
        residual (bool): Whether to use residual connection between
            intermediate layers, default False.

    Examples:
        >>> ft_sizes = 1433
        >>> num_class = 7
        >>> num_nodes = 2708
        >>> hid_units = [8]
        >>> n_heads = [8, 1]
        >>> activation = nn.ELU()
        >>> residual = False
        >>> input_data = Tensor(
                np.array(np.random.rand(1, 2708, 1433), dtype=np.float32))
        >>> biases = Tensor(np.array(np.random.rand(1, 2708, 2708), dtype=np.float32))
        >>> net = GAT(ft_sizes,
                      num_class,
                      num_nodes,
                      hidden_units=hid_units,
                      num_heads=n_heads,
                      attn_drop=0.6,
                      ftr_drop=0.6,
                      activation=activation,
                      residual=residual)
        >>> output = net(input_data, biases)
    """

    def __init__(self,
                 ftr_dims,
                 num_class,
                 num_nodes,
                 hidden_units,
                 num_heads,
                 attn_drop=0.0,
                 ftr_drop=0.0,
                 activation=nn.ELU(),
                 residual=False):
        super(GAT, self).__init__()
        self.ftr_dims = Validator.check_positive_int(ftr_dims)
        self.num_class = Validator.check_positive_int(num_class)
        self.num_nodes = Validator.check_positive_int(num_nodes)
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.ftr_drop = ftr_drop
        self.activation = activation
        self.residual = Validator.check_bool(residual)
        self.layers = []
        # first layer
        self.layers.append(AttentionAggregator(
            self.ftr_dims,
            self.hidden_units[0],
            self.num_heads[0],
            self.ftr_drop,
            self.attn_drop,
            self.activation,
            residual=False))
        # intermediate layer
        for i in range(1, len(self.hidden_units)):
            self.layers.append(AttentionAggregator(
                self.hidden_units[i-1]*self.num_heads[i-1],
                self.hidden_units[i],
                self.num_heads[i],
                self.ftr_drop,
                self.attn_drop,
                self.activation,
                residual=self.residual))
        # output layer
        self.layers.append(AttentionAggregator(
            self.hidden_units[-1]*self.num_heads[-2],
            self.num_class,
            self.num_heads[-1],
            self.ftr_drop,
            self.attn_drop,
            activation=None,
            residual=False,
            output_transform='sum'))
        self.layers = nn.layer.CellList(self.layers)

    def construct(self, input_data, bias_mat):
        for cell in self.layers:
            input_data = cell(input_data, bias_mat)
        return input_data/self.num_heads[-1]

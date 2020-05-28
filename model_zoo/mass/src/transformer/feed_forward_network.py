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
"""Forward network with two fc layers."""
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P

from .residual_conn import ResidualConnection
from .components import LayerNorm


class FeedForwardNet(nn.Cell):
    """
    Feed Forward Network (contain 2 fc layers).

    Args:
        in_channels (int): Dimensions of input matrix.
        hidden_size (int): Hidden size.
        out_channels (int): Dimensions of output matrix.
        hidden_act (str): Activation function.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        hidden_dropout_prob (float): The dropout probability for hidden outputs. Default: 0.1.
        dropout (float): Dropout in residual block. Default: 0.1.
        compute_type (mstype): Compute type in FeedForward. Default: mstype.float32.

    Returns:
        Tensor, shape of (N, T, D).
    """

    def __init__(self,
                 in_channels,
                 hidden_size,
                 out_channels,
                 hidden_act="relu",
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 dropout=None,
                 compute_type=mstype.float32):
        super(FeedForwardNet, self).__init__()

        self.fc1 = nn.Dense(in_channels,
                            hidden_size,
                            activation=hidden_act,
                            weight_init=TruncatedNormal(initializer_range)).to_float(compute_type)
        self.fc2 = nn.Dense(hidden_size,
                            out_channels,
                            weight_init=TruncatedNormal(initializer_range)).to_float(compute_type)

        self.layer_norm = LayerNorm(in_channels=in_channels,
                                    return_2d=True)
        self.residual = ResidualConnection(
            dropout_prob=hidden_dropout_prob if dropout is None else dropout
        )
        self.get_shape = P.Shape()
        self.reshape = P.Reshape()
        self.dropout = nn.Dropout(keep_prob=1 - hidden_dropout_prob)

    def construct(self, input_tensor):
        """
        Construct network.

        Args:
            input_tensor (Tensor): Shape (N, T, D).

        Returns:
            Tensor, (N, T, D).
        """
        shape = self.get_shape(input_tensor)
        batch_size = shape[0]
        max_len = shape[1]
        embed_dim = shape[2]

        output = self.layer_norm(input_tensor)
        output = self.fc1(output)
        output = self.dropout(output)
        output = self.fc2(output)  # (-1, D)
        output = self.residual(self.reshape(output, (batch_size, max_len, embed_dim)),
                               input_tensor)  # (N, T, D)
        return output

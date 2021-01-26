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
"""Residual block."""
import mindspore.nn as nn
from mindspore.ops import operations as P


class ResidualConnection(nn.Cell):
    """
    Add residual to output.

    Args:
        dropout_prob (float): Dropout rate.

    Returns:
        Tensor, with same shape of hidden_tensor.
    """

    def __init__(self, dropout_prob=0.1):
        super(ResidualConnection, self).__init__()
        self.add = P.Add()
        self.dropout = nn.Dropout(1.0 - dropout_prob)

    def construct(self, hidden_tensor, residual):
        """
        Construct network.

        Args:
            hidden_tensor (Tensor): Hidden tensor.
            residual (Tensor): Input tensor.

        Returns:
            Tensor, which has the same shape with hidden_tensor and residual.
        """
        output = self.dropout(hidden_tensor)
        output = self.add(output, residual)
        return output

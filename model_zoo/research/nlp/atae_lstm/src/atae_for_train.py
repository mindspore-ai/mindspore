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
"""AttentionLSTM for training"""
from mindspore import nn
from mindspore import ops as P
from mindspore.common import dtype as mstype


class NetWithLoss(nn.Cell):
    """
    calculate loss
    """
    def __init__(self, model, batch_size=1):
        super(NetWithLoss, self).__init__()

        self.batch_size = batch_size
        self.model = model

        self.cast = P.Cast()
        self.transpose = P.Transpose()
        self.trans_matrix = (1, 0)
        self.cross_entropy = nn.BCELoss(reduction='sum')
        self.reduce_sum = P.ReduceSum()

    def construct(self, content, sen_len, aspect, solution):
        """
        content: (batch_size, 50) int32
        sen_len: (batch_size,) Int32
        aspect: (batch_size,) int32
        solution: (batch_size, 3) Int32
        """

        pred = self.model(content, sen_len, aspect)
        label = self.cast(solution, mstype.float32)

        loss = self.cross_entropy(pred, label)

        return loss

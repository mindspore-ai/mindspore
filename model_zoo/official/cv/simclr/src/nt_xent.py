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
"""SimCLR Loss class."""

from mindspore import Tensor
from mindspore import ops as P
from mindspore.common import dtype as mstype
import mindspore.nn as nn


class CrossEntropyLoss(nn.Cell):
    """
        Cross Entropy Loss.
    """
    def __init__(self, reduction="mean"):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy = P.SoftmaxCrossEntropyWithLogits()
        if reduction == "sum":
            self.reduction = P.ReduceSum()
        if reduction == "mean":
            self.reduction = P.ReduceMean()
        self.one_hot = P.OneHot()
        self.one = Tensor(1.0, mstype.float32)
        self.zero = Tensor(0.0, mstype.float32)

    def construct(self, logits, label):
        loss = self.cross_entropy(logits, label)[0]
        loss = self.reduction(loss, (-1,))
        return loss


class NT_Xent_Loss(nn.Cell):
    """
        Loss for SimCLR.
    """
    def __init__(self, batch_size, temperature=1, world_size=1):
        super(NT_Xent_Loss, self).__init__()
        # Parameters.
        self.LARGE_NUM = 1e9
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        self.N = 2 * self.batch_size * self.world_size
        # Tail_Loss.
        self.criterion = CrossEntropyLoss(reduction="mean")
        self.norm = P.L2Normalize(axis=1)
        self.one_hot = P.OneHot()
        self.range = nn.Range(0, self.batch_size)
        self.one = Tensor(1.0, mstype.float32)
        self.zero = Tensor(0.0, mstype.float32)
        self.transpose = P.Transpose()
        self.matmul = nn.MatMul()
        # Operations.
        self.ones = P.Ones()
        self.zeros = P.Zeros()
        self.cat1 = P.Concat(axis=1)

    def construct(self, z_i, z_j):
        """
            Forward.
        """
        hidden1 = self.norm(z_i)
        hidden2 = self.norm(z_j)
        hidden1_large = hidden1
        hidden2_large = hidden2
        ones_mask = self.range()
        zeros_mask = self.zeros((self.batch_size, self.batch_size), mstype.float32)
        masks = self.one_hot(ones_mask, self.batch_size, self.one, self.zero)
        labels = self.cat1((masks, zeros_mask))
        logits_aa = self.matmul(hidden1, self.transpose(hidden1_large, (1, 0))) / self.temperature
        logits_aa = logits_aa - masks * self.LARGE_NUM
        logits_bb = self.matmul(hidden2, self.transpose(hidden2_large, (1, 0))) / self.temperature
        logits_bb = logits_bb - masks * self.LARGE_NUM
        logits_ab = self.matmul(hidden1, self.transpose(hidden2_large, (1, 0))) / self.temperature
        logits_ba = self.matmul(hidden2, self.transpose(hidden1_large, (1, 0))) / self.temperature
        loss_a = self.criterion(self.cat1((logits_ab, logits_aa)), labels)
        loss_b = self.criterion(self.cat1((logits_ba, logits_bb)), labels)
        loss = loss_a + loss_b
        return loss

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
"""OhemLoss."""
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F


class OhemLoss(nn.Cell):
    """Ohem loss cell."""
    def __init__(self, num, ignore_label):
        super(OhemLoss, self).__init__()
        self.mul = P.Mul()
        self.shape = P.Shape()
        self.one_hot = nn.OneHot(-1, num, 1.0, 0.0)
        self.squeeze = P.Squeeze()
        self.num = num
        self.cross_entropy = P.SoftmaxCrossEntropyWithLogits()
        self.mean = P.ReduceMean()
        self.select = P.Select()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.not_equal = P.NotEqual()
        self.equal = P.Equal()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.transpose = P.Transpose()
        self.ignore_label = ignore_label
        self.loss_weight = 1.0

    def construct(self, logits, labels):
        logits = self.transpose(logits, (0, 2, 3, 1))
        logits = self.reshape(logits, (-1, self.num))
        labels = F.cast(labels, mstype.int32)
        labels = self.reshape(labels, (-1,))
        one_hot_labels = self.one_hot(labels)
        losses = self.cross_entropy(logits, one_hot_labels)[0]
        weights = self.cast(self.not_equal(labels, self.ignore_label), mstype.float32) * self.loss_weight
        weighted_losses = self.mul(losses, weights)
        loss = self.reduce_sum(weighted_losses, (0,))
        zeros = F.fill(mstype.float32, self.shape(weights), 0.0)
        ones = F.fill(mstype.float32, self.shape(weights), 1.0)
        present = self.select(self.equal(weights, zeros), zeros, ones)
        present = self.reduce_sum(present, (0,))

        zeros = F.fill(mstype.float32, self.shape(present), 0.0)
        min_control = F.fill(mstype.float32, self.shape(present), 1.0)
        present = self.select(self.equal(present, zeros), min_control, present)
        loss = loss / present
        return loss

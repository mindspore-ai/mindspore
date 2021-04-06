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

import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as F
from mindspore.nn.loss.loss import _Loss


class CrossEntropyWithLogits(_Loss):
    def __init__(self):
        super(CrossEntropyWithLogits, self).__init__()
        self.transpose_fn = F.Transpose()
        self.reshape_fn = F.Reshape()
        self.softmax_cross_entropy_loss = nn.SoftmaxCrossEntropyWithLogits()
        self.cast = F.Cast()

    def construct(self, logits, label):
        # NCHW->NHWC
        logits = self.transpose_fn(logits, (0, 2, 3, 1))
        logits = self.cast(logits, mindspore.float32)
        label = self.transpose_fn(label, (0, 2, 3, 1))

        loss = self.reduce_mean(
            self.softmax_cross_entropy_loss(self.reshape_fn(logits, (-1, 2)), self.reshape_fn(label, (-1, 2))))
        return self.get_loss(loss)

class MultiCrossEntropyWithLogits(nn.Cell):
    def __init__(self):
        super(MultiCrossEntropyWithLogits, self).__init__()
        self.loss = CrossEntropyWithLogits()
        self.squeeze = F.Squeeze(axis=0)

    def construct(self, logits, label):
        total_loss = 0
        for i in range(len(logits)):
            total_loss += self.loss(self.squeeze(logits[i:i+1]), label)
        return total_loss

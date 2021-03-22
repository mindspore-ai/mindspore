# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from mindspore.nn.loss.loss import _Loss
from src.config import config

class SoftmaxCrossEntropyWithLogits(_Loss):
    def __init__(self):
        super(SoftmaxCrossEntropyWithLogits, self).__init__()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
        self.cast = P.Cast()
        self.reduce_mean = P.ReduceMean()

    def construct(self, logits, label):
        logits = self.transpose(logits, (0, 2, 3, 4, 1))
        label = self.transpose(label, (0, 2, 3, 4, 1))
        label = self.cast(label, mstype.float32)
        loss = self.reduce_mean(self.loss_fn(self.reshape(logits, (-1, config['num_classes'])), \
                                self.reshape(label, (-1, config['num_classes']))))
        return self.get_loss(loss)

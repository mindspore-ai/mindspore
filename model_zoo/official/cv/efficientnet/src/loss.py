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
"""define loss function for network."""
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from mindspore import Tensor
import mindspore.nn as nn

class LabelSmoothingCrossEntropy(LossBase):

    def __init__(self, smooth_factor=0.1, num_classes=1000):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.mean = P.ReduceMean(False)

    def construct(self, logits, label):
        one_hot_label = self.onehot(label, F.shape(logits)[1], self.on_value, self.off_value)
        loss_logit = self.ce(logits, one_hot_label)
        loss_logit = self.mean(loss_logit, 0)
        return loss_logit

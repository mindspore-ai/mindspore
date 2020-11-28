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
"""Face attribute cross entropy."""
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import Tensor
from mindspore.common import dtype as mstype


class CrossEntropyWithIgnoreIndex(nn.Cell):
    '''Cross Entropy With Ignore Index Loss.'''
    def __init__(self):
        super(CrossEntropyWithIgnoreIndex, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, dtype=mstype.float32)
        self.off_value = Tensor(0.0, dtype=mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.greater = P.Greater()
        self.maximum = P.Maximum()
        self.fill = P.Fill()
        self.sum = P.ReduceSum(keep_dims=False)
        self.dtype = P.DType()
        self.relu = P.ReLU()
        self.reshape = P.Reshape()
        self.const_one = Tensor(np.ones([1]), dtype=mstype.float32)
        self.const_eps = Tensor(0.00001, dtype=mstype.float32)

    def construct(self, x, label):
        '''Construct function.'''
        mask = self.reshape(label, (F.shape(label)[0], 1))
        mask = self.cast(mask, mstype.float32)
        mask = mask + self.const_eps
        mask = self.relu(mask)/mask
        x = x * mask
        one_hot_label = self.onehot(self.cast(label, mstype.int32), F.shape(x)[1], self.on_value, self.off_value)
        loss = self.ce(x, one_hot_label)
        positive = self.sum(self.cast(self.greater(loss, self.fill(self.dtype(loss), F.shape(loss), 0.0)),
                                      mstype.float32), 0)
        positive = self.maximum(positive, self.const_one)
        loss = self.sum(loss, 0) / positive
        return loss

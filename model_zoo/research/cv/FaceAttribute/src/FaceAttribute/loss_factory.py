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
"""Face attribute loss."""
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore.common import dtype as mstype

from src.FaceAttribute.cross_entropy import CrossEntropyWithIgnoreIndex


__all__ = ['get_loss']


class CriterionsFaceAttri(nn.Cell):
    '''Criterions Face Attribute.'''
    def __init__(self):
        super(CriterionsFaceAttri, self).__init__()

        # label
        self.gatherv2 = P.Gather()
        self.squeeze = P.Squeeze(axis=1)
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.mean = P.ReduceMean()

        self.label0_param = Tensor([0], dtype=mstype.int32)
        self.label1_param = Tensor([1], dtype=mstype.int32)
        self.label2_param = Tensor([2], dtype=mstype.int32)

        # loss
        self.ce_ignore_loss = CrossEntropyWithIgnoreIndex()

    def construct(self, x0, x1, x2, label):
        '''Construct function.'''
        # each sub attribute loss
        label0 = self.squeeze(self.gatherv2(label, self.label0_param, 1))
        loss0 = self.ce_ignore_loss(x0, label0)

        label1 = self.squeeze(self.gatherv2(label, self.label1_param, 1))
        loss1 = self.ce_ignore_loss(x1, label1)

        label2 = self.squeeze(self.gatherv2(label, self.label2_param, 1))
        loss2 = self.ce_ignore_loss(x2, label2)

        loss = loss0 + loss1 + loss2

        return loss


def get_loss():
    return CriterionsFaceAttri()

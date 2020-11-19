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
"""
define loss
"""
from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P



class BCELoss(nn.Cell):
    """
    BCELoss
    """
    def __init__(self, record=None):
        super(BCELoss, self).__init__(record)
        self.sm_scalar = P.ScalarSummary()
        self.cast = P.Cast()
        self.record = record
        self.weight = None
        self.bce = P.BinaryCrossEntropy()

    def construct(self, input_data, target):
        target = self.cast(target, mstype.float32)
        loss = self.bce(input_data, target, self.weight)
        if self.record:
            self.sm_scalar("loss", loss)
        return loss

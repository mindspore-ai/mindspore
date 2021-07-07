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
"""AttentionLSTM for inferring"""
from mindspore import ops as P
from mindspore import nn
from mindspore.common import dtype as mstype


class Infer(nn.Cell):
    """
    Infer module
    """
    def __init__(self, model, batch_size=1):
        super(Infer, self).__init__()
        self.model = model
        self.batch_size = batch_size
        self.cast = P.Cast()
        self.transpose = P.Transpose()
        self.transpose_orders = (1, 0, 2)

    def construct(self, content, sen_len, aspect):
        """
        batch first
        """
        content = self.cast(content, mstype.int32)
        aspect = self.cast(aspect, mstype.int32)

        pred = self.model(content, sen_len, aspect)

        pred = self.cast(pred, mstype.float32)
        return pred

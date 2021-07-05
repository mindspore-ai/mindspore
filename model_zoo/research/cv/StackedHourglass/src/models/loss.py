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
"""
define heatmap loss
"""
import mindspore.nn as nn
import mindspore.ops.operations as P


class HeatmapLoss(nn.Cell):
    """
    loss for detection heatmap
    """

    def __init__(self):
        super(HeatmapLoss, self).__init__()
        self.loss_function = nn.MSELoss()
        self.transpose = P.Transpose()

    def construct(self, pred, gt):
        """
        calculate loss
        """
        # pred size (batch, 8, 16, 64, 64), gt size (batch, 16, 16, 64)
        # Use broadcast to calculate loss
        pred_t = self.transpose(pred, (1, 0, 2, 3, 4))
        loss = self.loss_function(pred_t, gt)

        return loss

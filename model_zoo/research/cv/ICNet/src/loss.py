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
"""Custom losses."""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from src.losses import SoftmaxCrossEntropyLoss

__all__ = ['ICNetLoss']


class ICNetLoss(nn.Cell):
    """Cross Entropy Loss for ICNet"""

    def __init__(self, aux_weight=0.4, ignore_index=-1):
        super(ICNetLoss, self).__init__()
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index
        self.sparse = True
        self.base_loss = SoftmaxCrossEntropyLoss(num_cls=19, ignore_label=-1)
        self.resize_bilinear = nn.ResizeBilinear()  # 输入必须为4D

    def construct(self, *inputs):
        """construct"""
        preds, target = inputs

        pred = preds[0]
        pred_sub4 = preds[1]
        pred_sub8 = preds[2]
        pred_sub16 = preds[3]

        # [batch, H, W] -> [batch, 1, H, W]
        expand_dims = ops.ExpandDims()
        if target.shape[0] == 720 or target.shape[0] == 1024:
            target = expand_dims(target, 0).astype(ms.dtype.float32)
            target = expand_dims(target, 0).astype(ms.dtype.float32)
        else:
            target = expand_dims(target, 1).astype(ms.dtype.float32)

        h, w = pred.shape[2:]

        target_sub4 = self.resize_bilinear(target, size=(h / 4, w / 4)).squeeze(1)

        target_sub8 = self.resize_bilinear(target, size=(h / 8, w / 8)).squeeze(1)

        target_sub16 = self.resize_bilinear(target, size=(h / 16, w / 16)).squeeze(1)

        loss1 = self.base_loss(pred_sub4, target_sub4)
        loss2 = self.base_loss(pred_sub8, target_sub8)
        loss3 = self.base_loss(pred_sub16, target_sub16)

        return loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight

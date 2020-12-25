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
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.loss.loss import _Loss
from mindspore.common import dtype as mstype


class JointsMSELoss(_Loss):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.reshape = P.Reshape()
        self.squeeze = P.Squeeze(1)
        self.mul = P.Mul()

    def construct(self, output, target, target_weight):
        batch_size = F.shape(output)[0]
        num_joints = F.shape(output)[1]

        split = P.Split(1, num_joints)
        heatmaps_pred = self.reshape(output, (batch_size, num_joints, -1))
        heatmaps_pred = split(heatmaps_pred)

        heatmaps_gt = self.reshape(target, (batch_size, num_joints, -1))
        heatmaps_gt = split(heatmaps_gt)
        loss = 0
        for idx in range(num_joints):
            heatmap_pred = self.squeeze(heatmaps_pred[idx])
            heatmap_gt = self.squeeze(heatmaps_gt[idx])
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    self.mul(heatmap_pred, target_weight[:, idx]),
                    self.mul(heatmap_gt, target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        return loss / num_joints


class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to compute loss.

    Args:
        backbone (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
    """

    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, image, target, weight, scale=None,
                  center=None, score=None, idx=None):
        out = self._backbone(image)
        output = F.mixed_precision_cast(mstype.float32, out)
        target = F.mixed_precision_cast(mstype.float32, target)
        weight = F.mixed_precision_cast(mstype.float32, weight)
        return self._loss_fn(output, target, weight)

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, return backbone network.
        """
        return self._backbone

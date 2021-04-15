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
'''
network_with_loss
'''
from __future__ import division

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn.loss.loss import _Loss
from mindspore.common import dtype as mstype

class JointsMSELoss(_Loss):
    '''
    JointsMSELoss
    '''
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.squeeze = P.Squeeze(1)
        self.mul = P.Mul()

    def construct(self, output, target, target_weight):
        '''
        construct
        '''
        total_shape = self.shape(output)
        batch_size = total_shape[0]
        num_joints = total_shape[1]
        remained_size = 1
        for i in range(2, len(total_shape)):
            remained_size *= total_shape[i]

        split = P.Split(1, num_joints)
        new_shape = (batch_size, num_joints, remained_size)
        heatmaps_pred = split(self.reshape(output, new_shape))
        heatmaps_gt = split(self.reshape(target, new_shape))
        loss = 0

        for idx in range(num_joints):
            heatmap_pred_squeezed = self.squeeze(heatmaps_pred[idx])
            heatmap_gt_squeezed = self.squeeze(heatmaps_gt[idx])
            if self.use_target_weight:
                loss += 0.5 * self.criterion(self.mul(heatmap_pred_squeezed, target_weight[:, idx]),
                                             self.mul(heatmap_gt_squeezed, target_weight[:, idx]))
            else:
                loss += 0.5 * self.criterion(heatmap_pred_squeezed, heatmap_gt_squeezed)

        return loss / num_joints

class PoseResNetWithLoss(nn.Cell):
    """
    Pack the model network and loss function together to calculate the loss value.
    """
    def __init__(self, network, loss):
        super(PoseResNetWithLoss, self).__init__()
        self.network = network
        self.loss = loss

    def construct(self, image, target, weight, scale=None, center=None, score=None, idx=None):
        output = self.network(image)
        output = F.mixed_precision_cast(mstype.float32, output)
        target = F.mixed_precision_cast(mstype.float32, target)
        weight = F.mixed_precision_cast(mstype.float32, weight)
        return self.loss(output, target, weight)

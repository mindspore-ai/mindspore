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
"""network definition"""

import mindspore.nn as nn
import mindspore.ops as P
from mindspore.nn.loss.loss import _Loss
from mindspore.ops import functional as F

from src.layers import Dense
from src.pointnet2_utils import PointNetSetAbstraction


class PointNet2(nn.Cell):
    """PointNet2"""

    def __init__(self, num_class, normal_channel=False):
        super(PointNet2, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32,
                                          in_channel=in_channel, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64,
                                          in_channel=128 + 3, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None,
                                          in_channel=256 + 3, mlp=[256, 512, 1024],
                                          group_all=True)

        self.fc1 = Dense(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.6)
        self.fc2 = Dense(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = Dense(256, num_class)

        self.relu = P.ReLU()
        self.reshape = P.Reshape()
        self.log_softmax = P.LogSoftmax()
        self.transpose = P.Transpose()

    def construct(self, xyz):
        """
        construct method
        """
        if self.normal_channel:
            norm = self.transpose(xyz[:, :, 3:], (0, 2, 1))
            xyz = xyz[:, :, :3]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)  # [B, 3, 512], [B, 128, 512]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B, 3, 128], [B, 256, 128]
        _, l3_points = self.sa3(l2_xyz, l2_points)  # [B, 3, 1], [B, 1024, 1]
        x = self.reshape(l3_points, (-1, 1024))
        x = self.drop1(self.relu(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x


class NLLLoss(_Loss):
    """NLL loss"""

    def __init__(self, reduction='mean'):
        super(NLLLoss, self).__init__(reduction)
        self.one_hot = P.OneHot()
        self.reduce_sum = P.ReduceSum()

    def construct(self, logits, label):
        """
        construct method
        """
        label_one_hot = self.one_hot(label, F.shape(logits)[-1], F.scalar_to_array(1.0), F.scalar_to_array(0.0))
        loss = self.reduce_sum(-1.0 * logits * label_one_hot, (1,))
        return loss

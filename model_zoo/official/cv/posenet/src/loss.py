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
"""define evaluation loss function for network."""
import mindspore.nn as nn
from mindspore.nn.loss.loss import LossBase
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import common_config as config
from src.posenet import PoseNet

class EuclideanDistance(nn.Cell):
    """calculate euclidean distance"""
    def __init__(self):
        super(EuclideanDistance, self).__init__()
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.reduce_sum = P.ReduceSum()
        self.sqrt = P.Sqrt()

    def construct(self, predicted, real):
        res = self.sub(predicted, real)
        res = self.mul(res, res)
        res = self.reduce_sum(res, 0)
        res = self.sqrt(res)
        res = self.mul(res, res)
        res = self.reduce_sum(res, 0)
        res = self.sqrt(res)

        return res

class PoseLoss(LossBase):
    """define loss function"""
    def __init__(self, w1_x, w2_x, w3_x, w1_q, w2_q, w3_q):
        super(PoseLoss, self).__init__()
        self.w1_x = w1_x
        self.w2_x = w2_x
        self.w3_x = w3_x
        self.w1_q = w1_q
        self.w2_q = w2_q
        self.w3_q = w3_q
        self.ed = EuclideanDistance()

    def construct(self, p1_x, p1_q, p2_x, p2_q, p3_x, p3_q, poseGT):
        """construct"""
        pose_x = poseGT[:, 0:3]
        pose_q = poseGT[:, 3:]

        l1_x = self.ed(pose_x, p1_x) * self.w1_x
        l1_q = self.ed(pose_q, p1_q) * self.w1_q
        l2_x = self.ed(pose_x, p2_x) * self.w2_x
        l2_q = self.ed(pose_q, p2_q) * self.w2_q
        l3_x = self.ed(pose_x, p3_x) * self.w3_x
        l3_q = self.ed(pose_q, p3_q) * self.w3_q
        loss = l1_x + l1_q + l2_x + l2_q + l3_x + l3_q

        return loss

class PosenetWithLoss(nn.Cell):
    """net with loss, and do pre_trained"""
    def __init__(self, pre_trained=False):
        super(PosenetWithLoss, self).__init__()

        net = PoseNet()
        if pre_trained:
            param_dict = load_checkpoint(config.pre_trained_file)
            load_param_into_net(net, param_dict)

        self.network = net
        self.loss = PoseLoss(3.0, 3.0, 10.0, 150, 150, 500)
        self.cast = P.Cast()

    def construct(self, data, poseGT):
        p1_x, p1_q, p2_x, p2_q, p3_x, p3_q = self.network(data)
        loss = self.loss(p1_x, p1_q, p2_x, p2_q, p3_x, p3_q, poseGT)
        return self.cast(loss, mstype.float32)

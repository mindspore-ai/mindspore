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
python loss.py
"""
from mindspore import Tensor
import mindspore.nn as nn
from mindspore import Parameter
import mindspore.ops as ops
from mindspore import dtype as mstype
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer


class ArcFace(nn.Cell):
    '''
    Arcface loss
    '''
    def __init__(self, world_size, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.shape = ops.Shape()
        self.mul = ops.Mul()
        self.cos = ops.Cos()
        self.acos = ops.ACos()
        self.onehot = ops.OneHot().shard(((1, world_size), (), ()))
        # self.tile = ops.Tile().shard(((8, 1),))
        self.on_value = Tensor(m, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)

    def construct(self, cosine, label):
        m_hot = self.onehot(label, self.shape(
            cosine)[1], self.on_value, self.off_value)

        cosine = self.acos(cosine)
        cosine += m_hot
        cosine = self.cos(cosine)
        cosine = self.mul(cosine, self.s)
        return cosine


class SoftMaxCE(nn.Cell):
    '''
    softmax cross entrophy
    '''
    def __init__(self, world_size):
        super(SoftMaxCE, self).__init__()
        self.max = ops.ReduceMax(keep_dims=True)
        self.sum = ops.ReduceSum(keep_dims=True)
        self.mean = ops.ReduceMean(keep_dims=False)
        self.exp = ops.Exp()
        self.div = ops.Div()
        self.onehot = ops.OneHot().shard(((1, world_size), (), ()))
        self.mul = ops.Mul()
        self.log = ops.Log()
        self.onvalue = Tensor(1.0, mstype.float32)
        self.offvalue = Tensor(0.0, mstype.float32)
        self.eps = Tensor(1e-30, mstype.float32)

    def construct(self, logits, total_label):
        '''construct
        '''
        max_fc = self.max(logits, 1)

        logits_exp = self.exp(logits - max_fc)
        logits_sum_exp = self.sum(logits_exp, 1)

        logits_exp = self.div(logits_exp, logits_sum_exp)

        label = self.onehot(total_label, F.shape(
            logits)[1], self.onvalue, self.offvalue)

        softmax_result_log = self.log(logits_exp + self.eps)
        loss = self.sum((self.mul(softmax_result_log, label)), -1)
        loss = self.mul(ops.scalar_to_array(-1.0), loss)
        loss_v = self.mean(loss, 0)

        return loss_v


class PartialFC(nn.Cell):
    '''partialFC
    '''
    def __init__(self, num_classes, world_size):
        super(PartialFC, self).__init__()
        self.L2Norm = ops.L2Normalize(axis=1)
        self.weight = Parameter(initializer(
            "normal", (num_classes, 512)), name="mp_weight")
        self.sub_weight = self.weight
        self.linear = ops.MatMul(transpose_b=True).shard(
            ((1, 1), (world_size, 1)))
        self.margin_softmax = ArcFace(world_size=world_size)
        self.loss = SoftMaxCE(world_size=world_size)

    def construct(self, features, label):
        total_label, norm_weight = self.prepare(label)
        total_features = self.L2Norm(features)
        logits = self.forward(total_features, norm_weight)
        logits = self.margin_softmax(logits, total_label)
        loss_v = self.loss(logits, total_label)
        return loss_v

    def forward(self, total_features, norm_weight):
        logits = self.linear(F.cast(total_features, mstype.float16), F.cast(
            norm_weight, mstype.float16))
        return F.cast(logits, mstype.float32)

    def prepare(self, label):
        total_label = label
        norm_weight = self.L2Norm(self.sub_weight)
        return total_label, norm_weight

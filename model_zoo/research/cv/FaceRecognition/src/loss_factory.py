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
"""Face Recognition loss."""
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.nn.loss.loss import _Loss
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype

__all__ = ['get_loss']

eps = 1e-24

class CrossEntropy(_Loss):
    '''CrossEntropy'''
    def __init__(self, args):
        super(CrossEntropy, self).__init__()
        self.args_1 = args

        self.exp = P.Exp()
        self.sum = P.ReduceSum()
        self.reshape = P.Reshape()
        self.log = P.Log()
        self.cast = P.Cast()
        self.eps_const = Tensor(eps, dtype=mstype.float32)
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)

    def construct(self, logit, label):
        '''construct'''
        exp = self.exp(logit)
        exp_sum = self.sum(exp, -1)
        exp_sum = self.reshape(exp_sum, (F.shape(exp_sum)[0], 1))
        softmax_result = exp / exp_sum
        one_hot_label = self.onehot(
            self.cast(label, mstype.int32), F.shape(logit)[1], self.on_value, self.off_value)
        loss = self.sum((self.log(softmax_result + self.eps_const) *
                         self.cast(one_hot_label, mstype.float32) *
                         self.cast(F.scalar_to_array(-1), mstype.float32)), -1)
        batch_size = F.shape(logit)[0]
        batch_size_tensor = self.cast(
            F.scalar_to_array(batch_size), mstype.float32)
        loss = self.sum(loss, -1) / batch_size_tensor
        return loss


class Criterions(Cell):
    '''Criterions'''
    def __init__(self, args):
        super(Criterions, self).__init__()

        self.criterion_ce = CrossEntropy(args)
        self.total_loss = Tensor(0.0, dtype=mstype.float32)

    def construct(self, margin_logit, label):
        total_loss = self.total_loss
        loss_ce = self.criterion_ce(margin_logit, label)
        total_loss = total_loss + loss_ce

        return total_loss


def get_loss(args):
    return Criterions(args)

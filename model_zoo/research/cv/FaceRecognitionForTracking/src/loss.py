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
from mindspore.nn.loss.loss import _Loss
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import Tensor
from mindspore.common import dtype as mstype
import mindspore.nn as nn

eps = 1e-24


class CrossEntropyNew(_Loss):
    '''CrossEntropyNew'''
    def __init__(self, smooth_factor=0., num_classes=1000):
        super(CrossEntropyNew, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.mean = P.ReduceMean(False)

    def construct(self, logit, label):
        one_hot_label = self.onehot(self.cast(label, mstype.int32),
                                    F.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, one_hot_label)
        loss = self.mean(loss, 0)
        return loss


class CrossEntropy(_Loss):
    '''CrossEntropy'''
    def __init__(self):
        super(CrossEntropy, self).__init__()

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
        '''Construct function.'''
        exp = self.exp(logit)
        exp_sum = self.sum(exp, -1)
        exp_sum = self.reshape(exp_sum, (F.shape(exp_sum)[0], 1))
        softmax_result = exp / exp_sum
        one_hot_label = self.onehot(
            self.cast(label, mstype.int32), F.shape(logit)[1], self.on_value, self.off_value)
        loss = self.sum((self.log(softmax_result + self.eps_const) * self.cast(
            one_hot_label, mstype.float32) * self.cast(F.scalar_to_array(-1), mstype.float32)), -1)
        batch_size = F.shape(logit)[0]
        batch_size_tensor = self.cast(
            F.scalar_to_array(batch_size), mstype.float32)
        loss = self.sum(loss, -1) / batch_size_tensor
        return loss


class CrossEntropyWithIgnoreIndex(nn.Cell):
    '''CrossEntropyWithIgnoreIndex'''
    def __init__(self):
        super(CrossEntropyWithIgnoreIndex, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.greater = P.Greater()
        self.maximum = P.Maximum()
        self.fill = P.Fill()
        self.sum = P.ReduceSum(keep_dims=False)
        self.dtype = P.DType()
        self.relu = P.ReLU()
        self.reshape = P.Reshape()

    def construct(self, x, label):
        mask = self.reshape(label, (F.shape(label)[0], 1))
        mask = self.cast(mask, mstype.float32)
        mask = mask + F.scalar_to_array(0.00001)
        mask = self.relu(mask) / (mask)
        x = x * mask
        one_hot_label = self.onehot(self.cast(label, mstype.int32), F.shape(x)[1], self.on_value, self.off_value)
        loss = self.ce(x, one_hot_label)
        loss = self.sum(loss, 0)
        return loss


eps = 1e-24


class CEWithIgnoreIndex3D(_Loss):
    '''CEWithIgnoreIndex3D'''
    def __init__(self):
        super(CEWithIgnoreIndex3D, self).__init__()

        self.exp = P.Exp()
        self.sum = P.ReduceSum()
        self.reshape = P.Reshape()
        self.log = P.Log()
        self.cast = P.Cast()
        self.eps_const = Tensor(eps, dtype=mstype.float32)
        self.ones = P.OnesLike()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.relu = P.ReLU()
        self.resum = P.ReduceSum(keep_dims=False)

    def construct(self, logit, label):
        '''Construct function.'''
        mask = self.reshape(label, (F.shape(label)[0], F.shape(label)[1], 1))
        mask = self.cast(mask, mstype.float32)
        mask = mask + F.scalar_to_array(0.00001)
        mask = self.relu(mask) / (mask)
        logit = logit * mask

        exp = self.exp(logit)
        exp_sum = self.sum(exp, -1)
        exp_sum = self.reshape(exp_sum, (F.shape(exp_sum)[0], F.shape(exp_sum)[1], 1))
        softmax_result = self.log(exp / exp_sum + self.eps_const)
        one_hot_label = self.onehot(
            self.cast(label, mstype.int32), F.shape(logit)[2], self.on_value, self.off_value)
        loss = (softmax_result * self.cast(one_hot_label, mstype.float32) * self.cast(F.scalar_to_array(-1),
                                                                                      mstype.float32))

        loss = self.sum(loss, -1)
        loss = self.sum(loss, -1)
        loss = self.sum(loss, 0)
        loss = loss

        return loss

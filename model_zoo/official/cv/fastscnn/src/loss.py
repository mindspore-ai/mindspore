# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Custom losses."""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.nn import SoftmaxCrossEntropyWithLogits

__all__ = ['MixSoftmaxCrossEntropyLoss']
class MixSoftmaxCrossEntropyLoss(nn.Cell):
    '''MixSoftmaxCrossEntropyLoss'''
    def __init__(self, args, ignore_label=-1, aux=True, aux_weight=0.4, \
                 sparse=True, reduction='none', one_d_length=2*768*768, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__()
        self.ignore_label = ignore_label
        self.weight = aux_weight if aux else 1.0
        self.select = ops.Select()
        self.reduceSum = ops.ReduceSum(keep_dims=False)
        self.div_no_nan = ops.DivNoNan()
        self.mul = ops.Mul()
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()
        self.transpose = ops.Transpose()
        self.zero_tensor = Tensor([0]*one_d_length, mindspore.float32)
        self.SoftmaxCrossEntropyWithLogits = \
                        SoftmaxCrossEntropyWithLogits(sparse=sparse, reduction="none")
        args.logger.info('using MixSoftmaxCrossEntropyLoss....')
        args.logger.info('self.ignore_label:' + str(self.ignore_label))
        args.logger.info('self.aux:' + str(aux))
        args.logger.info('self.weight:' + str(self.weight))
        args.logger.info('one_d_length:' + str(one_d_length))
    def construct(self, *inputs, **kwargs):
        '''construct'''
        preds, target = inputs[:-1], inputs[-1]
        target = self.reshape(target, (-1,))
        valid_flag = target != self.ignore_label
        num_valid = self.reduceSum(self.cast(valid_flag, mindspore.float32))

        z = self.transpose(preds[0], (0, 2, 3, 1))#move the C-dim to the last, then reshape it.
                                               #This operation is vital, or the data would be soiled.
        loss = self.SoftmaxCrossEntropyWithLogits(self.reshape(z, (-1, 19)), target)
        loss = self.select(valid_flag, loss, self.zero_tensor)
        loss = self.reduceSum(loss)
        loss = self.div_no_nan(loss, num_valid)
        for i in range(1, len(preds)):
            z = self.transpose(preds[i], (0, 2, 3, 1))
            aux_loss = self.SoftmaxCrossEntropyWithLogits(self.reshape(z, (-1, 19)), target)
            aux_loss = self.select(valid_flag, aux_loss, self.zero_tensor)
            aux_loss = self.reduceSum(aux_loss)
            aux_loss = self.div_no_nan(aux_loss, num_valid)
            loss += self.mul(self.weight, aux_loss)
        return loss

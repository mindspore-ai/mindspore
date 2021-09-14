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
""" define loss function"""
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np


class MultiBoxLoss(nn.Cell):
    """ loss class """
    def __init__(self, batch_size=1):
        super(MultiBoxLoss, self).__init__()
        self.batch_size = batch_size
        self.cast = ops.Cast()
        self.realsum_false = ops.ReduceSum(keep_dims=False)
        self.realdiv = ops.RealDiv()
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.onehot = ops.OneHot()
        self.SmooL1loss = nn.SmoothL1Loss()
        self.realsum_true = ops.ReduceSum(keep_dims=True)
        self.div = ops.Div()

        self.equal = ops.Equal()
        self.select = ops.Select()
        self.tile = ops.Tile()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.split = ops.Split(axis=0, output_num=self.batch_size)

        self.ones = Tensor(np.ones((1, 1445)), mindspore.float32)
        self.zeros_class_pred = Tensor(np.zeros((1, 1445, 2)), mindspore.float32)
        self.nage_class_pred = Tensor(np.ones((1, 1445, 2)), mindspore.float32)
        self.zeros_class_target = Tensor(np.zeros((1, 1445)), mindspore.float32)

        self.depth, self.on_value, self.off_value = 2, Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32)

        self.c_all_loss = Tensor(0, mindspore.float32)
        self.r_all_loss = Tensor(0, mindspore.float32)

    def construct(self, predictions1, predictions2, targets):
        """ class """
        cout = self.transpose(self.reshape(predictions1, (-1, 2, 5 * 17 * 17)), (0, 2, 1))
        rout = self.transpose(self.reshape(predictions2, (-1, 4, 5 * 17 * 17)), (0, 2, 1))
        cout = self.split(cout)
        rout = self.split(rout)
        targets = self.cast(targets, mindspore.float32)
        ctargets = targets[:, :, 0:1]
        rtargets = targets[:, :, 1:]
        ctargets = self.split(ctargets)
        rtargets = self.split(rtargets)
        c_all_loss = self.c_all_loss
        r_all_loss = self.r_all_loss
        for batch in range(self.batch_size):
            class_pred, class_target = cout[batch], ctargets[batch]
            class_pred = self.reshape(class_pred, (1, -1, 2))
            class_target = self.reshape(class_target, (1, -1))
            class_target = self.cast(class_target, mindspore.float32)

            pos_mask = self.equal(class_target, self.ones)
            neg_mask = self.equal(class_target, self.zeros_class_target)
            class_target_pos = self.select(pos_mask, class_target, self.zeros_class_target - self.ones)
            class_target_pos = self.cast(class_target_pos, mindspore.int32)
            class_target_pos = self.reshape(class_target_pos, (-1,))
            class_target_neg = self.select(neg_mask, class_target, self.zeros_class_target - self.ones)
            class_target_neg = self.cast(class_target_neg, mindspore.int32)
            class_target_neg = self.reshape(class_target_neg, (-1,))
            pos_mask1 = self.cast(pos_mask, mindspore.int32)#
            pos_mask2 = self.cast(pos_mask, mindspore.float32)
            pos_num = self.realsum_false(pos_mask2)
            pos_mask1 = self.reshape(pos_mask1, (1, -1, 1))
            pos_mask1 = self.tile(pos_mask1, (1, 1, 2))
            neg_mask1 = self.cast(neg_mask, mindspore.int32)
            neg_mask2 = self.cast(neg_mask, mindspore.float32)
            neg_num = self.realsum_false(neg_mask2)
            neg_mask1 = self.reshape(neg_mask1, (1, -1, 1))
            neg_mask1 = self.tile(neg_mask1, (1, 1, 2))
            pos_mask1 = self.cast(pos_mask1, mindspore.bool_)
            neg_mask1 = self.cast(neg_mask1, mindspore.bool_)
            class_pred_pos = self.select(pos_mask1, class_pred, self.zeros_class_pred)
            class_pred_neg = self.select(neg_mask1, class_pred, self.zeros_class_pred)
            class_pos = self.reshape(class_pred_pos, (-1, 2))
            class_neg = self.reshape(class_pred_neg, (-1, 2))
            closs_pos = self.cross_entropy(class_pos, class_target_pos)
            closs_neg = self.cross_entropy(class_neg, class_target_neg)
            c_all_loss += (self.realdiv(self.realsum_false(closs_pos), (pos_num + 1e-6)) + \
                          self.realdiv(self.realsum_false(closs_neg), (neg_num + 1e-6)))/2
            reg_pred = rout[batch]
            reg_pred = self.reshape(reg_pred, (-1, 4))
            reg_target = rtargets[batch]
            reg_target = self.reshape(reg_target, (-1, 4))
            rloss = self.SmooL1loss(reg_pred, reg_target)  # 1445, 4
            rloss = self.realsum_false(rloss, -1)
            rloss = rloss / 4
            rloss = self.reshape(rloss, (1, -1))
            rloss = self.select(pos_mask, rloss, self.zeros_class_target)
            rloss = self.realsum_false(rloss) / (pos_num + 1e-6)
            r_all_loss += rloss
        loss = (c_all_loss + 5 * r_all_loss) / self.batch_size
        return loss

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
loss
"""
import math
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from src.util import Sigmoid, TransposeGatherFeature


class FocalLoss(nn.Cell):
    """
    Warpper for focal loss.

    Args:
        alpha(int): Super parameter in focal loss to mimic loss weight. Default: 2.
        beta(int): Super parameter in focal loss to mimic imbalance between positive and negative samples. Default: 4.

    Returns:
        Tensor, focal loss.
    """

    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.pow = ops.Pow()
        self.log = ops.Log()
        self.select = ops.Select()
        self.equal = ops.Equal()
        self.less = ops.Less()
        self.cast = ops.Cast()
        self.fill = ops.Fill()
        self.dtype = ops.DType()
        self.shape = ops.Shape()
        self.reduce_sum = ops.ReduceSum()

    def construct(self, out, target):
        """focal loss"""
        pos_inds = self.cast(self.equal(target, 1.0), mstype.float32)
        neg_inds = self.cast(self.less(target, 1.0), mstype.float32)
        neg_weights = self.pow(1 - target, self.beta)

        pos_loss = self.log(out) * self.pow(1 - out, self.alpha) * pos_inds
        neg_loss = self.log(1 - out) * self.pow(out, self.alpha) * neg_weights * neg_inds

        num_pos = self.reduce_sum(pos_inds, ())
        num_pos = self.select(self.equal(num_pos, 0.0),
                              self.fill(self.dtype(num_pos), self.shape(num_pos), 1.0), num_pos)
        pos_loss = self.reduce_sum(pos_loss, ())
        neg_loss = self.reduce_sum(neg_loss, ())
        loss = - (pos_loss + neg_loss) / num_pos
        return loss


class RegLoss(nn.Cell):
    """
    Warpper for regression loss.

    Args:
        mode(str): L1 or Smoothed L1 loss. Default: "l1"

    Returns:
        Tensor, regression loss.
    """

    def __init__(self, mode='l1'):
        super(RegLoss, self).__init__()
        self.reduce_sum = ops.ReduceSum()
        self.cast = ops.Cast()
        self.expand_dims = ops.ExpandDims()
        self.reshape = ops.Reshape()
        self.gather_feature = TransposeGatherFeature()
        if mode == 'l1':
            self.loss = nn.L1Loss(reduction='sum')
        elif mode == 'sl1':
            self.loss = nn.SmoothL1Loss()
        else:
            self.loss = None

    def construct(self, output, mask, ind, target):
        """Warpper for regression loss."""
        pred = self.gather_feature(output, ind)
        mask = self.cast(mask, mstype.float32)
        num = self.reduce_sum(mask, ())
        mask = self.expand_dims(mask, 2)
        target = target * mask
        pred = pred * mask
        regr_loss = self.loss(pred, target)
        regr_loss = regr_loss / (num + 1e-4)
        return regr_loss


class CenterNetMultiPoseLossCell(nn.Cell):
    """
    Provide pose estimation network losses.

    Args:
        net_config: The config info of CenterNet network.

    Returns:
        Tensor, total loss.
    """

    def __init__(self, opt):
        super(CenterNetMultiPoseLossCell, self).__init__()
        self.crit = FocalLoss()
        # self.crit_wh = RegWeightedL1Loss() if not config.net.dense_hp else nn.L1Loss(reduction='sum')
        self.crit_wh = RegLoss(opt.reg_loss)
        # wh
        self.crit_reg = RegLoss(opt.reg_loss)  # reg_loss = 'l1'
        self.hm_weight = opt.hm_weight  # hm_weight = 1 :loss weight for keypoint heatmaps
        self.wh_weight = opt.wh_weight  # wh_weight = 0.1 : loss weight for bounding box size
        self.off_weight = opt.off_weight  # off_weight = 1 : loss weight for keypoint local offsets
        self.reg_offset = opt.reg_offset  # reg_offset = True : regress local offset

        # self.reg_ind = self.hm_hp_ind + 1 if self.reg_offset else self.hm_hp_ind
        self.reg_ind = "reg" if self.reg_offset else "wh"

        # define id
        self.emb_dim = opt.reid_dim  # dataset.reid_dim = 128
        self.nID = opt.nID  # nId = 14455
        self.classifier = nn.Dense(self.emb_dim, self.nID).to_float(mstype.float16)
        self.IDLoss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)  # fix np
        self.s_det = Parameter(Tensor(-1.85 * np.ones(1), mstype.float32))
        self.s_id = Parameter(Tensor(-1.05 * np.ones(1), mstype.float32))
        # self.s_id = Tensor(-1.05 * self.ones(1, mindspore.float32))

        self.normalize = ops.L2Normalize(axis=1)
        self.greater = ops.Greater()
        self.expand_dims = ops.ExpandDims()
        self.tile = ops.Tile()
        self.multiples_1 = (1, 1, 128)
        # self.multiples_2 = (1, 1, 14455)
        self.select = ops.Select()
        self.zeros = ops.Zeros()
        self.exp = ops.Exp()
        self.squeeze = ops.Squeeze(0)
        self.TransposeGatherFeature = TransposeGatherFeature()
        self.reshape = ops.Reshape()
        self.reshape_mul = opt.batch_size * 500
        self.cast = ops.Cast()
        self.sigmoid = Sigmoid()

    def construct(self, feature, hm, reg_mask, ind, wh, reg, ids):
        """Defines the computation performed."""
        output_hm = feature["hm"]  # FocalLoss()
        output_hm = self.sigmoid(output_hm)

        hm_loss = self.crit(output_hm, hm)

        output_id = feature["feature_id"]  # SoftmaxCrossEntropyWithLogits()
        id_head = self.TransposeGatherFeature(output_id, ind)  # id_head=[1,500,128]
        # print(id_head.shape)

        # id_head = id_head[reg_mask > 0]
        cond = self.greater(reg_mask, 0)  # cond=[1,500]
        cond_cast = self.cast(cond, ms.int32)
        expand_output = self.expand_dims(cond_cast, 2)
        tile_out = self.tile(expand_output, self.multiples_1)
        tile_cast = self.cast(tile_out, ms.bool_)
        fill_zero = self.zeros(id_head.shape, mstype.float32)  # fill_zero=[1,500,128]
        id_head = self.select(tile_cast, id_head, fill_zero)  # id_head=[1,500,128]

        id_head = self.emb_scale * self.normalize(id_head)  # id_head=[1,500,128]
        # id_head = self.emb_scale * ops.L2Normalize(id_head)

        zero_input = self.zeros(ids.shape, mstype.int32)
        id_target = self.select(cond, ids, zero_input)  # id_target=[1,500]
        id_target_out = self.reshape(id_target, (self.reshape_mul,))
        # expand_output = self.expand_dims(id_target, 2)
        # tile_out = self.tile(expand_output, self.multiples_2)

        c_out = self.reshape(id_head, (self.reshape_mul, 128))
        c_out = self.cast(c_out, mstype.float16)
        id_output = self.classifier(c_out)  # id_output=[1,500,14455]
        id_output = self.cast(id_output, ms.float32)
        # id_output = self.squeeze(id_output)                                    # id_output=[500,14455]
        # id_target = self.squeeze(tile_out)                                     # id_target=[500,14455]
        id_loss = self.IDLoss(id_output, id_target_out)

        output_wh = feature["wh"]  # Regl1Loss
        wh_loss = self.crit_reg(output_wh, reg_mask, ind, wh)

        off_loss = 0
        if self.reg_offset and self.off_weight > 0:  # Regl1Loss
            output_reg = feature[self.reg_ind]
            off_loss = self.crit_reg(output_reg, reg_mask, ind, reg)

        det_loss = self.hm_weight * hm_loss + self.wh_weight * wh_loss + self.off_weight * off_loss
        loss = self.exp(-self.s_det) * det_loss + self.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)
        loss *= 0.5

        return loss

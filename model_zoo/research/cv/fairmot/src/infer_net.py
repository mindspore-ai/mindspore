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
"""infer net."""
from mindspore import dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from src.util import Sigmoid


class GatherFeat(nn.Cell):
    """gather feature."""

    def __init__(self):
        super(GatherFeat, self).__init__()
        self.expand_dims = ops.ExpandDims()
        self.gather = ops.GatherD()

    def construct(self, feat, ind):
        """gather feature."""
        dim = feat.shape[2]
        ind = self.expand_dims(ind, 2)
        shape = (ind.shape[0], ind.shape[1], dim)
        broadcast_to = ops.BroadcastTo(shape)
        ind = broadcast_to(ind)
        feat = self.gather(feat, 1, ind)
        # if mask is not None:
        #     mask = self.expand_dims(mask, 2)
        #     broadcast = ops.BroadcastTo(feat.shape)
        #     mask = broadcast_to(mask)
        #     # feat = feat[mask]
        #     # feat = feat.view(-1, dim)
        return feat


class TranposeAndGatherFeat(nn.Cell):
    """transpose and gather feature."""

    def __init__(self):
        super(TranposeAndGatherFeat, self).__init__()
        self.transpose = ops.Transpose()
        self.GatherFeat = GatherFeat()

    def construct(self, feat, ind):
        """transpose and gather feature."""
        feat = self.transpose(feat, (0, 2, 3, 1))
        feat = feat.view(feat.shape[0], -1, feat.shape[3])
        feat = self.GatherFeat(feat, ind)
        return feat


class MotDecode(nn.Cell):
    """
    Network tracking results of the decoder
    """

    def __init__(self, ltrb=False):
        super(MotDecode, self).__init__()
        self.cast = ops.Cast()
        self.concat = ops.Concat(axis=2)
        self.ltrb = ltrb
        self.topk = ops.TopK(sorted=True)
        self.div = ops.Div()
        self.GatherFeat = GatherFeat()
        self.TranposeAndGatherFeat = TranposeAndGatherFeat()
        self.select = ops.Select()
        self.zeroslike = ops.ZerosLike()
        self.equal = ops.Equal()
        self.pool = nn.MaxPool2d((3, 3), stride=1, pad_mode='same')

    def construct(self, heat, wh, K, reg):
        """
        Network tracking results of the decoder
        """
        batch, cat, height, width = heat.shape
        heat = self.cast(heat, mstype.float16)
        hmax = self.pool(heat)
        keep = self.equal(hmax, heat)
        input_x = self.zeroslike(heat)
        M = self.select(keep, heat, input_x)
        heat = self.cast(M, mstype.float32)
        topk_scores, topk_inds = self.topk(heat.view(batch, cat, -1), K)
        topk_inds = topk_inds % (height * width)
        topk_ys = self.cast(self.div(topk_inds, width), mstype.float32)
        topk_xs = self.cast((topk_inds % width), mstype.float32)
        scores, topk_ind = self.topk(topk_scores.view(batch, -1), K)
        clses = self.cast(self.div(topk_ind, K), mstype.int32)
        inds = self.GatherFeat(
            topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        ys = self.GatherFeat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        xs = self.GatherFeat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
        if reg is not None:
            reg = self.TranposeAndGatherFeat(reg, inds)
            reg = reg.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5
        wh = self.TranposeAndGatherFeat(wh, inds)
        if self.ltrb:
            wh = wh.view(batch, K, 4)
        else:
            wh = wh.view(batch, K, 2)
        clses = clses.view(batch, K, 1)
        clses = self.cast(clses, mstype.float32)
        scores = scores.view(batch, K, 1)
        if self.ltrb:
            bboxes = self.concat((xs - wh[..., 0:1],
                                  ys - wh[..., 1:2],
                                  xs + wh[..., 2:3],
                                  ys + wh[..., 3:4]))
        else:
            bboxes = self.concat((xs - wh[..., 0:1] / 2,
                                  ys - wh[..., 1:2] / 2,
                                  xs + wh[..., 0:1] / 2,
                                  ys + wh[..., 1:2] / 2))
        detections = self.concat((bboxes, scores, clses))
        return detections, inds


class InferNet(nn.Cell):
    """
    Network tracking results of the decoder
    """

    def __init__(self):
        super(InferNet, self).__init__()
        self.sigmoid = Sigmoid()
        self.l2_normalize = ops.L2Normalize(axis=1, epsilon=1e-12)
        self.mot_decode = MotDecode(ltrb=True)
        self.TranposeAndGatherFeat = TranposeAndGatherFeat()
        self.squeeze = ops.Squeeze(0)

    def construct(self, feature):
        hm = self.sigmoid(feature['hm'])
        id_feature = self.l2_normalize(feature['feature_id'])
        dets, inds = self.mot_decode(hm, feature['wh'], 500, feature['reg'])
        id_feature = self.TranposeAndGatherFeat(id_feature, inds)
        id_feature = self.squeeze(id_feature)
        return id_feature, dets

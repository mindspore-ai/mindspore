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
Decode from heads for evaluation
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from .utils import GatherFeature, TransposeGatherFeature


class NMS(nn.Cell):
    """
    Non-maximum suppression

    Args:
        kernel(int): Maxpooling kernel size. Default: 3.
        enable_nms_fp16(bool): Use float16 data for max_pool, adaption for CPU. Default: False.

    Returns:
        Tensor, heatmap after non-maximum suppression.
    """
    def __init__(self, kernel=3, enable_nms_fp16=False):
        super(NMS, self).__init__()
        self.cast = ops.Cast()
        self.dtype = ops.DType()
        self.equal = ops.Equal()
        self.Abs = P.Abs()
        self.max_pool_ = nn.MaxPool2d(kernel, stride=1, pad_mode="same")
        self.max_pool = P.MaxPoolWithArgmax(kernel_size=kernel, strides=1, pad_mode='same')
        self.enable_fp16 = enable_nms_fp16

    def construct(self, heat):
        """Non-maximum suppression"""
        dtype = self.dtype(heat)
        if self.enable_fp16:
            heat = self.cast(heat, mstype.float16)
            heat_max = self.max_pool_(heat)
            keep = self.equal(heat, heat_max)
            keep = self.cast(keep, dtype)
            heat = self.cast(heat, dtype)
        else:
            heat_max, _ = self.max_pool(heat)
            error = self.cast((heat - heat_max), mstype.float32)
            abs_error = self.Abs(error)
            abs_out = self.Abs(heat)
            error = abs_error / (abs_out + 1e-12)
            keep = P.Select()(P.LessEqual()(error, 1e-3),
                              P.Fill()(ms.float32, P.Shape()(error), 1.0),
                              P.Fill()(ms.float32, P.Shape()(error), 0.0))
        heat = heat * keep
        return heat


class GatherTopK(nn.Cell):
    """
    Gather topk features through all channels

    Args: None

    Returns:
        Tuple of Tensors, top_k scores, indexes, category ids, and the indexes in height and width direcction.
    """
    def __init__(self):
        super(GatherTopK, self).__init__()
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.topk = ops.TopK(sorted=True)
        self.cast = ops.Cast()
        self.dtype = ops.DType()
        self.gather_feat = GatherFeature()
        # The ops.Mod() operator will produce errors on the Ascend 310
        self.mod = P.FloorMod()
        self.div = ops.Div()

    def construct(self, scores, K=40):
        """gather top_k"""
        b, c, _, w = self.shape(scores)
        scores = self.reshape(scores, (b, c, -1))
        # (b, c, K)
        topk_scores, topk_inds = self.topk(scores, K)
        topk_ys = self.div(topk_inds, w)
        topk_xs = self.mod(topk_inds, w)
        # (b, K)
        topk_score, topk_ind = self.topk(self.reshape(topk_scores, (b, -1)), K)
        topk_clses = self.cast(self.div(topk_ind, K), self.dtype(scores))
        topk_inds = self.gather_feat(self.reshape(topk_inds, (b, -1, 1)), topk_ind)
        topk_inds = self.reshape(topk_inds, (b, K))
        topk_ys = self.gather_feat(self.reshape(topk_ys, (b, -1, 1)), topk_ind)
        topk_ys = self.cast(self.reshape(topk_ys, (b, K)), self.dtype(scores))
        topk_xs = self.gather_feat(self.reshape(topk_xs, (b, -1, 1)), topk_ind)
        topk_xs = self.cast(self.reshape(topk_xs, (b, K)), self.dtype(scores))
        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


class DetectionDecode(nn.Cell):
    """
    Decode from heads to gather multi-objects info.

    Args:
        net_config(edict): config info for CenterNet network.
        K(int): maximum objects number. Default: 100.
        enable_nms_fp16(bool): Use float16 data for max_pool, adaption for CPU. Default: True.

    Returns:
        Tensor, multi-objects detections.
    """
    def __init__(self, net_config, K=100, enable_nms_fp16=False):
        super(DetectionDecode, self).__init__()
        self.K = K
        self.nms = NMS(enable_nms_fp16=enable_nms_fp16)
        self.shape = ops.Shape()
        self.gather_topk = GatherTopK()
        self.half = ops.Split(axis=-1, output_num=2)
        self.add = ops.TensorAdd()
        self.concat_a2 = ops.Concat(axis=2)
        self.trans_gather_feature = TransposeGatherFeature()
        self.expand_dims = ops.ExpandDims()
        self.reshape = ops.Reshape()
        self.reg_offset = net_config.reg_offset
        self.Sigmoid = nn.Sigmoid()

    def construct(self, feature):
        """gather detections"""
        heat = feature['hm']
        heat = self.Sigmoid(heat)
        K = self.K
        b, _, _, _ = self.shape(heat)
        heat = self.nms(heat)
        scores, inds, clses, ys, xs = self.gather_topk(heat, K=K)
        ys = self.reshape(ys, (b, K, 1))
        xs = self.reshape(xs, (b, K, 1))

        wh = feature['wh']
        wh = self.trans_gather_feature(wh, inds)
        ws, hs = self.half(wh)

        if self.reg_offset:
            reg = feature['reg']
            reg = self.trans_gather_feature(reg, inds)
            reg = self.reshape(reg, (b, K, 2))
            reg_w, reg_h = self.half(reg)
            ys = self.add(ys, reg_h)
            xs = self.add(xs, reg_w)
        else:
            ys = ys + 0.5
            xs = xs + 0.5

        bboxes = self.concat_a2((xs - ws / 2, ys - hs / 2, xs + ws / 2, ys + hs / 2))
        clses = self.expand_dims(clses, 2)
        scores = self.expand_dims(scores, 2)
        detection = self.concat_a2((bboxes, scores, clses))
        return detection
